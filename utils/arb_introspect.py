"""geo5b: critic-only introspection of frozen AR-B (acarb, sample_mean readout) checkpoints.

Values: u is the code scale [0, 1) (BSC: P(success); QC: u = (q + 100) / 100). Query-file reference values
and SEs are in METHOD units (BSC: u; QC: raw return) and are converted to u here. Errors are reported in u
(suffix-free keys) and, for QC, also in return units (*_raw). Tolerances are in u.

Components (all jit-compiled, no parameter is written):
  trace(agent, obs, act, key, ...)         the production readout (M samples per copy at arb_readout_temp,
      Gumbel noise of shape (num_qs, ..., M, L, C) from `key` exactly as ACARBAgent._sampled_codes), recording
      per copy and sample the classes / tokens, prefix values, p(z_l | prefix) (temperature 1, full C-way) and the
      decoded code n before the clip. `forced` fixes positions to given classes (the remaining ones are sampled
      with the SAME noise); `ablate` masks classes of tokens < 0 or > 9 before sampling (inference intervention).
  trace_summary(tr)                         noncanonical-token usage (token < 0 or > 9), true OOV, decoded OOR.
  forced_recovery(agent, obs, act, key, u_ref, se_u)   section 7A (see its docstring).
  ablation(agent, obs, act, key)            normal vs ablated readout on the same noise.
  candidate_ablation(agent, obs, cands, key)            the same on candidate sets, with argmax changes.
  method_labels / label_fit(agent, batch, key, clean_u)  section 8 (see label_fit's docstring).
  run_mode(...)                             --mode=arb_introspect driver (resumable parts).
"""
import hashlib
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from agents import geo_codes
from agents.acarb import (QC_K_DEN, ar_bin, ar_label_dists, canonical_digits, soft_cross_entropy, raw_to_u)

TOLS = (0.005, 0.01, 0.02)  # predeclared recovery tolerances (u units)
OFFSETS = (-2, -1, 0, 1, 2)  # e = 0 is the control (reference prefix)
Z_DECIDE = 1.959963984540054  # reference interval u +- z SE decides a digit iff it does not cross a boundary
UNIT_W = np.array(geo_codes.UNIT_WEIGHTS)  # (100, 10, 1): w_l in units of 0.001
POS_SCALE = np.array([0.1, 0.01, 0.001])  # 10^-l


# ---- vocabulary ----------------------------------------------------------------------------------------------
def vocab(agent):
    """(num_classes, token_min, token_max): class c is token c + token_min."""
    if agent._redundant():
        return geo_codes.NUM_CLASSES, geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX
    return agent.config['arb_base'], 0, agent.config['arb_base'] - 1


def class_tokens(agent):
    nc, lo, _ = vocab(agent)
    return jnp.arange(nc, dtype=jnp.int32) + lo


def is_qc(agent):
    return agent.config['arb_value'] == 'qc'


def to_u(agent, v):
    return raw_to_u(np.asarray(v, np.float64)) if is_qc(agent) else np.asarray(v, np.float64)


def se_to_u(agent, se):
    return np.asarray(se, np.float64) / 100.0 if is_qc(agent) else np.asarray(se, np.float64)


def check_agent(agent):
    c = agent.config
    assert c['agent_name'] == 'acarb' and c['arb_readout'] == 'sample_mean', 'introspection needs acarb sample_mean'
    assert c['arb_base'] == 10 and c['arb_length'] == 3, 'introspection assumes base 10, 3 positions'


# ---- token trace ---------------------------------------------------------------------------------------------
def _trace(agent, obs, act, key, forced_classes, forced_pos, ablate, target, with_probs):
    c = agent.config
    L, M, num_qs = c['arb_length'], c['arb_readout_samples'], c['num_qs']
    nc, lo, _ = vocab(agent)
    batch_shape = act.shape[:-1]
    gumbel = jax.random.gumbel(key, (num_qs, *batch_shape, M, L, nc))  # as ACARBAgent._sampled_codes
    fn = agent._percopy_critic(target)
    temp = c['arb_readout_temp'] if 'arb_readout_temp' in c else 1.0
    tokens_all = jnp.arange(nc, dtype=jnp.int32) + lo
    canon = (tokens_all >= 0) & (tokens_all <= 9)
    digits = jnp.zeros(gumbel.shape[:-1], jnp.int32)
    probs = []
    for l in range(L):
        logits = fn(obs, act, digits)[..., l, :]
        if with_probs:
            probs.append(jax.nn.softmax(logits, axis=-1))
        if l in forced_pos:
            z = jnp.broadcast_to(forced_classes[..., l][None, ..., None], digits.shape[:-1])
        else:
            s = logits / temp + gumbel[..., l, :]
            if ablate:
                s = jnp.where(canon, s, -jnp.inf)
            z = jnp.argmax(s, axis=-1)
        digits = digits.at[..., l].set(z.astype(jnp.int32))
    tokens = digits + lo
    n = geo_codes.decode_units(tokens)  # == ACARBAgent._decode(digits), before the clip
    out = dict(classes=digits, tokens=tokens, n=n,
               prefix_units=jnp.stack([geo_codes._prefix_units(tokens, l) if l else jnp.zeros_like(n)
                                       for l in range(L)], -1))
    if with_probs:
        out['probs'] = jnp.stack(probs, axis=-2)  # (num_qs, ..., M, L, C)
    return out


_trace_jit = jax.jit(_trace, static_argnums=(5, 6, 7, 8))


def trace(agent, obs, act, key, forced_classes=None, forced_pos=(), ablate=False, target=False, with_probs=True):
    """Production sample_mean readout with full records (see the module docstring). Shapes: classes / tokens / n /
    prefix_units (num_qs, ..., M[, L]); probs (num_qs, ..., M, L, C). Derived: value (...) = mean of clip(n, 0, 999)
    / 1000 over copies and samples; S (...) = the integer score of critic_score for the same key."""
    check_agent(agent)
    if forced_classes is None:
        forced_classes = jnp.zeros((*act.shape[:-1], agent.config['arb_length']), jnp.int32)
    out = _trace_jit(agent, obs, act, key, jnp.asarray(forced_classes, jnp.int32), tuple(forced_pos), bool(ablate),
                     bool(target), bool(with_probs))
    clipped = jnp.clip(out['n'], 0, 999)
    out['S'] = clipped.sum(axis=(0, -1))
    out['value'] = out['S'].astype(jnp.float32) / (1000 * agent.config['num_qs'] * agent.config['arb_readout_samples'])
    return out


def trace_summary(agent, tr):
    """Noncanonical-token usage per position, true OOV, decoded OOR (before the clip) with excursion size."""
    _, lo, hi = vocab(agent)
    t, n = np.asarray(tr['tokens']), np.asarray(tr['n'])
    nonc = (t < 0) | (t > 9)
    oov = (t < lo) | (t > hi)
    exc = np.where(n < 0, -n, np.where(n > 999, n - 999, 0))
    oor = exc > 0
    return dict(noncanonical_token_frac_pos=nonc.reshape(-1, t.shape[-1]).mean(0).tolist(),
                noncanonical_token_any_frac=float(nonc.any(-1).mean()),
                true_oov_count=int(oov.sum()),
                decoded_oor_frac=float(oor.mean()), decoded_oor_low_frac=float((n < 0).mean()),
                decoded_oor_high_frac=float((n > 999).mean()),
                oor_excursion_mean=float(exc[oor].mean()) if oor.any() else 0.0,
                oor_excursion_max=int(exc.max()) if exc.size else 0)


# ---- reference digits ----------------------------------------------------------------------------------------
def reference_digits(u_ref, se_u=None, z=Z_DECIDE):
    """Canonical digits (Q, 3) of floor(u_ref * 1000) (clipped to 0..999) and decidable (Q, 3): the interval
    u_ref +- z se_u gives the same floor(u * 10^l) at its two ends (se None / 0: always decidable)."""
    u = np.asarray(u_ref, np.float64)
    se = np.zeros_like(u) if se_u is None else np.asarray(se_u, np.float64)
    n = np.clip(np.floor(u * 1000 + 1e-9), 0, 999).astype(np.int64)
    digits = np.stack([n // 100, (n // 10) % 10, n % 10], -1)
    dec = np.stack([np.floor((u - z * se) * 10 ** l + 1e-9) == np.floor((u + z * se) * 10 ** l + 1e-9)
                    for l in (1, 2, 3)], -1)
    return digits, dec


# ---- forced-prefix recovery (7A) -----------------------------------------------------------------------------
def forced_recovery(agent, obs, act, key, u_ref, se_u=None, offsets=OFFSETS):
    """Section 7A. Reference digits c = canonical digits of u_ref. For position p in {1, 2} and e in `offsets`:
    position p forced to c_p + e (positions before p forced to the reference digits), later positions by the
    production readout with the same noise key for every (p, e). Unsupported (forced token outside the vocabulary:
    ordinary 0..9, redundant -22..31) -> flagged, every metric NaN; tokens are never clamped.

    Per query (arrays (Q,)): value = readout value; err = value - u_ref; err_prefix = decode(forced prefix, later
    positions at reference digits) / 1000 - u_ref (unclipped); removed = |err_prefix| - |err|;
    recov_<tol> = P(|clip(n)/1000 - u_ref| <= tol) over copies x samples; at the next position p + 1 (prefix is
    fixed there, so p(z | prefix) is exact): r = required residual (u_ref * 1000 - prefix units) / w_{p+1},
    tok_mean_err = E_p[token] - r, corrective_mass = p(|token - r| <= 1) (tokens bringing the decoded value back
    within one unit of position p + 1). digit_ok: reference digits decidable at positions 1..p+1; summaries of
    tok_mean_err / corrective_mass use digit_ok queries only, value metrics use all supported queries."""
    nc, lo, hi = vocab(agent)
    u_ref = np.asarray(u_ref, np.float64)
    ref, dec = reference_digits(u_ref, se_u)
    tok = np.asarray(class_tokens(agent), np.float64)
    out = {}
    for p in (1, 2):
        for e in offsets:
            forced_t = ref.copy()
            forced_t[:, p - 1] += e
            supported = (forced_t[:, :p] >= lo).all(-1) & (forced_t[:, :p] <= hi).all(-1)
            cls = np.where(supported[:, None], forced_t - lo, 0).astype(np.int32)  # dummy class: rows dropped
            tr = trace(agent, obs, act, key, cls, tuple(range(p)))
            n = np.asarray(tr['n'])                              # (num_qs, Q, M)
            vals = np.clip(n, 0, 999) / 1000.0
            value = vals.mean(axis=(0, 2))
            pre = (forced_t * UNIT_W).sum(-1) / 1000.0           # later positions at the reference digits
            r = (u_ref * 1000 - (forced_t[:, :p] * UNIT_W[:p]).sum(-1)) / UNIT_W[p]
            probs = np.asarray(tr['probs'])[..., 0, p, :].mean(0)  # (Q, C): same prefix for every sample
            tmean = (probs * tok).sum(-1)
            mass = (probs * (np.abs(tok[None] - r[:, None]) <= 1)).sum(-1)
            nan = lambda x: np.where(supported, x, np.nan)  # noqa: E731
            res = dict(supported=supported, digit_ok=supported & dec[:, :p + 1].all(-1),
                       value=nan(value), err=nan(value - u_ref), err_prefix=nan(pre - u_ref),
                       removed=nan(np.abs(pre - u_ref) - np.abs(value - u_ref)), r=nan(r),
                       tok_mean_err=nan(tmean - r), corrective_mass=nan(mass),
                       tokens=np.where(supported[None, :, None, None], np.asarray(tr['tokens']), 0))
            for t in TOLS:
                res[f'recov_{t}'] = nan((np.abs(vals - u_ref[None, :, None]) <= t).mean(axis=(0, 2)))
            out[f'p{p}_e{e:+d}'] = res
    return out


def forced_summary(agent, fr):
    """Per (p, e): counts and means (value metrics over supported queries, digit metrics over digit_ok)."""
    raw = 100.0 if is_qc(agent) else None
    s = {}
    for name, r in fr.items():
        sup, ok = r['supported'], r['digit_ok']
        m = lambda x, w: float(np.mean(x[w])) if w.any() else None  # noqa: E731
        d = dict(n=int(len(sup)), n_supported=int(sup.sum()), n_unsupported=int((~sup).sum()), n_digit_ok=int(ok.sum()),
                 err_mean=m(r['err'], sup), abs_err_mean=m(np.abs(r['err']), sup),
                 err_prefix_mean=m(r['err_prefix'], sup), abs_err_prefix_mean=m(np.abs(r['err_prefix']), sup),
                 removed_mean=m(r['removed'], sup),
                 tok_mean_err=m(r['tok_mean_err'], ok), abs_tok_mean_err=m(np.abs(r['tok_mean_err']), ok),
                 corrective_mass=m(r['corrective_mass'], ok))
        for t in TOLS:
            d[f'recov_{t}'] = m(r[f'recov_{t}'], sup)
        if raw and d['abs_err_mean'] is not None:
            d['abs_err_mean_raw'] = raw * d['abs_err_mean']
            d['abs_err_prefix_mean_raw'] = raw * d['abs_err_prefix_mean']
        s[name] = d
    return s


# ---- correction ablation (7B) --------------------------------------------------------------------------------
def ablation(agent, obs, act, key):
    """Normal vs ablated (tokens < 0 or > 9 masked before sampling) readout, same noise. Inference intervention:
    ablated prefixes may lie outside the training distribution. Returns per-query arrays."""
    a = trace(agent, obs, act, key, with_probs=False)
    b = trace(agent, obs, act, key, ablate=True, with_probs=False)
    return dict(value=np.asarray(a['value']), value_ablated=np.asarray(b['value']),
                S=np.asarray(a['S']), S_ablated=np.asarray(b['S']),
                tokens_ablated=np.asarray(b['tokens']), tokens=np.asarray(a['tokens']))


def candidate_ablation(agent, obs, cands, key):
    """Candidate sets (S states x N candidates): integer scores normal / ablated with the same key, argmax (first
    index on ties, as sample_actions) and whether it changes."""
    o = jnp.repeat(jnp.asarray(obs)[:, None, :], cands.shape[1], axis=1)
    a = trace(agent, o, jnp.asarray(cands), key, with_probs=False)
    b = trace(agent, o, jnp.asarray(cands), key, ablate=True, with_probs=False)
    S, Sb = np.asarray(a['S']), np.asarray(b['S'])
    den = 1000.0 * agent.config['num_qs'] * agent.config['arb_readout_samples']
    return dict(S=S, S_ablated=Sb, argmax=S.argmax(-1), argmax_ablated=Sb.argmax(-1),
                argmax_changed=S.argmax(-1) != Sb.argmax(-1), value_diff=(Sb - S) / den)


# ---- label-objective fit (8) ---------------------------------------------------------------------------------
def method_labels(agent, obs, act, target_q, target_k, rollin_rng, params=None):
    """The method's label distribution q (num_qs, Q, L, C), the roll-in prefix classes (num_qs, Q, L) and the
    teacher-forced logits (num_qs, Q, L, C), by the same code path and keys as ACARBAgent.critic_ce."""
    c = agent.config
    B, L, num_qs = c['arb_base'], c['arb_length'], c['num_qs']
    params = agent.network.params if params is None else params
    label_mode = c['arb_label'] if 'arb_label' in c else 'hard'
    sg = jax.lax.stop_gradient
    if label_mode in geo_codes.LABEL_MODES:
        k = jnp.broadcast_to(target_k, (num_qs, *target_k.shape))
        to_classes = lambda t: jnp.clip(geo_codes.token_to_class(t), 0, geo_codes.NUM_CLASSES - 1)  # noqa: E731
        model_token = None
        if c['arb_rollin'] == 'model_mix':
            model_rng = jax.random.split(rollin_rng, 4)[3]
            current = agent._percopy_critic(False, params=sg(params['modules_critic']))
            temp = c['arb_readout_temp']

            def model_token(prefix_tokens, position):
                logits = sg(current(obs, act, to_classes(prefix_tokens)))[..., position, :]
                gumbel = jax.random.gumbel(jax.random.fold_in(model_rng, position), logits.shape)
                return geo_codes.class_to_token(jnp.argmax(logits / temp + gumbel, axis=-1).astype(jnp.int32))

        q, rollin, _ = geo_codes.build(k, label_mode, c['arb_rollin'], rollin_rng, model_token, k_den=agent.label_den())
        prefix = to_classes(rollin)
        logits = agent._percopy_critic(False, params=params['modules_critic'])(obs, act, prefix)
        return q, prefix, logits
    if label_mode == 'hard' and agent._qc_grid():
        labels = canonical_digits(target_k, QC_K_DEN, B, L)
    elif label_mode == 'hard':
        labels = ar_bin(target_q, B, L)
    else:
        labels, q, _ = ar_label_dists(target_k, agent.label_den(), B, L, label_mode)
    if label_mode == 'hard':
        q = jax.nn.one_hot(labels, B)
    logits = agent.network.select('critic')(obs, act, labels, params=params)
    q = jnp.broadcast_to(q, logits.shape)
    return q, jnp.broadcast_to(labels, logits.shape[:-1]), logits


def corrective_twohot(k, k_den, prefix_tokens, lo, nc):
    """Clean corrective two-hot q* (..., L, C) of target k / k_den given prefix tokens (..., L) (exact integer
    residuals, geo_codes.residual), the exact residual r (..., L) and the mass outside the vocabulary (..., L)
    (clipped to the edge class, as geo_codes._scatter)."""
    qs, rs, oov = [], [], []
    for l in range(geo_codes.LENGTH):
        pu = geo_codes._prefix_units(prefix_tokens, l) if l else jnp.zeros_like(prefix_tokens[..., 0])
        cc, rem, den = geo_codes.residual(k, pu, l, k_den)
        f = rem.astype(jnp.float32) / den.astype(jnp.float32)
        t = jnp.stack([cc, cc + 1], -1)
        m = jnp.stack([1 - f, jnp.where(rem > 0, f, 0.0)], -1)
        out = (t < lo) | (t > lo + nc - 1)
        cls = jnp.clip(t - lo, 0, nc - 1)
        qs.append((jax.nn.one_hot(cls, nc) * m[..., None]).sum(-2))
        rs.append(cc.astype(jnp.float32) + f)
        oov.append(jnp.where(out, m, 0.0).sum(-1))
    return jnp.stack(qs, -2), jnp.stack(rs, -1), jnp.stack(oov, -1)


def _xlogy_sum(q, logp):
    return (jnp.where(q > 0, q * logp, 0.0)).sum(-1)


def dist_stats(q, logp, qstar, r, tok):
    """Per position: CE(q,p), H(q), KL(q||p), KL(q*||p), E_p[token] - r, p-mass on supp(q*)."""
    p = jnp.exp(logp)
    ce = -(q * logp).sum(-1)
    h = -_xlogy_sum(q, jnp.log(jnp.where(q > 0, q, 1.0)))
    hstar = -_xlogy_sum(qstar, jnp.log(jnp.where(qstar > 0, qstar, 1.0)))
    kl_star = -(qstar * logp).sum(-1) - hstar
    return dict(ce=ce, h=h, kl=ce - h, kl_star=kl_star, tok_mean_err=(p * tok).sum(-1) - r,
                clean_mass=(p * (qstar > 0)).sum(-1))


def label_fit(agent, batch, key, clean_u=None):
    """Section 8 on a panel batch (replay format). key is split as critic_loss splits its rng:
    (_, sample_rng, readout_rng, rollin_rng) = split(key, 4); the TD target is agent.td_target (frozen target
    critic, the agent's own code); labels by method_labels (= critic_ce). The model readout for (b) uses
    fold_in(key, 5).

    (a) target-derived prefixes: the method's roll-in; (b) model-generated prefixes: the production readout's
    sampled prefixes (num_qs x M per query); the method's q under them (redundant: label_and_info on the actual
    prefix; ordinary: prefix-independent canonical-digit labels). q*: corrective two-hot of the target k given
    the prefix. Returns per-query / per-position arrays: <a|b>_<stat> (Q, L) averaged over copies (and samples),
    y_u (Q,), tf_value (Q,) = dec(target-derived prefix) + w_L E_p[z_L], readout_value (Q,), clean_u."""
    c = agent.config
    actions = batch['actions']
    act = jnp.reshape(actions, (actions.shape[0], -1)) if c['action_chunking'] else actions[..., 0, :]
    obs = batch['observations']
    _, sample_rng, readout_rng, rollin_rng = jax.random.split(key, 4)
    y, u, info = agent.td_target(batch, sample_rng, readout_rng)
    target_k = info.get('target_k')
    nc, lo, _ = vocab(agent)
    tok = class_tokens(agent).astype(jnp.float32)
    if target_k is not None:
        k, k_den = target_k, agent.label_den()
    else:
        k, k_den = geo_codes.units_from_value(u, 10 ** 6), 10 ** 6
    y_u = k.astype(jnp.float32) / k_den
    q, prefix_cls, logits = method_labels(agent, obs, act, u, target_k, rollin_rng)
    logp = jax.nn.log_softmax(logits, -1)
    prefix_tok = prefix_cls + lo
    kb = jnp.broadcast_to(k, prefix_tok.shape[:-1])
    qstar, r, oov = corrective_twohot(kb, k_den, prefix_tok, lo, nc)
    a = dist_stats(q, logp, qstar, r, tok)
    a['qstar_oov_mass'] = oov
    w = jnp.asarray(POS_SCALE, jnp.float32)
    prefix_u = (geo_codes._prefix_units(prefix_tok, 2)).astype(jnp.float32) / 1000
    tf_value = (prefix_u + w[2] * (jnp.exp(logp[..., 2, :]) * tok).sum(-1)).mean(0)

    tr = trace(agent, obs, act, jax.random.fold_in(key, 5))
    mp = tr['tokens']                                            # (num_qs, Q, M, L)
    km = jnp.broadcast_to(k[None, :, None], mp.shape[:-1])
    label_mode = c['arb_label'] if 'arb_label' in c else 'hard'
    if label_mode in geo_codes.LABEL_MODES:
        qb, _ = geo_codes.label_and_info(km, mp, label_mode, k_den)
    else:
        qb = jnp.broadcast_to(q[:, :, None], (*mp.shape, nc))
    qstar_b, r_b, oov_b = corrective_twohot(km, k_den, mp, lo, nc)
    logp_b = jnp.log(jnp.maximum(tr['probs'], 1e-38))
    b = dist_stats(qb, logp_b, qstar_b, r_b, tok)
    b['qstar_oov_mass'] = oov_b
    out = {f'a_{n}': np.asarray(v.mean(0)) for n, v in a.items()}
    out.update({f'b_{n}': np.asarray(v.mean((0, 2))) for n, v in b.items()})
    out.update(y=np.asarray(y), y_u=np.asarray(y_u), tf_value=np.asarray(tf_value),
               readout_value=np.asarray(tr['value']), label_mode=label_mode)
    if clean_u is not None:
        out['clean_u'] = np.asarray(clean_u, np.float64)
    return out


def label_fit_summary(agent, lf):
    s = {}
    for kk, v in lf.items():
        if (kk.startswith('a_') or kk.startswith('b_')) and np.ndim(v) == 2:
            s[kk] = np.mean(v, 0).tolist()
            if kk.endswith('tok_mean_err'):
                s[kk + '_abs'] = np.mean(np.abs(v), 0).tolist()
                s[kk + '_abs_scaled'] = (np.mean(np.abs(v), 0) * POS_SCALE).tolist()  # x 10^-l: value units
    y = lf['y_u'].astype(np.float64)
    for name in ('tf_value', 'readout_value'):
        e = lf[name].astype(np.float64) - y
        s[f'{name}_err_vs_y'] = float(e.mean())
        s[f'{name}_abs_err_vs_y'] = float(np.abs(e).mean())
        if 'clean_u' in lf:
            e = lf[name].astype(np.float64) - lf['clean_u']
            s[f'{name}_err_vs_clean'] = float(e.mean())
            s[f'{name}_abs_err_vs_clean'] = float(np.abs(e).mean())
    s['label_mode'] = str(lf['label_mode'])
    return s


# ---- --mode=arb_introspect driver ----------------------------------------------------------------------------
PANEL_KEYS = ('observations', 'actions', 'rewards', 'masks', 'terminals', 'valid', 'next_observations',
              'num_chunks_to_success')


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_npz(path, arrays):
    import io
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    with open(path + '.tmp', 'wb') as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + '.tmp', path)


def flat_actions(actions):
    a = np.asarray(actions, np.float32)
    return a.reshape(a.shape[0], -1) if a.ndim == 3 else a


def chunk_actions(actions, H):
    a = np.asarray(actions, np.float32)
    return a.reshape(a.shape[0], H, -1) if a.ndim == 2 else a


def _preempted():
    flag = os.environ.get('AE_PREEMPT_FLAG')
    return bool(flag) and os.path.exists(flag)


def run_mode(agent, out_dir, ident, queries_path, candidates_path=None, chunk=256, cand_chunk=32, seed=0,
             components=('trace', 'forced', 'ablation', 'candidates', 'labelfit')):
    """Resumable driver. Parts <out_dir>/arb_introspect_parts/<component>_<i>.npz (written atomically, each with
    its ident); a restart skips parts present with the same ident and refuses others. On AE_PREEMPT_FLAG between
    parts: exit 143. Keys: fold_in(fold_in(PRNGKey(seed), component id), chunk index). Then
    <out_dir>/arb_introspect.json (summaries) and arb_introspect.npz (per-query arrays, parts concatenated)."""
    check_agent(agent)
    parts_dir = os.path.join(out_dir, 'arb_introspect_parts')
    os.makedirs(parts_dir, exist_ok=True)
    with np.load(queries_path) as z:
        qd = {kk: z[kk] for kk in z.files}
    ident = dict(ident, queries_sha256=file_sha256(queries_path), seed=seed, chunk=chunk, cand_chunk=cand_chunk)
    if candidates_path:
        ident['candidates_sha256'] = file_sha256(candidates_path)
    ident_s = json.dumps(ident, sort_keys=True, default=str)
    obs_all = np.asarray(qd['observations'], np.float32)
    act_all = flat_actions(qd['actions'])
    Q = len(obs_all)
    has_ref = 'ref_value' in qd
    u_ref = to_u(agent, qd['ref_value']) if has_ref else None
    se_u = se_to_u(agent, qd['ref_se']) if has_ref and 'ref_se' in qd else None
    has_panel = all(kk in qd for kk in PANEL_KEYS)
    base = jax.random.PRNGKey(seed)
    comp_id = dict(trace=1, forced=2, ablation=3, candidates=4, labelfit=5)
    todo = []
    for c in components:
        if c == 'forced' and not has_ref:
            print('ARB_INTROSPECT_SKIP forced (no ref_value in the query file)', flush=True)
            continue
        if c == 'labelfit' and not has_panel:
            print(f'ARB_INTROSPECT_SKIP labelfit (query file lacks {[k for k in PANEL_KEYS if k not in qd]})',
                  flush=True)
            continue
        if c == 'candidates':
            if not candidates_path:
                continue
            with np.load(candidates_path) as z:
                n_states = len(z['observations'])
            todo += [(c, i, slice(s, s + cand_chunk)) for i, s in enumerate(range(0, n_states, cand_chunk))]
            continue
        todo += [(c, i, slice(s, s + chunk)) for i, s in enumerate(range(0, Q, chunk))]
    cands = None
    H = agent.config['horizon_length']
    for c, i, sl in todo:
        path = os.path.join(parts_dir, f'{c}_{i}.npz')
        if os.path.exists(path):
            with np.load(path) as z:
                if str(z['ident']) != ident_s:
                    print(f'ARB_INTROSPECT_REFUSED {path} was written with other settings', flush=True)
                    sys.exit(2)
            print(f'ARB_INTROSPECT_PART_SKIP {c} {i}', flush=True)
            continue
        if _preempted():
            print(f'EXIT_PREEMPTED source=flag phase=arb_introspect next_part={c}_{i}', flush=True)
            sys.exit(143)
        key = jax.random.fold_in(jax.random.fold_in(base, comp_id[c]), i)
        obs, act = jnp.asarray(obs_all[sl]), jnp.asarray(act_all[sl])
        if c == 'trace':
            tr = trace(agent, obs, act, key)
            arrays = {kk: np.asarray(v) for kk, v in tr.items()}
        elif c == 'forced':
            fr = forced_recovery(agent, obs, act, key, u_ref[sl], None if se_u is None else se_u[sl])
            arrays = {f'{name}/{kk}': v for name, r in fr.items() for kk, v in r.items()}
        elif c == 'ablation':
            arrays = ablation(agent, obs, act, key)
        elif c == 'candidates':
            if cands is None:
                with np.load(candidates_path) as z:
                    cobs = np.asarray(z['observations'], np.float32)
                    cact = np.asarray(z['candidates'] if 'candidates' in z.files else z['actions'], np.float32)
                cands = (cobs, cact.reshape(*cact.shape[:2], -1))
            arrays = candidate_ablation(agent, cands[0][sl], cands[1][sl], key)
        else:
            batch = {kk: jnp.asarray(qd[kk][sl]) for kk in PANEL_KEYS}
            batch['actions'] = jnp.asarray(chunk_actions(qd['actions'][sl], H))
            clean = to_u(agent, qd['clean_value'][sl]) if 'clean_value' in qd else None
            arrays = {kk: np.asarray(v) for kk, v in label_fit(agent, batch, key, clean).items()}
        arrays['ident'] = np.asarray(ident_s)
        _write_npz(path, arrays)
        print(f'ARB_INTROSPECT_PART {c} {i}', flush=True)

    # Assemble.
    merged, summary = {}, dict(ident=ident, n_queries=Q, value_units='u in [0, 1) (QC: (q + 100) / 100)')
    for c in dict.fromkeys(t[0] for t in todo):
        idx = sorted(t[1] for t in todo if t[0] == c)
        loaded = []
        for i in idx:
            with np.load(os.path.join(parts_dir, f'{c}_{i}.npz')) as z:
                loaded.append({kk: z[kk] for kk in z.files if kk != 'ident'})
        axis = lambda kk, v: 1 if kk.split('/')[-1] in ('classes', 'tokens', 'n', 'prefix_units', 'probs',  # noqa
                                                         'tokens_ablated') else 0
        cat = {}
        for kk in loaded[0]:
            v0 = loaded[0][kk]
            if v0.ndim == 0:
                cat[kk] = v0
            else:
                cat[kk] = np.concatenate([d[kk] for d in loaded], axis=axis(kk, v0))
        merged.update({f'{c}/{kk}': v for kk, v in cat.items()})
        if c == 'trace':
            summary['trace'] = trace_summary(agent, cat)
            summary['trace']['value_mean'] = float(cat['value'].mean())
        elif c == 'forced':
            fr = {}
            for kk, v in cat.items():
                name, field = kk.split('/')
                fr.setdefault(name, {})[field] = v
            summary['forced'] = forced_summary(agent, fr)
            summary['forced_note'] = ('inference intervention on forced prefixes; key shared by every (p, e); '
                                      'tolerances in u units')
            summary['n_undecidable_pos'] = reference_digits(u_ref, se_u)[1].__invert__().sum(0).tolist()
        elif c == 'ablation':
            d = cat['value_ablated'].astype(np.float64) - cat['value']
            summary['ablation'] = dict(intervention='inference intervention: OOD prefixes possible',
                                       value_diff_mean=float(d.mean()), abs_value_diff_mean=float(np.abs(d).mean()),
                                       changed_frac=float((cat['S'] != cat['S_ablated']).mean()),
                                       ablated_noncanonical=int(((cat['tokens_ablated'] < 0) |
                                                                 (cat['tokens_ablated'] > 9)).sum()))
        elif c == 'candidates':
            summary['candidates'] = dict(intervention='inference intervention: OOD prefixes possible',
                                         n_states=int(len(cat['argmax'])),
                                         argmax_changed_frac=float(cat['argmax_changed'].mean()),
                                         abs_value_diff_mean=float(np.abs(cat['value_diff']).mean()))
        elif c == 'labelfit':
            summary['labelfit'] = label_fit_summary(agent, cat)
    _write_npz(os.path.join(out_dir, 'arb_introspect.npz'), merged)
    with open(os.path.join(out_dir, 'arb_introspect.json.tmp'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    os.replace(os.path.join(out_dir, 'arb_introspect.json.tmp'), os.path.join(out_dir, 'arb_introspect.json'))
    print('ARB_INTROSPECT ' + json.dumps({k: v for k, v in summary.items() if k in ('n_queries', 'trace', 'ablation')},
                                         default=str), flush=True)
    return summary


# ---- query files ---------------------------------------------------------------------------------------------
def make_queries(env_name, H, n, seed, out, src_run_dir=None, online_only=False, discount=0.99):
    """Query / panel npz from the processed demos (train_v1.process_demos) and, with src_run_dir, the online rows
    of that run's committed main buffer (segments read-only): n sequences sampled as the training replay samples
    them (SeqReplay.sample_sequence, np.random.Generator(PCG64(seed))). No reference values (add ref_value /
    ref_se / clean_value in method units from an MC estimate, e.g. geo5a's)."""
    from envs.env_utils import make_env_and_datasets
    from train_v1 import process_demos, read_committed_state
    from utils.replay_v1 import SeqReplay
    _, _, ds, _ = make_env_and_datasets(env_name)
    demos = process_demos(ds, env_name, H, True)
    rng = np.random.Generator(np.random.PCG64(seed))
    meta, segs = None, []
    if src_run_dir:
        state, man = read_committed_state(src_run_dir)
        meta = state['buffers']['main']
        for e in man['segments'].get('main', []):
            with np.load(os.path.join(src_run_dir, 'ckpt', 'segments', e['file'])) as z:
                segs.append((e['k_from'], e['k_to'], {kk: z[kk] for kk in z.files}))
    buf = SeqReplay(demos, meta['cap'] if meta else len(demos['observations']) + 1, H, episodic=False, rng=rng)
    if meta:
        buf.restore(meta, segs)
    if online_only:
        assert buf.num_online_starts > 0, 'no online rows'

        def online_starts(b):
            k0 = buf.k_lo + rng.integers(buf.num_online_starts, size=b)
            return buf.slot(k0), np.ones(b, bool), k0

        buf.sample_starts = online_starts
    batch = buf.sample_sequence(n, H, discount)
    arrays = {kk: np.asarray(batch[kk]) for kk in PANEL_KEYS}
    arrays['observations'] = arrays['observations'].astype(np.float32)
    arrays['actions'] = arrays['actions'].astype(np.float32)
    _write_npz(out, arrays)
    print(f'MAKE_QUERIES out={out} n={n} sha256={file_sha256(out)}', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['make_queries'])
    ap.add_argument('--env_name', default='square-mh-low_dim')
    ap.add_argument('--horizon_length', type=int, default=5)
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--src_run_dir', default=None)
    ap.add_argument('--online_only', action='store_true')
    a = ap.parse_args()
    make_queries(a.env_name, a.horizon_length, a.n, a.seed, a.out, a.src_run_dir, a.online_only)
