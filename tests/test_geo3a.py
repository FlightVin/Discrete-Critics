"""geo3a: E10 = (noisy_fixed, model_mix) and E11 = (noisy_resid, model_mix) in the redundant vocabulary.

Roll-ins as E6 (the clean branch does not depend on the label rule), model tokens from the current copy at
T=0.7, labels from the actual prefix (Fraction references), the executed loss against geo_codes.build called
directly, exhaustive vocabulary coverage, E6 / E8 / E9 runs against the parent commit, and preempted / resumed
E10 / E11 runs against the uninterrupted ones.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set (the e2e tests; -s shows the coverage
ranges). Subprocesses run with XLA_FLAGS=--xla_gpu_deterministic_ops=true (tests/e2e_v1_interrupt.launch).
"""
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

import train_v1 as T  # noqa: E402
from agents import geo_codes as gc  # noqa: E402
from agents.acarb import GEO_COMBOS, geo_config_errors  # noqa: E402
from e2e_v1_interrupt import chain, flag, progress_at_least  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, H, e_args, rows, run  # noqa: E402
from test_geo2b_agent import (COMBOS, Recorder, direct_loss, g_agent, leaf_hashes, loss_keys, make_batch,  # noqa: E402
                              sharpen, with_other_target)
from test_geo_codes import K_MAX, _all_prefixes, ref_masses, ref_residual  # noqa: E402

PARENT = os.environ.get('GEO3A_PARENT_COMMIT', '6551f0d')  # the commit geo3a branched from
NEW = {'E10': ('noisy_fixed', 'model_mix'), 'E11': ('noisy_resid', 'model_mix')}
MODEL_KEY = jax.random.PRNGKey(321)
ALL_K = np.arange(K_MAX + 1, dtype=np.int32)


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def binom_ok(frac, n, p=0.5):
    return abs(frac - p) < 5 * math.sqrt(p * (1 - p) / n)


def prefix_model(prefix, position):
    """Synthetic model tokens in [-10, 42] that depend on the actual prefix at position 1."""
    base = jax.random.randint(jax.random.fold_in(MODEL_KEY, position), prefix.shape[:-1], -10, 41)
    return base + (jnp.mod(prefix[..., 0], 3) if position == 1 else 0)


def check_labels_exact(ks, labels, rollin, info, mode):
    """Labels at all 3 positions against Fraction references from the actual prefix; rows, support, means."""
    tok = np.arange(gc.TOKEN_MIN, gc.TOKEN_MAX + 1)
    assert (labels >= 0).all() and np.abs(labels.sum(-1) - 1).max() <= 1e-6
    live = np.nonzero(labels > 0)[-1] + gc.TOKEN_MIN
    assert gc.TOKEN_MIN <= live.min() and live.max() <= gc.TOKEN_MAX
    assert int(info['oov_label']) == 0
    for i, k in enumerate(ks):
        for l in range(3):
            r, c, f = ref_residual(k, rollin[i], l)
            assert c == info['c'][i, l] and Fraction(int(info['rem'][i, l]), int(info['den'][i, l])) == f, (k, l)
            masses = ref_masses(mode, f)
            expected = np.zeros(gc.NUM_CLASSES)
            for off, m in zip((-1, 0, 1), masses):
                if m > 0:
                    assert gc.TOKEN_MIN <= c + off <= gc.TOKEN_MAX, (k, l, c, off)
                    expected[c + off - gc.TOKEN_MIN] += float(m)
            np.testing.assert_allclose(labels[i, l], expected, atol=1e-6, err_msg=f'k={k} l={l}')
            mean = sum(m * (c + off) for off, m in zip((-1, 0, 1), masses))
            assert mean == (c if mode == 'noisy_fixed' else r - Fraction(1, 2)), (k, l, mean)
            assert abs(float((labels[i, l] * tok).sum()) - float(mean)) < 1e-5


# ---- config -------------------------------------------------------------------------------------------
def test_combos():
    assert GEO_COMBOS[NEW['E10']] == 'E10' and GEO_COMBOS[NEW['E11']] == 'E11' and len(GEO_COMBOS) == 7
    for v, (label, rollin) in NEW.items():
        assert geo_config_errors(dict(arb_vocab='redundant', arb_label=label, arb_rollin=rollin,
                                      arb_readout='sample_mean', arb_base=10, arb_length=3)) == [], v
    hashes = {v: leaf_hashes(g_agent(v, split=False).network.params) for v in ('E6', 'E10', 'E11')}
    assert hashes['E10'] == hashes['E6'] and hashes['E11'] == hashes['E6']


# ---- a. roll-ins do not depend on the label rule -------------------------------------------------------
def test_rollin_independent_of_label_rule():
    key = jax.random.PRNGKey(5)
    k = jnp.asarray(ALL_K)
    outs = {m: gc.build(k, m, 'model_mix', key, prefix_model) for m in gc.LABEL_MODES}
    r0, i0 = np.asarray(outs['corrective'][1]), outs['corrective'][2]
    for m in ('noisy_fixed', 'noisy_resid'):
        np.testing.assert_array_equal(np.asarray(outs[m][1]), r0, err_msg=m)
        np.testing.assert_array_equal(np.asarray(outs[m][2]['model_mask']), np.asarray(i0['model_mask']))
        assert int(outs[m][2]['model_clamp']) == int(i0['model_clamp'])
    # The clean branch is the two-hot draw c + [u < rem] with build's own Bernoulli stream.
    mask, c, rem = (np.asarray(i0[x]) for x in ('model_mask', 'c', 'rem'))
    dens = jnp.array([gc.denominator(l) for l in range(2)], jnp.int32)
    bern_u = np.asarray(jax.random.randint(jax.random.split(key, 3)[0], (len(ALL_K), 2), 0, dens))
    for l in range(2):
        clean = ~mask[:, l]
        np.testing.assert_array_equal(r0[clean, l], (c[:, l] + (bern_u[:, l] < rem[:, l]))[clean])
        assert binom_ok(mask[:, l].mean(), len(ALL_K))
    # Position 1 (same prefix for every rule): the clean tokens are those of the E5 roll-in with the same key.
    e5 = np.asarray(gc.build(k, 'noisy_resid', 'corrective', key)[1])
    np.testing.assert_array_equal(r0[~mask[:, 0], 0], e5[~mask[:, 0], 0])


def test_agent_rollin_independent_of_label_rule(monkeypatch):
    batch, rng = make_batch(n=64, seed=2), jax.random.PRNGKey(4)
    out = {}
    for v in ('E6', 'E10', 'E11'):
        rec, agent = Recorder(monkeypatch), g_agent(v)
        agent.critic_loss(batch, agent.network.params, rng)
        out[v] = jax.device_get(rec.calls[0]['out'])
    for v in ('E10', 'E11'):
        np.testing.assert_array_equal(out[v][1], out['E6'][1], err_msg=v)
        np.testing.assert_array_equal(out[v][2]['model_mask'], out['E6'][2]['model_mask'], err_msg=v)


# ---- b. model branch ------------------------------------------------------------------------------------
@pytest.mark.parametrize('variant', list(NEW))
def test_model_tokens(variant, monkeypatch):
    agent = sharpen(with_other_target(g_agent(variant, temp=0.7)), 10)
    batch = make_batch(n=256, seed=4)
    rng = jax.random.PRNGKey(11)
    rec = Recorder(monkeypatch)
    _, info = agent.critic_loss(batch, agent.network.params, rng)
    assert (rec.calls[0]['label_mode'], rec.calls[0]['rollin_mode']) == NEW[variant]
    _, rollin, binfo = rec.calls[0]['out']
    rollin, mask, c = (np.asarray(x) for x in (rollin, binfo['model_mask'], binfo['c']))
    model_rng = jax.random.split(loss_keys(rng)[2], 4)[3]
    act = batch['actions'].reshape(256, -1)
    f = agent._percopy_critic(False)  # CURRENT critic (target differs)
    clamps = 0
    for l in range(2):
        prefix = rollin.copy()
        prefix[..., l:] = 0
        logits = np.asarray(f(batch['observations'], act, gc.token_to_class(jnp.asarray(prefix))))[..., l, :]
        g = np.asarray(jax.random.gumbel(jax.random.fold_in(model_rng, l), logits.shape))
        m = np.argmax(logits / 0.7 + g, -1) + gc.TOKEN_MIN
        np.testing.assert_array_equal(rollin[..., l][mask[..., l]], np.clip(m, c[..., l] - 2, c[..., l] + 2)[mask[..., l]])
        assert (np.argmax(logits + g, -1) + gc.TOKEN_MIN != m)[mask[..., l]].any()  # T=1 would differ
        assert (np.abs(rollin[..., l] - c[..., l]) <= 2).all()
        clamps += int((np.abs(m - c[..., l]) > 2)[mask[..., l]].sum())
    assert int(binfo['model_clamp']) == clamps and clamps > 0
    assert binom_ok(mask.mean(), mask.size), mask.mean()
    assert float(info['model_used_frac']) == pytest.approx(mask.mean(), abs=1e-6)
    assert float(info['model_clamp_frac']) == pytest.approx(clamps / mask.sum(), abs=1e-6)


# ---- c / d. labels from the actual (forced) prefix ------------------------------------------------------
def offset_model(k):
    """Model tokens c + d at both positions (from the actual prefix), d1, d2 cycling over {-2..2}^2."""
    i = jnp.arange(k.shape[0])
    d = (i % 5 - 2, (i // 5) % 5 - 2)

    def model(prefix, position):
        num = k - 20 * (100 * prefix[..., 0]) if position == 1 else k
        return jnp.floor_divide(num, 2000 // 10 ** position) + d[position]
    return model, np.asarray(d[0]), np.asarray(d[1])


@pytest.mark.parametrize('mode', ['noisy_fixed', 'noisy_resid'])
@pytest.mark.parametrize('callback', ['offset', 'constant'])
def test_labels_from_forced_prefix(mode, callback):
    rs = np.random.default_rng(8)
    base = np.concatenate([[0, K_MAX], 20 * rs.integers(0, 1000, 500), rs.integers(0, K_MAX + 1, 1500)])
    ks = np.repeat(base, 4).astype(np.int32)
    k = jnp.asarray(ks)
    if callback == 'offset':
        model, d1, d2 = offset_model(k)
    else:  # fixed tokens: 7 then -3 (often clamped)
        model = lambda p, l: jnp.full(p.shape[:-1], 7 if l == 0 else -3, jnp.int32)  # noqa: E731
    labels, rollin, info = gc.build(k, mode, 'model_mix', jax.random.PRNGKey(12), model)
    labels, rollin = np.asarray(labels), np.asarray(rollin)
    info = {x: np.asarray(v) for x, v in info.items()}
    assert int(info['oov_rollin']) == 0
    check_labels_exact(ks, labels, rollin, info, mode)
    mask, c = info['model_mask'], info['c']
    both = mask.all(-1)
    if callback == 'offset':
        np.testing.assert_array_equal(rollin[both, 0], c[both, 0] + d1[both])
        np.testing.assert_array_equal(rollin[both, 1], c[both, 1] + d2[both])
        assert int(info['model_clamp']) == 0
        changed = both & (d1 != 0) & (d2 != 0)
        neg = changed & (c[:, 1] < 0) & (c[:, 2] < 0)
        print(f'\n[forced] {mode}: {both.sum()} rows with both positions forced, {changed.sum()} both changed, '
              f'{neg.sum()} of those with c2 < 0 and c3 < 0')
        assert neg.sum() > 50
    else:
        np.testing.assert_array_equal(rollin[mask[:, 0], 0], np.clip(7, c[:, 0] - 2, c[:, 0] + 2)[mask[:, 0]])
        np.testing.assert_array_equal(rollin[mask[:, 1], 1], np.clip(-3, c[:, 1] - 2, c[:, 1] + 2)[mask[:, 1]])
        assert int(info['model_clamp']) > 0


# ---- e. the executed loss ------------------------------------------------------------------------------
def agent_model_token(agent, obs, act, rollin_rng):
    """The protocol's model token, written out: one Gumbel-argmax sample of the current copy at T=0.7."""
    model_rng = jax.random.split(rollin_rng, 4)[3]
    f = agent._percopy_critic(False)

    def model(prefix, position):
        logits = f(obs, act, gc.token_to_class(prefix))[..., position, :]
        g = jax.random.gumbel(jax.random.fold_in(model_rng, position), logits.shape)
        return gc.class_to_token(jnp.argmax(logits / 0.7 + g, axis=-1).astype(jnp.int32))
    return model


@pytest.mark.parametrize('variant', list(NEW))
def test_loss_uses_build_directly(variant, monkeypatch):
    agent = with_other_target(g_agent(variant))
    batch = make_batch()
    rng = jax.random.PRNGKey(3)
    rec = Recorder(monkeypatch)
    loss, info = agent.critic_loss(batch, agent.network.params, rng)
    assert len(rec.calls) == 1
    call = rec.calls[0]
    sample_rng, readout_rng, rollin_rng = loss_keys(rng)
    _, target_q, tinfo = agent.td_target(batch, sample_rng, readout_rng)
    k = np.broadcast_to(np.asarray(tinfo['target_k']), (2, 16))
    np.testing.assert_array_equal(np.asarray(call['k']), k)
    assert np.array_equal(np.asarray(call['key']), np.asarray(rollin_rng))
    assert (call['label_mode'], call['rollin_mode'], call['k_den']) == (*NEW[variant], 20000)
    labels, rollin, binfo = (jax.device_get(x) for x in call['out'])
    obs, act = batch['observations'], batch['actions'].reshape(16, -1)
    l2, r2, i2 = gc.build(jnp.asarray(k), *NEW[variant], rollin_rng, agent_model_token(agent, obs, act, rollin_rng))
    np.testing.assert_array_equal(labels, np.asarray(l2))
    np.testing.assert_array_equal(rollin, np.asarray(r2))
    assert 0 < int(binfo['model_used']) < 64 and int(binfo['oov_label']) == 0 and int(binfo['oov_rollin']) == 0
    # CE at all 3 positions: -sum q log softmax(logits), recomputed here.
    w = batch['valid'][..., -1]
    ce, _, _, _ = agent.critic_ce(obs, act, target_q, tinfo['target_k'], agent.network.params, rollin_rng, w)
    logits = np.asarray(agent._percopy_critic(False)(obs, act, gc.token_to_class(jnp.asarray(rollin))), np.float64)
    logp = logits - logits.max(-1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
    ref = -(labels * logp).sum(-1)
    assert ref.shape == (2, 16, 3)
    np.testing.assert_allclose(np.asarray(ce), ref, rtol=1e-5, atol=1e-6)
    assert float(loss) == pytest.approx(float((ref.mean(axis=(0, 2)) * w).mean()), rel=1e-5)
    # Gradient: only the critic, and the same as with labels / roll-ins held fixed (nothing flows through them).
    params = agent.network.params
    g = jax.grad(lambda p: agent.critic_loss(batch, p, rng)[0])(params)
    for name in g:
        mx = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g[name]))
        assert (mx > 0) == (name == 'modules_critic'), (name, mx)
    gd = jax.grad(lambda p: direct_loss(agent, batch, p, labels, rollin))(params)
    for (pa, x), (_, y) in zip(jax.tree_util.tree_leaves_with_path(g['modules_critic']),
                               jax.tree_util.tree_leaves_with_path(gd['modules_critic'])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-7, err_msg=jax.tree_util.keystr(pa))
    jl, _ = jax.jit(agent.critic_loss)(batch, params, rng)
    assert float(jl) == pytest.approx(float(loss), rel=1e-5)


# ---- f. coverage --------------------------------------------------------------------------------------
def test_coverage_exhaustive():
    k, z1, z2 = _all_prefixes(range(-2, 3))  # every k, z_l = c_l(actual prefix) + d_l, d_l in [-2, 2]
    assert k.shape[0] == 25 * (K_MAX + 1)
    fed = np.concatenate([np.asarray(z1), np.asarray(z2)])
    print(f'\n[coverage] roll-in tokens, all k x 5x5 prefixes: [{fed.min()}, {fed.max()}]')
    assert gc.TOKEN_MIN <= fed.min() and fed.max() <= gc.TOKEN_MAX
    prefix = jnp.stack([z1, z2, jnp.zeros_like(z1)], -1)
    for mode in ('noisy_fixed', 'noisy_resid'):
        labels, info = gc.label_and_info(k, prefix, mode)
        assert int(info['oov_label']) == 0 and float(info['oov_label_mass']) == 0.0
        tokens, masses = gc.label_support(info['c'], info['rem'], info['den'], mode)
        live = np.asarray(tokens)[np.asarray(masses) > 0]
        cls = np.nonzero(np.asarray(labels) > 0)[-1] + gc.TOKEN_MIN
        print(f'[coverage] {mode:12s} positive-mass label tokens: [{live.min()}, {live.max()}] '
              f'(from labels: [{cls.min()}, {cls.max()}])')
        assert gc.TOKEN_MIN <= live.min() and live.max() <= gc.TOKEN_MAX
        assert (cls.min(), cls.max()) == (live.min(), live.max())
    # Through build (model_mix) with models at both extremes: the clamp keeps every path inside.
    for mode in ('noisy_fixed', 'noisy_resid'):
        for val in (-99, 99):
            labels, rollin, info = gc.build(jnp.asarray(ALL_K), mode, 'model_mix', jax.random.PRNGKey(val + 100),
                                            lambda p, l, v=val: jnp.full(p.shape[:-1], v, jnp.int32))
            r = np.asarray(rollin)[:, :2]
            assert int(info['oov_label']) == 0 and int(info['oov_rollin']) == 0, (mode, val)
            assert gc.TOKEN_MIN <= r.min() and r.max() <= gc.TOKEN_MAX
            print(f'[coverage] build {mode} model={val}: roll-ins [{r.min()}, {r.max()}]')


# ---- g / h. end to end ----------------------------------------------------------------------------------
GEO = {'E6': COMBOS['E6'], 'E8': COMBOS['E8'], 'E9': COMBOS['E9'], **NEW}


def g_args(variant, cands):
    label, rollin = GEO[variant]
    return [*e_args(label, cands), '--agent.arb_vocab=redundant', f'--agent.arb_rollin={rollin}']


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """E6 / E8 / E9 here and at the parent commit; E10 / E11 uninterrupted and interrupted (pretrain, joint)."""
    d = tmp_path_factory.mktemp('geo3a')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    actor, cands = str(d / 'bc' / 'final_actor.pkl'), str(d / 'cands.npz')
    r = run(['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--probe_states=256',
             '--mode=make_probe_candidates', f'--init_actor={actor}', f'--out={cands}'], d, 'cands')
    assert r['rc'] == 0, r['lines'][-30:]
    parent = d / 'parent_src'
    tar = subprocess.run(['git', 'archive', PARENT], cwd=REPO, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(parent)
    online = [*COMMON, *ONLINE, f'--init_actor={actor}']
    jobs = {}
    for v in GEO:
        args = [*online, *g_args(v, cands)]
        jobs[v] = lambda v=v, a=args: run([*a, f'--run_dir={d / v}'], d, v)
        if v in NEW:
            interrupts = [flag('during pretrain updates', progress_at_least('pretrain', 'updates', 100)),
                          flag('during joint updates', progress_at_least('joint', 'env_steps', 2000))]
            jobs[v + 'I'] = lambda v=v, a=args, i=interrupts: chain(v + 'I', [*a, f'--run_dir={d / (v + "I")}'],
                                                                    str(d / f'{v}I.log'), str(d / f'{v}I.flag'), i)
        else:
            jobs[v + 'p'] = lambda v=v, a=args: run([*a, f'--run_dir={d / (v + "p")}'], d, v + 'p', cwd=str(parent))
    with ThreadPoolExecutor(6) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    return d, out


@pytest.mark.parametrize('name', ['E6', 'E8', 'E9'])
def test_geo2b_paths_match_parent_commit(name, runs):
    d, out = runs
    for k in (name, name + 'p'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    new, old = rows(d / name), rows(d / (name + 'p'))  # rows() drops the wall-clock fields only
    assert len(new) == len(old) and len(new) > 10
    for x, y in zip(new, old):
        assert x == y, (x['kind'], x['step'], {k: (x.get(k), y.get(k)) for k in set(x) | set(y)
                                               if x.get(k) != y.get(k)})


@pytest.mark.parametrize('name', list(NEW))
def test_interrupted_resume_matches_uninterrupted(name, runs):
    d, out = runs
    assert out[name]['rc'] == 0, out[name]['lines'][-40:]
    assert [e['rc'] for e in out[name + 'I']] == [143, 143], out[name + 'I']
    a, i = rows(d / name), rows(d / (name + 'I'))
    assert len(a) == len(i) and len(a) > 10
    for x, y in zip(a, i):
        assert x == y, (x, y)
    cfg = [x for x in a if x['kind'] == 'config'][0]['agent']
    assert (cfg['arb_label'], cfg['arb_rollin'], cfg['arb_vocab']) == (*NEW[name], 'redundant')
    for run_dir in (name, name + 'I'):
        args = json.load(open(d / run_dir / 'args.json'))
        assert (args['agent']['arb_label'], args['agent']['arb_rollin']) == NEW[name], run_dir
    tr = [x for x in a if x['kind'] in ('pretrain', 'joint')]
    assert tr and all(x['critic/oov_label'] == 0 and x['critic/oov_rollin'] == 0 for x in tr)
    assert all(0.3 < x['critic/model_used_frac'] < 0.7 for x in tr)
