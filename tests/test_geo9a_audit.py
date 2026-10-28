"""geo9a: audit tests of the code the final runs EXECUTED (amendment 16).

The project is imported from an extracted source tree of a run commit, AUDIT_SRC (default: this repo), so the same
file runs against every tree: features absent from a tree are skipped with a reason. Worked values are appended, one
JSON object per test (and parameter), to AUDIT_OUT (if set). Tiny networks on CPU. Run as

  env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu AUDIT_SRC=$HOME/ae/audit-src/<sha> \
      AUDIT_OUT=$HOME/ae/audit-out/<sha>.jsonl python -m pytest tests/test_geo9a_audit.py -q -rsx
"""
import importlib
import inspect
import itertools
import json
import os
import sys
from fractions import Fraction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.abspath(os.environ.get('AUDIT_SRC', REPO))
SHA = os.path.basename(SRC.rstrip('/'))
OUT = os.environ.get('AUDIT_OUT')
sys.dont_write_bytecode = True  # the audit trees are read-only
sys.path.insert(0, SRC)
os.chdir(SRC)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from absl import flags  # noqa: E402
from ml_collections import config_flags  # noqa: E402

import train_v1 as T  # noqa: E402
from agents import agents, v1_config  # noqa: E402
from utils import replay_v1  # noqa: E402
from utils.flax_utils import TrainState  # noqa: E402

acarb = importlib.import_module('agents.acarb')
acbsc = importlib.import_module('agents.acbsc')
acfql = importlib.import_module('agents.acfql')
ACARBAgent, ACBSCAgent, ACFQLAgent = acarb.ACARBAgent, acbsc.ACBSCAgent, acfql.ACFQLAgent
assert os.path.abspath(T.__file__).startswith(SRC) and os.path.abspath(acarb.__file__).startswith(SRC), \
    (T.__file__, acarb.__file__, SRC)
try:
    geo_codes = importlib.import_module('agents.geo_codes')
except ImportError:
    geo_codes = None

ARB_DEFAULTS = acarb.get_config()
HAS_SAMPLE_MEAN = 'arb_readout_temp' in ARB_DEFAULTS            # 22bf78c+
HAS_REDUNDANT = 'arb_vocab' in ARB_DEFAULTS                     # 6551f0d+
GEO_COMBOS = getattr(acarb, 'GEO_COMBOS', {})                  # E5-E9 at 6551f0d, + E10/E11 at f4ca034
COMBOS = {v: k for k, v in GEO_COMBOS.items()}
HAS_QC_GRID = hasattr(acarb, 'QC_K_DEN')                         # eb39863+
ACFQL_TD1 = 'critic_target_bon' in acfql.get_config()          # 17f94d9+
ORDINARY = {'E1': 'hard', 'E2': 'twohot_last'}
OBS, ADIM = 6, 3
GAMMA = 0.99


# ---- helpers ------------------------------------------------------------------------------------------------
def _py(x):
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, Fraction):
        return f'{x.numerator}/{x.denominator}'
    if isinstance(x, (jnp.ndarray, np.ndarray, np.generic)):
        return _py(np.asarray(x).tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def record(test, **values):
    if not OUT:
        return
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(_py(dict(sha=SHA, test=test, **values))) + '\n')


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', '--horizon_length=5'])
    yield


def tiny(cfg, H=5):
    cfg.actor_hidden_dims = (32, 32)
    cfg.value_hidden_dims = (32, 32)
    if 'arb_d_model' in cfg:
        cfg.arb_d_model = 32
    cfg.flow_steps = 3
    cfg.horizon_length = H
    cfg.batch_size = 16
    return cfg


def arb_cfg(value='bsc', variant='E6', H=5, **over):
    """acarb config of the run recipes (sample_mean M=10 T=0.7, one actor next chunk) as far as the tree has it.

    variant: 'E0' (greedy, ordinary digits), 'E1' / 'E2' (ordinary labels), 'E5'-'E11' (redundant)."""
    if value == 'qc' and not HAS_QC_GRID:
        pytest.skip(f'arb_value=qc with arb_readout=sample_mean is absent at {SHA} (before eb39863)')
    cfg = tiny(v1_config('arb_qc' if value == 'qc' else 'arb_bsc'), H)
    cfg.critic_target_bon = False
    if variant != 'E0':
        if not HAS_SAMPLE_MEAN:
            pytest.skip(f'arb_readout=sample_mean is absent at {SHA} (before 22bf78c)')
        cfg.arb_readout = 'sample_mean'
        cfg.arb_readout_samples = 10
        cfg.arb_readout_temp = 0.7
        if variant in ORDINARY:
            cfg.arb_label = ORDINARY[variant]
        else:
            if variant not in COMBOS:
                pytest.skip(f'{variant} (redundant vocabulary) is absent at {SHA}')
            cfg.arb_vocab = 'redundant'
            cfg.arb_label, cfg.arb_rollin = COMBOS[variant]
    cfg.update(over)
    return cfg


def make(name, cfg, seed=0, other_target=True):
    """Agent with (optionally) a target critic different from the online critic."""
    cls = agents[name]
    ex_o, ex_a = np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32)
    agent = cls.create(seed, ex_o, ex_a, cfg)
    if other_target:
        other = cls.create(seed + 7, ex_o, ex_a, dict(agent.config))
        p = dict(agent.network.params)
        p['modules_target_critic'] = other.network.params['modules_critic']
        agent = agent.replace(network=agent.network.replace(params=p))
    return agent


def cum(r, gamma=GAMMA):
    """Per-step rewards (..., H) -> the replay's cumulative discounted rewards."""
    r = np.asarray(r, np.float64)
    return np.cumsum(r * gamma ** np.arange(r.shape[-1]), axis=-1)


def make_batch(n=16, H=5, seed=0):
    rs = np.random.RandomState(seed)
    terminals = np.maximum.accumulate((rs.rand(n, H) < 0.25).astype(np.float32), axis=1)
    masks = np.where(terminals > 0, (rs.rand(n, 1) < 0.5), 1.0).astype(np.float32)
    return dict(observations=rs.randn(n, OBS).astype(np.float32),
                actions=rs.uniform(-1, 1, (n, H, ADIM)).astype(np.float32),
                next_observations=rs.randn(n, H, OBS).astype(np.float32),
                terminals=terminals, masks=masks, valid=np.ones((n, H), np.float32),
                rewards=cum(-np.ones((n, H))).astype(np.float32),
                num_chunks_to_success=rs.randn(n).astype(np.float32))


def end_batch(cases, H):
    """Rows by case: 'success' (reward 0 at step s, terminal / mask 0 from s on), 'timeout' (terminal at the last
    step, mask 1), 'none'. cases: list of (kind, step)."""
    n = len(cases)
    rs = np.random.RandomState(1)
    r = -np.ones((n, H))
    terminals, masks = np.zeros((n, H)), np.ones((n, H))
    for i, (kind, s) in enumerate(cases):
        if kind == 'success':
            r[i, s] = 0.0
            terminals[i, s:] = 1.0
            masks[i, s:] = 0.0
        elif kind == 'timeout':
            terminals[i, s:] = 1.0
    valid = np.ones((n, H))
    valid[:, 1:] = 1.0 - terminals[:, :-1]
    return dict(observations=rs.randn(n, OBS).astype(np.float32),
                actions=rs.uniform(-1, 1, (n, H, ADIM)).astype(np.float32),
                next_observations=rs.randn(n, H, OBS).astype(np.float32),
                terminals=terminals.astype(np.float32), masks=masks.astype(np.float32),
                valid=valid.astype(np.float32), rewards=cum(r).astype(np.float32),
                num_chunks_to_success=np.where([c[0] == 'success' for c in cases], 0.0, -1.0).astype(np.float32))


def leaves_equal_zero(tree):
    return all(not np.any(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree))


# ---- 1. resolve_config ----------------------------------------------------------------------------------
_TH3_E6 = ['--mode=online', '--method=arb_qc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
           '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', '--agent.arb_vocab=redundant',
           '--agent.arb_label=corrective', '--agent.arb_rollin=model_mix', '--replay_rng=separate',
           '--env_name=tool_hang-ph-low_dim', '--horizon_length=8', '--seed=0']
_REF = ['--mode=online', '--method=qc', '--agent.critic_target_bon=False', '--replay_rng=separate',
        '--probe_candidates=$CANDS', '--init_actor=$BC', '--run_dir=$RUN', '--env_name=tool_hang-ph-low_dim',
        '--horizon_length=8', '--warmup_steps=100000', '--online_steps=4000000', '--buffer_cap=5000000', '--seed=0',
        '--pretrain_updates=50000', '--eval_interval=100000', '--n_eval_envs=32', '--eval_episodes=2']
RECIPES = {
    'geo-th3-qc-E6-model-rollin-s0': _TH3_E6,
    'geo-th3-E6-model-rollin-s0': [a.replace('=arb_qc', '=arb_bsc') for a in _TH3_E6],
    'geo-th3-ref-bsc-td1-s0': [a.replace('--method=qc', '--method=bsc') for a in _REF],
    'geo-th3-ref-qc-td1-s0': _REF,
}
RECORD_KEYS = ('agent_name', 'arb_value', 'discount', 'critic_target_bon', 'arb_label', 'arb_rollin', 'arb_vocab',
               'arb_readout', 'arb_readout_samples', 'arb_readout_temp', 'num_qs', 'q_agg', 'tau', 'lr', 'batch_size',
               'flow_steps', 'actor_num_samples', 'actor_type', 'horizon_length')


@pytest.mark.parametrize('recipe', list(RECIPES))
def test_resolve_config(recipe):
    """The agent config train_v1 builds for the recipe's command line: the tree's argv_with_method_agent, the absl
    / ml_collections --agent parsing (on a fresh FlagValues) and resolve_agent_config."""
    flags_ = RECIPES[recipe]
    argv = T.argv_with_method_agent(['train_v1.py', *flags_])
    agent_argv = [a for a in argv[1:] if a.startswith('--agent')]
    method = [a.split('=', 1)[1] for a in flags_ if a.startswith('--method=')][-1]
    fv = flags.FlagValues()
    saved = list(sys.argv)
    try:
        sys.argv[:] = ['train_v1.py', *agent_argv]  # DEFINE_config_file reads sys.argv (as in train_v1)
        config_flags.DEFINE_config_file('agent', 'agents/acbsc.py', flag_values=fv, lock_config=False)
        fv(['train_v1.py', *agent_argv])
        config = T.resolve_agent_config(method, fv.agent, argv)
    except Exception as e:  # an --agent.<key> the tree does not have
        record('resolve_config', recipe=recipe, argv=flags_, refused=f'{type(e).__name__}: {e}'[:300])
        pytest.skip(f'{recipe}: the tree cannot parse this command line ({type(e).__name__}: {str(e)[:120]})')
    finally:
        sys.argv[:] = saved
    got = {k: (config[k] if k in config else None) for k in RECORD_KEYS}
    record('resolve_config', recipe=recipe, argv=flags_, config=got)
    assert got['critic_target_bon'] is False
    if method == 'arb_qc':
        assert got['arb_value'] == 'qc' and got['discount'] == pytest.approx(0.99, abs=0)
    if method == 'arb_bsc':
        assert got['arb_value'] == 'bsc' and got['discount'] == 1.0
    if method.startswith('arb_'):
        assert (got['arb_readout'], got['arb_readout_samples'], got['arb_readout_temp']) == ('sample_mean', 10, 0.7)
        assert (got['arb_vocab'], got['arb_label'], got['arb_rollin']) == ('redundant', 'corrective', 'model_mix')
    assert (got['num_qs'], got['q_agg'], got['actor_num_samples'], got['actor_type']) == (2, 'mean', 32, 'best-of-n')


# ---- 2. seq_replay_toy ----------------------------------------------------------------------------------
def _rows(rewards, terminals, masks, base):
    n = len(rewards)
    return dict(observations=(base + np.arange(n, dtype=np.float32))[:, None],
                actions=np.zeros((n, 1), np.float32), rewards=np.asarray(rewards, np.float32),
                terminals=np.asarray(terminals, np.float32), masks=np.asarray(masks, np.float32),
                next_observations=(base + 1 + np.arange(n, dtype=np.float32))[:, None],
                num_chunks_to_timeout=np.zeros(n, np.float32), num_chunks_to_success=np.zeros(n, np.float32))


def test_seq_replay_toy(monkeypatch):
    H = 3
    demo = _rows([-1, -1, -1, -1, -1, 0], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0], 0)
    ep1 = _rows([-1, -1, -1, -1], [0, 0, 0, 1], [1, 1, 1, 1], 100)   # timeout (terminal 1, mask 1)
    ep2 = _rows([-1, -1, 0], [0, 0, 1], [1, 1, 0], 200)              # success (terminal 1, mask 0, reward 0)
    buf = replay_v1.SeqReplay(demo, capacity=64, horizon=H, episodic=False)
    for ep in (ep1, ep2):
        for i in range(len(ep['rewards'])):
            buf.add_transition({k: v[i] for k, v in ep.items()})
    total = buf.num_valid_starts
    # Every valid start exactly once: the start draw returns 0..total-1 (global numpy RNG; no rng passed).
    monkeypatch.setattr(np.random, 'randint', lambda hi, size=None: np.arange(hi)[:size])
    b = buf.sample_sequence(total, H, GAMMA)
    raw = {k: np.concatenate([demo[k], ep1[k], ep2[k]]) for k in ('rewards', 'terminals', 'observations')}
    ids = list(raw['observations'][:, 0])
    starts = []
    for j in range(total):
        i0 = ids.index(b['observations'][j, 0])
        r, term = raw['rewards'][i0:i0 + H], raw['terminals'][i0:i0 + H]
        starts.append(dict(start_obs_id=int(b['observations'][j, 0]), rewards=b['rewards'][j], masks=b['masks'][j],
                           terminals=b['terminals'][j], valid=b['valid'][j], raw_rewards=r, raw_terminals=term))
        assert (b['valid'][j, -1] == 0) == bool(term[:H - 1].any()), starts[-1]
        assert b['rewards'][j, -1] == pytest.approx(float(np.sum(GAMMA ** np.arange(H) * r)), abs=1e-12)
    record('seq_replay_toy', D=6, H=H, n_valid_starts=total, starts=starts)
    assert total == (6 - H + 1) + (7 - H + 1)


# ---- 3. bsc_target_rule ---------------------------------------------------------------------------------
BSC_CASES = [('success', 2), ('timeout', 4), ('none', 0)]  # (a) success inside the chunk, (b) timeout at last, (c)


def test_bsc_target_rule_arb(monkeypatch):
    H = 5
    batch = end_batch(BSC_CASES, H)
    if HAS_SAMPLE_MEAN:
        agent = make('acarb', arb_cfg('bsc', 'E6' if 'E6' in COMBOS else 'E1', H=H))
        assert agent.readout_den() == 20000
        S = np.full(3, 12000, np.int32)
        monkeypatch.setattr(ACARBAgent, 'critic_score', lambda self, o, a, target=False, rng=None: (
            jnp.asarray(S), jnp.zeros(S.shape, jnp.float32)))
        y, u, info = agent.td_target(batch, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
        k = np.asarray(info['target_k'])
        record('bsc_target_rule', agent='acarb', readout='sample_mean', S=S, y=y, u=u, k=k)
        np.testing.assert_array_equal(k, [19980, 0, 12000])
        np.testing.assert_allclose(np.asarray(u), [0.999, 0.0, 0.6], atol=1e-7)
    else:  # E0: greedy readout, target critic value stubbed to 0.6 per copy
        agent = make('acarb', arb_cfg('bsc', 'E0', H=H))
        monkeypatch.setattr(ACARBAgent, 'critic_value', lambda self, o, a, target=False, rng=None, readout=None: (
            jnp.full((2, o.shape[0]), 0.6, jnp.float32)))
        y, u, info = agent.td_target(batch, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
        labels = acarb.ar_bin(u, 10, 3)
        record('bsc_target_rule', agent='acarb', readout='greedy', y=y, u=u, labels=labels)
        np.testing.assert_allclose(np.asarray(u), [0.999, 0.0, 0.6], atol=1e-7)
    np.testing.assert_allclose(np.asarray(y), [1.0, 0.0, 0.6], atol=1e-7)


def _stub_select(monkeypatch, target_values, critic_values=None):
    """TrainState.select: 'target_critic' -> fixed (num_qs, batch) outputs; 'critic' -> fixed outputs if given."""
    orig = TrainState.select

    def select(self, name):
        f = orig(self, name)

        def g(*args, **kw):
            n = args[0].shape[0]
            if name == 'target_critic':
                return jnp.stack([jnp.full((n,), v, jnp.float32) for v in target_values])
            if name == 'critic' and critic_values is not None:
                return jnp.stack([jnp.full((n,), v, jnp.float32) for v in critic_values])
            return f(*args, **kw)
        return g

    monkeypatch.setattr(TrainState, 'select', select)


def test_bsc_target_rule_scalar(monkeypatch):
    """acbsc (BSC-TD1 reference): target logits with mean sigmoid 0.6; the critic is stubbed to logit 0, so the
    single-row loss is BCE(0, y) = log 2 (independent of y) -- y is read from info['target_q'] instead."""
    H = 5
    cfg = tiny(v1_config('bsc'), H)
    cfg.critic_target_bon = False
    agent = make('acbsc', cfg)
    logit = float(np.log(0.6 / 0.4))
    _stub_select(monkeypatch, [logit, logit])
    batch = end_batch(BSC_CASES, H)
    ys = []
    for i in range(3):
        row = {k: v[i:i + 1] for k, v in batch.items()}
        _, info = agent.critic_loss(row, agent.network.params, jax.random.PRNGKey(3))
        ys.append(float(info['target_q']))
    record('bsc_target_rule', agent='acbsc', target_logit=logit, y=ys)
    np.testing.assert_allclose(ys, [1.0, 0.0, 0.6], atol=1e-6)


# ---- 4. qc_target_rule ----------------------------------------------------------------------------------
QC_H = 8
QC_CASES = [('none', 0), ('timeout', 7), ('success', 7)]
QC_EXPECT = [-sum(GAMMA ** i for i in range(8)) - GAMMA ** 8 * 10] * 2 + [-sum(GAMMA ** i for i in range(7))]


def test_qc_target_rule_arb(monkeypatch):
    agent = make('acarb', arb_cfg('qc', 'E6' if 'E6' in COMBOS else 'E1', H=QC_H))
    S = np.full(3, 18000, np.int32)  # Qbar = 100 S / 20000 - 100 = -10
    monkeypatch.setattr(ACARBAgent, 'critic_score', lambda self, o, a, target=False, rng=None: (
        jnp.asarray(S), jnp.zeros(S.shape, jnp.float32)))
    y, u, info = agent.td_target(end_batch(QC_CASES, QC_H), jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    y, k = np.asarray(y), np.asarray(info['target_k'])
    ex_y = np.asarray([-48.7, -48.65, 0.0, -100.5], np.float32)
    ex_k, ex_lo, ex_hi = (np.asarray(x) for x in acarb.qc_units(jnp.asarray(ex_y)))
    record('qc_target_rule', agent='acarb', S=S, y=y, expect=QC_EXPECT, k=k, u=u,
           qc_units=dict(y=ex_y, k=ex_k, lo=ex_lo, hi=ex_hi))
    np.testing.assert_allclose(y, QC_EXPECT, atol=1e-4)
    np.testing.assert_array_equal(k, [int(acarb.qc_units(jnp.float32(v))[0]) for v in y])
    np.testing.assert_array_equal(ex_k, [513000, 513500, 999000, 0])
    np.testing.assert_array_equal(ex_hi, [False, False, True, False])
    np.testing.assert_array_equal(ex_lo, [False, False, False, True])


def test_qc_target_rule_scalar(monkeypatch):
    if not ACFQL_TD1:
        pytest.skip(f'acfql critic_target_bon is absent at {SHA} (before 17f94d9)')
    cfg = tiny(v1_config('qc'), QC_H)
    cfg.critic_target_bon = False
    agent = make('acfql', cfg)
    calls = []
    orig = ACFQLAgent.compute_flow_actions.__wrapped__

    def rec(self, observations, noises):
        calls.append((tuple(observations.shape), tuple(noises.shape)))
        return orig(self, observations, noises)

    monkeypatch.setattr(ACFQLAgent, 'compute_flow_actions', rec)
    monkeypatch.setattr(ACFQLAgent, 'sample_actions', lambda *a, **k: pytest.fail('best-of-n in the TD target'))
    _stub_select(monkeypatch, [-8.0, -12.0], critic_values=[0.0, 0.0])  # mean -10, min -12
    batch = end_batch(QC_CASES, QC_H)
    ys = []
    for i in range(3):
        row = {k: v[i:i + 1] for k, v in batch.items()}
        loss, _ = agent.critic_loss(row, agent.network.params, jax.random.PRNGKey(3))
        ys.append(-float(np.sqrt(float(loss))))  # critic 0: loss = y^2 (y < 0 here)
    record('qc_target_rule', agent='acfql', target_copies=[-8.0, -12.0], y=ys, expect=QC_EXPECT, flow_calls=calls)
    np.testing.assert_allclose(ys, QC_EXPECT, atol=1e-4)
    assert calls == [((1, OBS), (1, ADIM * QC_H))] * 3, calls


# ---- 5. labels_and_paths --------------------------------------------------------------------------------
BSC_US = [0.513, 0.555, 0.6374]
LABEL_VARIANTS = ['E1', 'E2', 'E5', 'E6', 'E7', 'E10', 'E11']


def _dist(row, tokens):
    return {int(t): float(p) for t, p in zip(tokens, np.asarray(row)) if p > 0}


def _ordinary_labels(variant, k, k_den, grid):
    if variant == 'E1':
        if grid == 'qc':
            d = acarb.canonical_digits(jnp.int32(k), k_den, 10, 3)
        else:
            d = acarb.ar_bin(jnp.float32(k) / k_den, 10, 3)  # the BSC hard label: ar_bin(target_q)
        return np.asarray(d), np.asarray(jax.nn.one_hot(d, 10)), np.zeros(3, bool)
    d, q, clip = acarb.ar_label_dists(jnp.int32(k), k_den, 10, 3, ORDINARY[variant])
    return np.asarray(d), np.asarray(q), np.asarray(clip)


def _rollin_options(mode, c, rem, den):
    """[(token, exact clean prob, via)] of the roll-in at one position given its residual."""
    f = Fraction(int(rem), int(den))
    clean = [(int(c), 1 - f), (int(c) + 1, f)]
    if mode == 'corrective':
        return [(z, p, 'T(r)') for z, p in clean if p > 0]
    if mode == 'fixed':
        return [(int(c) + d, Fraction(1, 5), 'delta') for d in range(-2, 3)]
    out = {z: [p / 2, 'T(r)'] for z, p in clean if p > 0}  # model_mix: 0.5 T(r) + 0.5 clip(model, c +- 2)
    for d in range(-2, 3):
        out.setdefault(int(c) + d, [Fraction(0), 'model'])
        if out[int(c) + d][1] == 'T(r)':
            out[int(c) + d][1] = 'T(r)+model'
    return [(z, p, via) for z, (p, via) in sorted(out.items())]


def _redundant_paths(variant, k, k_den):
    label, rollin = COMBOS[variant]
    tokens = np.arange(geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX + 1)
    paths = []
    c0, r0, d0 = (int(x) for x in geo_codes.residual(jnp.int32(k), 0, 0, k_den))
    for z0, p0, v0 in _rollin_options(rollin, c0, r0, d0):
        c1, r1, d1 = (int(x) for x in geo_codes.residual(jnp.int32(k), 100 * z0, 1, k_den))
        for z1, p1, v1 in _rollin_options(rollin, c1, r1, d1):
            labels, info = geo_codes.label_and_info(jnp.int32(k), jnp.asarray([z0, z1, 0], jnp.int32), label, k_den)
            labels = np.asarray(labels)
            np.testing.assert_allclose(labels.sum(-1), 1.0, atol=1e-6)
            assert int(info['oov_label']) == 0 and float(info['oov_label_mass']) == 0.0, (variant, k, z0, z1)
            in_vocab = geo_codes.TOKEN_MIN <= z0 <= geo_codes.TOKEN_MAX and geo_codes.TOKEN_MIN <= z1 <= \
                geo_codes.TOKEN_MAX
            paths.append(dict(prefix=[z0, z1], via=[v0, v1], p_clean=p0 * p1 if rollin != 'model_mix' else None,
                              p_clean_terms=[p0, p1], prefix_in_vocab=in_vocab,
                              c=np.asarray(info['c']), rem=np.asarray(info['rem']), den=np.asarray(info['den']),
                              labels=[_dist(labels[l], tokens) for l in range(3)]))
    if rollin != 'model_mix':
        assert sum(p['p_clean'] for p in paths) == 1
    return paths


@pytest.mark.parametrize('variant', LABEL_VARIANTS)
def test_labels_and_paths(variant):
    if not HAS_SAMPLE_MEAN or geo_codes is None:
        pytest.skip(f'integer-grid labels (ar_label_dists / geo_codes) are absent at {SHA} (before 22bf78c)')
    if variant not in ORDINARY and variant not in COMBOS:
        pytest.skip(f'{variant} (redundant vocabulary) is absent at {SHA}')
    values = [('bsc', u, 20000, int(round(u * 20000))) for u in BSC_US]
    if HAS_QC_GRID:
        values.append(('qc', -48.7, acarb.QC_K_DEN, int(acarb.qc_units(jnp.float32(-48.7))[0])))
    if variant == 'E2':
        values.append(('bsc', 0.10995, 20000, 2199))
    out = []
    for grid, v, k_den, k in values:
        if variant in ORDINARY:
            d, q, clip = _ordinary_labels(variant, k, k_den, grid)
            np.testing.assert_allclose(q.sum(-1), 1.0, atol=1e-6)
            assert q.shape == (3, 10)
            out.append(dict(grid=grid, value=v, k=k, k_den=k_den, rollin='canonical', paths=[dict(
                prefix=d[:2], p_clean=Fraction(1), labels=[_dist(q[l], range(10)) for l in range(3)], clip=clip)]))
        else:
            out.append(dict(grid=grid, value=v, k=k, k_den=k_den, rollin=COMBOS[variant][1],
                            label=COMBOS[variant][0], paths=_redundant_paths(variant, k, k_den)))
    record('labels_and_paths', variant=variant, values=out)
    if HAS_QC_GRID:
        assert out[3]['k'] == 513000
    if variant == 'E2':
        e = out[-1]['paths'][0]  # u = 0.10995: canonical 1, 0, 9; r_3 = 9.95 -> clamped to class 9
        assert list(e['prefix']) == [1, 0] and e['labels'][2] == {9: 1.0} and bool(e['clip'][2])


# ---- 6. model_mix_rollin --------------------------------------------------------------------------------
class Recorder:
    def __init__(self, monkeypatch, const=None):
        self.calls = []
        orig = geo_codes.build

        def build(k, label_mode, rollin_mode, key, model_token=None, k_den=geo_codes.K_DEN):
            out = const if const is not None else orig(k, label_mode, rollin_mode, key, model_token, k_den)
            self.calls.append(dict(k=k, out=out))
            return out

        monkeypatch.setattr(geo_codes, 'build', build)


@pytest.mark.parametrize('value,variant', [('bsc', 'E6'), ('qc', 'E10'), ('qc', 'E6')])
def test_model_mix_rollin(value, variant, monkeypatch):
    if variant not in COMBOS:
        pytest.skip(f'{variant} (model_mix roll-in) is absent at {SHA}')
    agent = make('acarb', arb_cfg(value, variant))
    n = 2048  # 2 copies x 2048 elements x 2 positions = 8192 roll-in positions
    batch = make_batch(n=n, seed=4)
    rng = jax.random.PRNGKey(11)
    loss_fn = lambda p: agent.critic_loss(batch, p, rng)[0]  # noqa: E731
    rec = Recorder(monkeypatch)
    g1 = jax.grad(loss_fn)(agent.network.params)
    labels, rollin, binfo = rec.calls[0]['out']
    rollin, mask = np.asarray(rollin), np.asarray(binfo['model_mask'])
    c, rem, den = (np.asarray(binfo[x])[..., :2] for x in ('c', 'rem', 'den'))
    model_frac = float(mask.mean())
    in_clamp = bool(np.all(np.abs(rollin[..., :2] - c)[mask] <= 2))
    clean = ~mask
    dz = (rollin[..., :2] - c)[clean]
    f = (rem / den)[clean]
    clean_support = bool(np.all((dz == 0) | ((dz == 1) & (rem[clean] > 0))))
    # E[z - c] on the clean branch = E[f] (T(r) draws c + Bernoulli(f)); standard error of the difference.
    se = float(np.sqrt(np.mean(f * (1 - f)) / clean.sum()))
    # Stop-gradient: the same loss with the roll-in / labels fed as constants (no model call) has the same grad.
    const = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)), rec.calls[0]['out'])
    Recorder(monkeypatch, const=const)
    g2 = jax.grad(loss_fn)(agent.network.params)
    diff = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
               for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2)))
    record('model_mix_rollin', value=value, variant=variant, positions=int(mask.size), model_frac=model_frac,
           model_clamp=int(binfo['model_clamp']), in_clamp=in_clamp, clean_support_ok=clean_support,
           clean_mean_dz=float(dz.mean()), clean_mean_f=float(f.mean()), clean_se=se, grad_max_abs_diff=diff)
    assert abs(model_frac - 0.5) <= 0.03
    assert in_clamp and clean_support
    assert abs(float(dz.mean()) - float(f.mean())) <= 4 * se + 1e-9
    assert diff == 0.0


# ---- 7. readout_vs_enumeration ----------------------------------------------------------------------------
def _exact_readout(agent, obs, act, temp):
    """Exact E[clip(code, 0, 999)] / 1000 per copy at temperature temp, by enumerating every sequence: prefixes
    (C^2) in one teacher-forced pass, then the last position's distribution."""
    C = acarb.geo_codes.NUM_CLASSES if agent.config.get('arb_vocab', 'ordinary') == 'redundant' else 10
    pre = np.array(list(itertools.product(range(C), repeat=2)), np.int32)  # (C^2, 2)
    digits = np.concatenate([pre, np.zeros((len(pre), 1), np.int32)], -1)
    logits = np.asarray(agent.network.select('critic')(obs, act, jnp.asarray(digits)), np.float64)  # (Q, K, 3, C)
    z = logits / temp
    logp = z - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - z.max(-1, keepdims=True)
    K = np.arange(len(pre))
    lp_prefix = logp[:, K, 0, pre[:, 0]] + logp[:, K, 1, pre[:, 1]]  # (Q, K)
    tok = (lambda x: x + acarb.geo_codes.TOKEN_MIN) if C != 10 else (lambda x: x)
    last = tok(np.arange(C))
    vals = np.clip(100 * tok(pre[:, 0])[:, None] + 10 * tok(pre[:, 1])[:, None] + last[None, :], 0, 999)  # (K, C)
    e = (np.exp(lp_prefix) * (np.exp(logp[:, :, 2, :]) * vals[None]).sum(-1)).sum(-1)  # (Q,)
    return e / 1000.0


@pytest.mark.parametrize('variant', ['E1', 'E6'])
def test_readout_vs_enumeration(variant):
    agent = make('acarb', arb_cfg('bsc', variant), other_target=False)
    rs = np.random.RandomState(5)
    obs, act = rs.randn(OBS).astype(np.float32), rs.uniform(-1, 1, ADIM * 5).astype(np.float32)
    exact = _exact_readout(agent, obs, act, 0.7)
    N = 4000  # independent readout calls (one key, N copies of the same (s, a): independent noise per element)
    S, oor = agent.critic_score(jnp.broadcast_to(obs, (N, OBS)), jnp.broadcast_to(act, (N, act.size)),
                                rng=jax.random.PRNGKey(9))
    S = np.asarray(S)
    assert S.dtype == np.int32
    vals = S / agent.readout_den()  # each call's value is on the 1/20000 grid
    mean, se = float(vals.mean()), float(vals.std() / np.sqrt(N))
    record('readout_vs_enumeration', variant=variant, exact_per_copy=exact, exact=float(exact.mean()),
           sampled_mean=mean, sampled_se=se, n_calls=N, M=10, temp=0.7, readout_den=agent.readout_den(),
           S_first=S[:20], oor_frac=float(np.asarray(oor).mean()))
    assert abs(mean - float(exact.mean())) <= 5 * se + 1e-9


def test_greedy_readout_e0():
    agent = make('acarb', arb_cfg('bsc', 'E0'), other_target=False)
    rs = np.random.RandomState(5)
    obs, act = rs.randn(2, OBS).astype(np.float32), rs.uniform(-1, 1, (2, ADIM * 5)).astype(np.float32)
    q = np.asarray(agent.critic_value(obs, act, readout='greedy'))  # (num_qs, 2)
    # Manual greedy decode, each copy on its own prefix: argmax of the teacher-forced logits, position by position.
    per = [jnp.zeros((2, 3), jnp.int32) for _ in range(agent.config['num_qs'])]
    for l in range(3):
        per = [p.at[:, l].set(jnp.argmax(agent.network.select('critic')(obs, act, p)[i][:, l], -1)) for i, p in
               enumerate(per)]
    manual = np.stack([np.asarray(acarb.ar_decode(p, 10)) for p in per])
    record('greedy_readout_e0', q=q, manual=manual, digits=[np.asarray(p) for p in per], agg=q.mean(0))
    np.testing.assert_allclose(q, manual, atol=1e-6)


def test_bo32_ties_first_index(monkeypatch):
    """Scores [3, 7, 7, 1] over 4 candidates: the chosen candidate is index 1."""
    sm = HAS_SAMPLE_MEAN
    agent = make('acarb', arb_cfg('bsc', 'E1' if sm else 'E0', actor_num_samples=4), other_target=False)
    N = 4
    cands = jnp.asarray(np.arange(2 * N * ADIM * 5, dtype=np.float32).reshape(2, N, ADIM * 5))
    scores = np.array([[3, 7, 7, 1], [3, 7, 7, 1]])
    monkeypatch.setattr(ACARBAgent, '_candidates', lambda self, o, rng: (jnp.repeat(o[:, None], N, 1), cands))
    if sm:
        monkeypatch.setattr(ACARBAgent, 'critic_score',
                            lambda self, o, a, target=False, rng=None: (jnp.asarray(scores, jnp.int32), None))
    else:
        monkeypatch.setattr(ACARBAgent, 'critic_value', lambda self, o, a, target=False, rng=None, readout=None: (
            jnp.broadcast_to(jnp.asarray(scores, jnp.float32) / 10, (2, 2, N))))
    out = np.asarray(ACARBAgent.sample_actions.__wrapped__(agent, jnp.zeros((2, OBS)), rng=jax.random.PRNGKey(0)))
    idx = [int(np.argmin(np.abs(np.asarray(cands[i]) - out[i]).sum(-1))) for i in range(2)]
    record('bo32_ties', scores=scores[0], chosen=idx, readout='sample_mean' if sm else 'greedy')
    assert idx == [1, 1]


# ---- 8. grad_isolation ------------------------------------------------------------------------------------
def _agent_for(kind):
    if kind == 'acarb_bsc':
        return make('acarb', arb_cfg('bsc', 'E6' if 'E6' in COMBOS else ('E1' if HAS_SAMPLE_MEAN else 'E0')))
    if kind == 'acarb_qc':
        return make('acarb', arb_cfg('qc', 'E6' if 'E6' in COMBOS else 'E1'))
    cfg = tiny(v1_config('bsc' if kind == 'acbsc' else 'qc'))
    if kind == 'acbsc' or ACFQL_TD1:
        cfg.critic_target_bon = False
    return make(kind, cfg)


def _is_critic(module):
    return 'critic' in module


@pytest.mark.parametrize('kind', ['acarb_bsc', 'acarb_qc', 'acbsc', 'acfql'])
def test_grad_isolation(kind):
    agent = _agent_for(kind)
    batch = make_batch(n=16, seed=2)
    params = agent.network.params
    rng = jax.random.PRNGKey(4)
    gc = jax.grad(lambda p: agent.critic_loss(batch, p, rng)[0])(params)
    ga = jax.grad(lambda p: agent.actor_loss(batch, p, rng)[0])(params)
    critic_on_actor = {m: leaves_equal_zero(g) for m, g in gc.items() if not _is_critic(m)}
    actor_on_critic = {m: leaves_equal_zero(g) for m, g in ga.items() if _is_critic(m)}
    critic_nonzero = not leaves_equal_zero(gc['modules_critic'])
    actor_nonzero = not leaves_equal_zero(ga['modules_actor_bc_flow'])
    # Target update: tau * pre-step critic + (1 - tau) * pre-step target, for agent.update and T.joint_update.
    tau = agent.config['tau']
    want = jax.tree_util.tree_map(lambda c, t: tau * c + (1 - tau) * t, params['modules_critic'],
                                  params['modules_target_critic'])
    errs = {}
    for name, upd in (('agent.update', lambda a: a.update(batch)),
                      ('train_v1.joint_update', lambda a: T.joint_update(a, batch, batch))):
        a = T.split_optimizer(agent) if name.startswith('train_v1') else agent
        new, _ = upd(a)
        got = new.network.params['modules_target_critic']
        errs[name] = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                         for x, y in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)))
        moved = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(
            jax.tree_util.tree_leaves(new.network.params['modules_critic']),
            jax.tree_util.tree_leaves(params['modules_critic'])))
        errs[name + '/critic_moved'] = moved
    record('grad_isolation', kind=kind, tau=tau, critic_grad_on_actor_zero=critic_on_actor,
           actor_grad_on_critic_zero=actor_on_critic, critic_grad_nonzero=critic_nonzero,
           actor_grad_nonzero=actor_nonzero, target_update_max_abs_err=errs)
    assert all(critic_on_actor.values()) and critic_on_actor
    assert all(actor_on_critic.values()) and actor_on_critic
    assert critic_nonzero and actor_nonzero
    for name in ('agent.update', 'train_v1.joint_update'):
        assert errs[name + '/critic_moved'] > 0
        assert errs[name] <= 1e-6, errs


# ---- 9. actor_flow ----------------------------------------------------------------------------------------
def _euler_np(agent, obs, noise):
    x = np.asarray(noise, np.float32)
    steps = agent.config['flow_steps']
    for i in range(steps):
        t = np.full((*obs.shape[:-1], 1), i / steps, np.float32)
        v = np.asarray(agent.network.select('actor_bc_flow')(obs, x, t, is_encoded=True), np.float32)
        x = (x + v / np.float32(steps)).astype(np.float32)
    return np.clip(x, -1, 1)


@pytest.mark.parametrize('kind', ['acarb_bsc', 'acbsc'])
def test_actor_flow(kind):
    agent = _agent_for(kind)
    B, N, adim = 3, agent.config['actor_num_samples'], ADIM * 5
    rs = np.random.RandomState(8)
    obs = rs.randn(B, OBS).astype(np.float32)
    key = jax.random.PRNGKey(21)
    # Candidate noise as each agent's sample_actions draws it.
    cand_key = jax.random.split(key)[0] if kind.startswith('acarb') else key
    noise = np.asarray(jax.random.normal(cand_key, (B, N, adim)))
    obs_rep = np.repeat(obs[:, None], N, 1)
    euler = _euler_np(agent, obs_rep, noise)
    flow = np.asarray(agent.compute_flow_actions(obs_rep, noise))
    flow_diff = float(np.max(np.abs(flow - euler)))
    out = np.asarray(agent.sample_actions(obs, rng=key))
    match = np.abs(euler - out[:, None]).max(-1)  # (B, N)
    chosen, chosen_diff = match.argmin(-1), float(match.min(-1).max())
    # BC loss with x0 / t from the actor key's split(rng, 3)[1:].
    batch = make_batch(n=8, seed=3)
    batch['valid'][0, 3:] = 0.0  # one row with invalid steps
    rng = jax.random.PRNGKey(31)
    loss = float(agent.actor_loss(batch, agent.network.params, rng)[0])
    _, x_rng, t_rng = jax.random.split(rng, 3)
    x1 = batch['actions'].reshape(8, -1)
    x0 = np.asarray(jax.random.normal(x_rng, x1.shape))
    t = np.asarray(jax.random.uniform(t_rng, (8, 1)))
    xt = (1 - t) * x0 + t * x1
    v = np.asarray(agent.network.select('actor_bc_flow')(batch['observations'], xt, t))
    ref = float(np.mean(((v - (x1 - x0)) ** 2).reshape(8, 5, ADIM) * batch['valid'][..., None]))
    record('actor_flow', kind=kind, flow_steps=agent.config['flow_steps'], flow_vs_numpy_max_abs=flow_diff,
           sample_actions_vs_euler_candidate_max_abs=chosen_diff, chosen=chosen, out_min=float(out.min()),
           out_max=float(out.max()), bc_loss=loss, bc_loss_numpy=ref)
    assert flow_diff <= 1e-5 and chosen_diff <= 1e-5
    assert out.min() >= -1 and out.max() <= 1
    assert loss == pytest.approx(ref, rel=1e-5)


# ---- 10. jax_rng ----------------------------------------------------------------------------------------
def test_jax_rng():
    k = jax.random.PRNGKey(0)
    s4, s3 = np.asarray(jax.random.split(k, 4)), np.asarray(jax.random.split(k, 3))
    part = getattr(jax.config, 'jax_threefry_partitionable', None)
    in3 = any(np.array_equal(s4[3], x) for x in s3)
    prefix_stable = all(np.array_equal(s4[i], s3[i]) for i in range(3))
    record('jax_rng', jax_version=jax.__version__, threefry_partitionable=part, split4_3_in_split3=in3,
           split_prefix_stable=prefix_stable)
    assert not in3
