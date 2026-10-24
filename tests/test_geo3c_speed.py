"""geo3c: real-size joint-update throughput, QC E6 vs BSC E6 at L=3 (batch 256, default network sizes, square
shapes: obs 23, action 7 x H=5). Skipped unless GEO3C_SPEED=1 (timing only, no assertion on speed)."""
import os
import sys
import time

import jax
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import train_v1 as T  # noqa: E402
from agents import agents, v1_config  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get('GEO3C_SPEED') != '1', reason='timing only: set GEO3C_SPEED=1')
H, OBS, ADIM, N = 5, 23, 7, 256


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def batch(seed):
    rs = np.random.RandomState(seed)
    return dict(observations=rs.randn(N, OBS).astype(np.float32),
                actions=rs.uniform(-1, 1, (N, H, ADIM)).astype(np.float32),
                next_observations=rs.randn(N, H, OBS).astype(np.float32),
                terminals=np.zeros((N, H), np.float32), masks=np.ones((N, H), np.float32),
                valid=np.ones((N, H), np.float32), rewards=-np.ones((N, H), np.float32),
                num_chunks_to_success=-np.ones(N, np.float32))


@pytest.mark.parametrize('method', ['arb_qc', 'arb_bsc'])
def test_e6_updates_per_second(method):
    cfg = v1_config(method)
    cfg.update(horizon_length=H, critic_target_bon=False, arb_readout='sample_mean', arb_readout_samples=10,
               arb_readout_temp=0.7, arb_vocab='redundant', arb_label='corrective', arb_rollin='model_mix')
    agent = T.split_optimizer(agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg))
    bs = [jax.device_put(batch(i)) for i in range(4)]
    for i in range(5):  # compile + warm up
        agent, info = T.joint_update(agent, bs[i % 4], bs[(i + 1) % 4])
    jax.block_until_ready(info)
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        agent, info = T.joint_update(agent, bs[i % 4], bs[(i + 1) % 4])
    jax.block_until_ready(info)
    dt = time.perf_counter() - t0
    print(f'SPEED {method} E6 L=3 batch {N}: {n / dt:.1f} updates/s ({jax.devices()[0].device_kind})')
    assert np.isfinite(float(info['critic/critic_loss']))
