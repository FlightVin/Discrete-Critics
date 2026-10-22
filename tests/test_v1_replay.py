"""Replay buffer invariants (utils/replay_v1.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import Dataset  # noqa: E402
from utils.replay_v1 import FIELDS, SeqReplay  # noqa: E402

OBS, ACT, H = 4, 2, 5


def make_demos(lengths, start_id=0):
    """Demo episodes; observations[:, 0] is a unique row id, every episode succeeds."""
    n = sum(lengths)
    d = {
        'observations': np.zeros((n, OBS), np.float32),
        'next_observations': np.zeros((n, OBS), np.float32),
        'actions': np.random.RandomState(0).uniform(-1, 1, (n, ACT)).astype(np.float32),
        'rewards': np.full(n, -1.0, np.float32),
        'terminals': np.zeros(n, np.float32),
        'masks': np.ones(n, np.float32),
        'num_chunks_to_timeout': np.ones(n, np.float32),
        'num_chunks_to_success': np.ones(n, np.float32),
    }
    d['observations'][:, 0] = np.arange(n) + start_id
    d['next_observations'][:, 0] = np.arange(n) + start_id + 0.5
    ends = np.cumsum(lengths) - 1
    d['terminals'][ends] = 1
    d['masks'][ends] = 0
    d['rewards'][ends] = 0
    return d


def row(i, terminal=False, success=False):
    return dict(observations=np.array([i, 1, 2, 3], np.float32), actions=np.full(ACT, 0.1, np.float32),
                rewards=0.0 if success else -1.0, terminals=float(terminal), masks=0.0 if success else 1.0,
                next_observations=np.array([i + 0.5, 1, 2, 3], np.float32), num_chunks_to_timeout=1.0,
                num_chunks_to_success=-1.0)


def check_sequences(buf, n_samples=4000):
    """Every sampled sequence is H contiguous held rows (ids step by one), demo or online, never mixed."""
    batch = buf.sample_sequence(n_samples, H, 0.99)
    ids = batch['full_observations'][..., 0]
    assert np.all(np.diff(ids, axis=1) == 1), 'sequence is not contiguous in write order'
    demo = ids[:, 0] < DEMO_ID_LIMIT
    assert np.all(ids[demo, -1] < buf.D), 'demo sequence runs past the demos'
    online_k = ids[~demo] - ONLINE_ID0
    assert np.all(online_k[:, 0] >= buf.k_lo) and np.all(online_k[:, -1] <= buf.n - 1), 'crosses the write head'
    assert np.array_equal(batch['observations'][:, 0], ids[:, 0])
    return ids


DEMO_ID_LIMIT = 10 ** 6
ONLINE_ID0 = 10 ** 7  # online row k has id ONLINE_ID0 + k


def test_demo_only_matches_dataset_sample_sequence():
    demos = make_demos([30, 40, 25])
    buf = SeqReplay(demos, capacity=200, horizon=H, episodic=False)
    ds = Dataset.create(**{k: demos[k] for k in FIELDS})
    np.random.seed(3)
    a = ds.sample_sequence(512, sequence_length=H, discount=0.99)
    np.random.seed(3)
    b = buf.sample_sequence(512, H, 0.99)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_main_ring_demos_protected_and_no_head_crossing():
    np.random.seed(0)
    demos = make_demos([30, 40, 25])
    D = 95
    buf = SeqReplay(demos, capacity=D + 37, horizon=H, episodic=False)
    snapshot = {k: v[:D].copy() for k, v in buf.data.items()}
    total = 3 * buf.cap + 11  # well past capacity
    for k in range(total):
        buf.add_transition(row(ONLINE_ID0 + k, terminal=(k % 13 == 12)))
        if k % 7 == 0 or k < 2 * H:
            assert buf.num_valid_starts == (D - H + 1) + max(0, buf.n - buf.k_lo - H + 1)
            if buf.n >= 1:
                check_sequences(buf, 500)
    for k in FIELDS:
        assert np.array_equal(buf.data[k][:D], snapshot[k]), f'demo field {k} was overwritten'
    assert buf.n == total and buf.k_lo == total - buf.R and buf.overwrites == total - buf.R
    # Held online rows are exactly the newest R, in their ring slots.
    held = buf.data['observations'][buf.slot(np.arange(buf.k_lo, buf.n)), 0] - ONLINE_ID0
    assert np.array_equal(held, np.arange(buf.k_lo, buf.n))
    # Valid starts never include a start whose sequence would wrap over the head.
    starts = buf.valid_start_slots()
    ks = buf.data['observations'][starts, 0]
    online = ks >= ONLINE_ID0
    assert np.all(ks[online] - ONLINE_ID0 + H - 1 <= buf.n - 1)
    assert online.sum() == buf.R - H + 1


def test_success_buffer_whole_episode_fifo():
    rs = np.random.RandomState(1)
    demos = make_demos([30, 40, 25])
    D = 95
    buf = SeqReplay(demos, capacity=D + 120, horizon=H, episodic=True)
    snapshot = {k: v[:D].copy() for k, v in buf.data.items()}
    inserted = []  # (start_k, length)
    k = 0
    np.random.seed(0)
    for _ in range(60):
        L = int(rs.randint(5, 50))
        rows = [row(ONLINE_ID0 + k + i, terminal=i == L - 1, success=i == L - 1) for i in range(L)]
        buf.add_episode(rows)
        inserted.append((k, L))
        k += L
        # Held episodes are the newest ones, whole, in insertion order (FIFO eviction).
        n_held = len(buf.episodes)
        assert [tuple(e) for e in buf.episodes] == inserted[-n_held:]
        assert buf.k_lo == buf.episodes[0][0] and buf.n == k
        assert buf.n - buf.k_lo <= buf.R
        if n_held < len(inserted):  # the next older episode did not fit
            prev_k, prev_L = inserted[-n_held - 1]
            assert buf.n - prev_k > buf.R
        for s, length in buf.episodes:
            slots = buf.slot(np.arange(s, s + length))
            assert np.array_equal(buf.data['observations'][slots, 0] - ONLINE_ID0, np.arange(s, s + length))
            assert buf.data['terminals'][slots][-1] == 1 and buf.data['masks'][slots][-1] == 0
            assert np.all(buf.data['terminals'][slots][:-1] == 0)
        # Slots not held are zero (evicted episodes are cleared).
        held = set(buf.slot(np.arange(buf.k_lo, buf.n)).tolist())
        free = [s for s in range(D, buf.cap) if s not in held]
        assert np.all(buf.data['observations'][free] == 0)
        check_sequences(buf, 500)
    for f in FIELDS:
        assert np.array_equal(buf.data[f][:D], snapshot[f])
    assert buf.overwrites == sum(L for _, L in inserted) - (buf.n - buf.k_lo)


def test_segments_restore_bit_exact():
    rs = np.random.RandomState(2)
    demos = make_demos([30, 40, 25])
    for episodic in (False, True):
        buf = SeqReplay(demos, capacity=95 + 64, horizon=H, episodic=episodic)
        segs = []
        k = 0
        for _ in range(25):
            L = int(rs.randint(5, 30))
            rows = [row(ONLINE_ID0 + k + i, terminal=i == L - 1, success=episodic and i == L - 1) for i in range(L)]
            if episodic:
                buf.add_episode(rows)
            else:
                for r in rows:
                    buf.add_transition(r)
            k += L
            if rs.rand() < 0.5:
                seg = buf.new_segment()
                if seg is not None:
                    segs.append(seg)
        seg = buf.new_segment()
        if seg is not None:
            segs.append(seg)
        fresh = SeqReplay(demos, capacity=95 + 64, horizon=H, episodic=episodic)
        fresh.restore(buf.meta(), segs)
        assert fresh.meta() == buf.meta()
        for f in FIELDS:
            assert np.array_equal(fresh.data[f], buf.data[f]), (episodic, f)
        assert np.array_equal(fresh.valid_start_slots(), buf.valid_start_slots())


def test_episode_longer_than_ring_rejected():
    buf = SeqReplay(make_demos([10]), capacity=20, horizon=H, episodic=True)
    with pytest.raises(AssertionError):
        buf.add_episode([row(i) for i in range(11)])
