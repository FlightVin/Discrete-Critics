"""Atomic latest-only publish (utils/ckpt_v1.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ckpt_v1 import CheckpointStore, SimulatedCrash  # noqa: E402


def seg(a, b):
    return a, b, {'observations': np.arange(a, b, dtype=np.float32)[:, None].repeat(3, 1),
                  'rewards': -np.ones(b - a, np.float32)}


def load_all(root):
    store = CheckpointStore(root)
    state, man = store.load()
    rows = {b: list(store.iter_segments(b)) for b in store.segments}
    return store, state, man, rows


def listing(root):
    return sorted(os.listdir(root)), sorted(os.listdir(os.path.join(root, 'segments')))


@pytest.mark.parametrize('stage', ['after_segment', 'before_verify', 'before_switch'])
def test_crash_before_switch_keeps_previous_generation(tmp_path, stage):
    root = str(tmp_path / 'ckpt')
    store = CheckpointStore(root)
    store.load()
    store.save({'step': 1, 'w': np.arange(5.0)}, {'main': seg(0, 10), 'success': None}, {'main': 0}, meta={'phase': 'a'})
    store.save({'step': 2, 'w': np.arange(6.0)}, {'main': seg(10, 20), 'success': seg(0, 4)}, {'main': 0}, meta={'phase': 'b'})

    store.fail_at = stage
    with pytest.raises(SimulatedCrash):
        store.save({'step': 3}, {'main': seg(20, 30), 'success': seg(4, 9)}, {'main': 15}, meta={'phase': 'c'})

    # A new process sees generation 2, fully intact; leftovers of the crashed save are removed.
    _, state, man, rows = load_all(root)
    assert man['gen'] == 2 and state['step'] == 2 and np.array_equal(state['w'], np.arange(6.0))
    assert [(a, b) for a, b, _ in rows['main']] == [(0, 10), (10, 20)]
    assert [(a, b) for a, b, _ in rows['success']] == [(0, 4)]
    for a, b, r in rows['main']:
        assert np.array_equal(r['observations'][:, 0], np.arange(a, b))
    dirs, segs = listing(root)
    assert dirs == ['LATEST', 'gen_00000002', 'segments']
    assert len(segs) == 3


def test_crash_after_switch_commits_new_generation(tmp_path):
    root = str(tmp_path / 'ckpt')
    store = CheckpointStore(root)
    store.load()
    store.save({'step': 1}, {'main': seg(0, 10)}, {'main': 0})
    store.fail_at = 'after_switch'
    with pytest.raises(SimulatedCrash):
        store.save({'step': 2}, {'main': seg(10, 20)}, {'main': 12})
    _, state, man, rows = load_all(root)
    assert man['gen'] == 2 and state['step'] == 2
    # keep_from=12: the [0, 10) segment is no longer referenced and was deleted on load.
    assert [(a, b) for a, b, _ in rows['main']] == [(10, 20)]
    dirs, segs = listing(root)
    assert dirs == ['LATEST', 'gen_00000002', 'segments'] and len(segs) == 1


def test_latest_only_and_segment_gc(tmp_path):
    root = str(tmp_path / 'ckpt')
    store = CheckpointStore(root)
    store.load()
    for g in range(1, 8):
        store.save({'g': g}, {'main': seg(10 * (g - 1), 10 * g)}, {'main': max(0, 10 * g - 25)})
        dirs, segs = listing(root)
        assert dirs == ['LATEST', f'gen_{g:08d}', 'segments']
        # Only segments overlapping [k_lo, n) are kept.
        assert len(segs) == min(g, 3)


def test_corrupt_state_detected(tmp_path):
    root = str(tmp_path / 'ckpt')
    store = CheckpointStore(root)
    store.load()
    store.save({'g': 1}, {'main': seg(0, 10)}, {'main': 0})
    p = os.path.join(root, 'gen_00000001', 'state.pkl')
    with open(p, 'r+b') as f:
        f.seek(5)
        f.write(b'\x00\x01\x02')
    with pytest.raises(AssertionError, match='corrupt'):
        CheckpointStore(root).load()


def test_save_requires_load(tmp_path):
    root = str(tmp_path / 'ckpt')
    s1 = CheckpointStore(root)
    s1.load()
    s1.save({'g': 1}, {}, {})
    with pytest.raises(AssertionError):
        CheckpointStore(root).save({'g': 2}, {}, {})  # would overwrite gen 1 without having loaded it
