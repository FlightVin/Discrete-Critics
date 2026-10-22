"""Latest-only, atomically published checkpoints with incremental replay segments (train_v1.py).

Layout under <run_dir>/ckpt:
  LATEST                 name of the committed generation (switched by rename: the commit point)
  gen_<N>/state.pkl      everything except replay rows: agent, RNG states, counters, buffer metadata
  gen_<N>/manifest.json  sizes + sha256 of the small files, the replay segments this generation uses
  segments/<buf>_<from>_<to>_g<N>.npz   immutable online replay rows, write-order [from, to)

A save writes new segments and gen_<N> (uncommitted until LATEST names it), fsyncs, re-reads and checks
the small files, then switches LATEST, then deletes the previous generation and every segment the new
one does not reference. A crash at any point leaves the previous generation loadable; leftovers are
removed by the next load or save. Segments are written once, so a checkpoint writes only the replay rows
added since the previous one.
"""
import hashlib
import json
import os
import pickle
import shutil
import time
import zlib

import numpy as np


class SimulatedCrash(RuntimeError):
    pass


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path, data):
    with open(path, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def rows_crc(rows):
    crc = 0
    for k in sorted(rows):
        crc = zlib.crc32(np.ascontiguousarray(rows[k]).data, crc)
    return crc


class CheckpointStore:
    def __init__(self, root):
        self.root = root
        self.seg_dir = os.path.join(root, 'segments')
        os.makedirs(self.seg_dir, exist_ok=True)
        self.gen = None          # committed generation number
        self.segments = {}       # buffer name -> list of segment entries of the committed generation
        self.fail_at = None      # tests: raise SimulatedCrash at this stage of save()

    # ---- reading --------------------------------------------------------------------------------
    def latest(self):
        p = os.path.join(self.root, 'LATEST')
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return f.read().strip()

    def manifest(self):
        name = self.latest()
        if name is None:
            return None
        with open(os.path.join(self.root, name, 'manifest.json')) as f:
            return json.load(f)

    def load(self):
        """Verify and load the committed generation: (state, manifest), or (None, None) if none."""
        name = self.latest()
        if name is None:
            self.cleanup()
            return None, None
        gdir = os.path.join(self.root, name)
        man = self.manifest()
        for fname, info in man['files'].items():
            with open(os.path.join(gdir, fname), 'rb') as f:
                data = f.read()
            assert len(data) == info['size'] and _sha256(data) == info['sha256'], f'{gdir}/{fname} is corrupt'
        for entries in man['segments'].values():
            for e in entries:
                p = os.path.join(self.seg_dir, e['file'])
                assert os.path.getsize(p) == e['size'], f'segment {p} has the wrong size'
        with open(os.path.join(gdir, 'state.pkl'), 'rb') as f:
            state = pickle.load(f)
        self.gen = man['gen']
        self.segments = {k: [dict(e) for e in v] for k, v in man['segments'].items()}
        self.cleanup()
        return state, man

    def iter_segments(self, buf):
        """(k_from, k_to, rows) of each segment of buffer `buf` in the committed generation, CRC-checked."""
        for e in self.segments.get(buf, []):
            with np.load(os.path.join(self.seg_dir, e['file'])) as z:
                rows = {k: z[k] for k in z.files}
            assert rows_crc(rows) == e['crc'], f"segment {e['file']} fails its checksum"
            yield e['k_from'], e['k_to'], rows

    # ---- writing --------------------------------------------------------------------------------
    def _maybe_crash(self, stage):
        if self.fail_at == stage:
            raise SimulatedCrash(stage)

    def save(self, state, new_segments, keep_from, meta=None):
        """Publish a new generation.

        new_segments: {buf: (k_from, k_to, rows) or None}, rows added since the previous save.
        keep_from: {buf: k_lo}; segments wholly below k_lo are no longer referenced.
        meta: small JSON-able dict copied into the manifest (e.g. phase), readable without state.pkl.
        Returns timing / size info.
        """
        t0 = time.time()
        latest = self.latest()
        assert latest == (None if self.gen is None else f'gen_{self.gen:08d}'), 'load() before save()'
        gen = (self.gen or 0) + 1
        name = f'gen_{gen:08d}'
        gdir = os.path.join(self.root, name)
        if os.path.exists(gdir):
            shutil.rmtree(gdir)
        os.makedirs(gdir)

        segments = {b: [e for e in v if e['k_to'] > keep_from.get(b, 0)] for b, v in self.segments.items()}
        seg_bytes = 0
        for buf, seg in new_segments.items():
            segments.setdefault(buf, [])
            if seg is None:
                continue
            a, b, rows = seg
            fname = f'{buf}_{a:012d}_{b:012d}_g{gen:08d}.npz'
            path = os.path.join(self.seg_dir, fname)
            with open(path, 'wb') as f:
                np.savez(f, **rows)
                f.flush()
                os.fsync(f.fileno())
            size = os.path.getsize(path)
            seg_bytes += size
            segments[buf].append(dict(file=fname, k_from=int(a), k_to=int(b), size=size, crc=rows_crc(rows)))
            self._maybe_crash('after_segment')
        _fsync_dir(self.seg_dir)
        t_seg = time.time()

        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        _write_bytes(os.path.join(gdir, 'state.pkl'), blob)
        man = dict(gen=gen, time=time.time(), meta=meta or {},
                   files={'state.pkl': dict(size=len(blob), sha256=_sha256(blob))}, segments=segments)
        _write_bytes(os.path.join(gdir, 'manifest.json'), json.dumps(man, indent=1).encode())
        _fsync_dir(gdir)
        self._maybe_crash('before_verify')

        # Verify what is on disk before publishing it.
        with open(os.path.join(gdir, 'state.pkl'), 'rb') as f:
            assert _sha256(f.read()) == man['files']['state.pkl']['sha256'], 'state.pkl verify failed'
        for entries in segments.values():
            for e in entries:
                assert os.path.getsize(os.path.join(self.seg_dir, e['file'])) == e['size']
        self._maybe_crash('before_switch')

        tmp = os.path.join(self.root, 'LATEST.tmp')
        _write_bytes(tmp, name.encode())
        os.replace(tmp, os.path.join(self.root, 'LATEST'))
        _fsync_dir(self.root)
        self.gen, self.segments = gen, segments
        t_commit = time.time()
        self._maybe_crash('after_switch')

        self.cleanup()
        t_end = time.time()
        return dict(gen=gen, state_bytes=len(blob), segment_bytes=seg_bytes,
                    t_segments=t_seg - t0, t_state=t_commit - t_seg, t_cleanup=t_end - t_commit,
                    t_total=t_end - t0)

    def cleanup(self):
        """Delete every generation except the committed one, and segments it does not reference."""
        current = self.latest()
        for n in os.listdir(self.root):
            p = os.path.join(self.root, n)
            if n.startswith('gen_') and n != current:
                shutil.rmtree(p, ignore_errors=True)
            elif n == 'LATEST.tmp':
                os.remove(p)
        referenced = {e['file'] for v in self.segments.values() for e in v} if current else set()
        for n in os.listdir(self.seg_dir):
            if n not in referenced:
                os.remove(os.path.join(self.seg_dir, n))
