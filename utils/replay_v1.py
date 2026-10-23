"""Replay buffers for the v1 protocol (train_v1.py).

Layout: slots [0, D) hold the demos and are never written after creation. Online data lives in the ring
[D, cap). Every online transition gets a write-order index k = 0, 1, 2, ... and is stored in slot
D + k % R (R = cap - D). The buffer keeps the online rows with write-order in [k_lo, n): n is the number
written so far, k_lo the oldest row still held.

  main buffer     k_lo = max(0, n - R): the newest write overwrites the oldest online row.
  success buffer  insert and evict whole episodes (FIFO); k_lo is the start of the oldest held episode.
                  Evicted slots are zeroed, so memory equals what a checkpoint restores.

A sequence start is valid iff its H steps are all held and contiguous in write order: demo starts
[0, D - H] and online starts k in [k_lo, n - H]. Sequences never cross from demos into online data, nor
across the ring's write head. Sampling is uniform over valid starts and draws from numpy's global RNG,
like utils/datasets.py: Dataset.sample_sequence, and returns the same dict; with `rng` (a
np.random.Generator, train_v1 --replay_rng=separate) it draws from that generator instead.

Persistence is incremental: `new_segment()` returns the rows written since the last call (write-order
[seg_from, n)). Rows are final once written, because train_v1 only checkpoints at episode boundaries
(the main buffer's num_chunks_to_success is set retroactively when an episode succeeds, before that).
"""
import numpy as np

FIELDS = ('observations', 'actions', 'rewards', 'terminals', 'masks', 'next_observations',
          'num_chunks_to_timeout', 'num_chunks_to_success')


class SeqReplay:
    def __init__(self, demos, capacity, horizon, episodic, rng=None):
        """demos: dict of arrays with FIELDS (the processed demo dataset). episodic: whole-episode eviction.
        rng: np.random.Generator for sampling (None: numpy's global RNG). Not part of meta(): its owner
        saves and restores its state."""
        self.rng = rng
        self.D = int(len(demos['observations']))
        self.cap = int(capacity)
        self.R = self.cap - self.D
        assert self.R > 0, f'buffer capacity {capacity} must exceed the {self.D} demo steps'
        self.H = int(horizon)
        self.episodic = bool(episodic)
        self.data = {}
        for k in FIELDS:
            src = np.asarray(demos[k])
            self.data[k] = np.zeros((self.cap, *src.shape[1:]), dtype=src.dtype)
            self.data[k][:self.D] = src
        self.n = 0              # online rows ever written
        self.k_lo = 0           # oldest online write-order index still held
        self.seg_from = 0       # first write-order index not yet handed out by new_segment()
        self.episodes = []      # success buffer: [start_k, length] of each held episode, oldest first
        self.overwrites = 0     # online rows overwritten (main) or evicted (success)

    # ---- indexing -------------------------------------------------------------------------------
    def slot(self, k):
        return self.D + np.asarray(k) % self.R

    @property
    def online_size(self):
        return self.n - self.k_lo

    @property
    def num_demo_starts(self):
        return max(0, self.D - self.H + 1)

    @property
    def num_online_starts(self):
        return max(0, self.n - self.H + 1 - self.k_lo)

    @property
    def num_valid_starts(self):
        return self.num_demo_starts + self.num_online_starts

    def valid_start_slots(self):
        """All valid start slots (for tests and diagnostics; sampling does not build this)."""
        demo = np.arange(self.num_demo_starts)
        online = self.slot(np.arange(self.k_lo, self.k_lo + self.num_online_starts))
        return np.concatenate([demo, online]).astype(np.int64)

    # ---- writing --------------------------------------------------------------------------------
    def _write(self, k, row):
        s = self.D + k % self.R
        for f in FIELDS:
            self.data[f][s] = row[f]

    def add_transition(self, row):
        """Main buffer: append one transition; when full, overwrite the oldest online row."""
        assert not self.episodic
        if self.n - self.k_lo == self.R:
            self.k_lo += 1
            self.overwrites += 1
        self._write(self.n, row)
        self.n += 1
        return self.D + (self.n - 1) % self.R

    def add_episode(self, rows):
        """Success buffer: append a whole episode, first evicting whole oldest episodes to make room."""
        assert self.episodic
        L = len(rows)
        assert 0 < L <= self.R, f'episode of {L} steps does not fit the {self.R}-step ring'
        while self.n + L - self.k_lo > self.R:
            start, length = self.episodes.pop(0)
            assert start == self.k_lo
            s = self.slot(np.arange(start, start + length))
            for f in FIELDS:
                self.data[f][s] = 0
            self.k_lo += length
            self.overwrites += length
        for i, row in enumerate(rows):
            self._write(self.n + i, row)
        self.episodes.append([self.n, L])
        self.n += L

    def set_field(self, field, slots, values):
        self.data[field][slots] = values

    # ---- sampling -------------------------------------------------------------------------------
    def sample_starts(self, batch_size):
        """Uniform over valid starts: (start slot, is_online, write-order index or 0 for demos)."""
        nd, total = self.num_demo_starts, self.num_valid_starts
        if self.rng is None:
            u = np.random.randint(total, size=batch_size)
        else:
            u = self.rng.integers(total, size=batch_size)
        online = u >= nd
        k0 = np.where(online, self.k_lo + (u - nd), 0)
        idxs = np.where(online, self.slot(k0), u)
        return idxs, online, k0

    def sample_sequence(self, batch_size, sequence_length, discount):
        """Same output as Dataset.sample_sequence, over valid starts only."""
        assert sequence_length == self.H
        H = sequence_length
        idxs, online, k0 = self.sample_starts(batch_size)
        # Slot of step j of each sequence: demos are contiguous; online rows step through the ring.
        off = np.arange(H)[None, :]
        demo_slots = idxs[:, None] + off
        online_slots = self.slot(k0[:, None] + off)
        all_slots = np.where(online[:, None], online_slots, demo_slots)
        # next_actions: the step after each, clamped to the last held row of the same region.
        demo_next = np.minimum(demo_slots + 1, self.D - 1)
        online_next = self.slot(np.minimum(k0[:, None] + off + 1, self.n - 1))
        next_slots = np.where(online[:, None], online_next, demo_next)

        d = self.data
        flat = all_slots.reshape(-1)
        obs = d['observations'][flat].reshape(batch_size, H, *d['observations'].shape[1:])
        next_obs = d['next_observations'][flat].reshape(batch_size, H, *d['next_observations'].shape[1:])
        actions = d['actions'][flat].reshape(batch_size, H, *d['actions'].shape[1:])
        next_actions = d['actions'][next_slots.reshape(-1)].reshape(batch_size, H, *d['actions'].shape[1:])
        b_rew = d['rewards'][all_slots]
        b_mask = d['masks'][all_slots]
        b_term = d['terminals'][all_slots]

        rewards = np.zeros((batch_size, H), dtype=float)
        masks = np.ones((batch_size, H), dtype=float)
        terminals = np.zeros((batch_size, H), dtype=float)
        valid = np.ones((batch_size, H), dtype=float)
        rewards[:, 0] = b_rew[:, 0]
        masks[:, 0] = b_mask[:, 0]
        terminals[:, 0] = b_term[:, 0]
        discount_powers = discount ** np.arange(H)
        for i in range(1, H):
            rewards[:, i] = rewards[:, i - 1] + b_rew[:, i] * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i - 1], b_mask[:, i])
            terminals[:, i] = np.maximum(terminals[:, i - 1], b_term[:, i])
            valid[:, i] = 1.0 - terminals[:, i - 1]

        return dict(
            observations=d['observations'][idxs].copy(),
            full_observations=obs,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_obs,
            next_actions=next_actions,
            num_chunks_to_timeout=d['num_chunks_to_timeout'][idxs],
            num_chunks_to_success=d['num_chunks_to_success'][idxs],
        )

    # ---- stats ----------------------------------------------------------------------------------
    def stats(self):
        return dict(demo=self.D, online=self.online_size, occupancy=(self.D + self.online_size) / self.cap,
                    valid_starts=self.num_valid_starts, online_overwrites=self.overwrites,
                    online_written=self.n, episodes=len(self.episodes))

    # ---- persistence ----------------------------------------------------------------------------
    def rows(self, k_from, k_to):
        """Copies of the online rows with write-order [k_from, k_to) (all must be held)."""
        assert self.k_lo <= k_from <= k_to <= self.n
        s = self.slot(np.arange(k_from, k_to))
        return {f: self.data[f][s] for f in FIELDS}

    def new_segment(self):
        """(k_from, k_to, rows) written since the last call, or None. Marks them handed out."""
        a, b = max(self.seg_from, self.k_lo), self.n
        self.seg_from = self.n
        if b <= a:
            return None
        return a, b, self.rows(a, b)

    def meta(self):
        return dict(D=self.D, cap=self.cap, H=self.H, episodic=self.episodic, n=self.n, k_lo=self.k_lo,
                    seg_from=self.seg_from, episodes=[list(e) for e in self.episodes],
                    overwrites=self.overwrites)

    def restore(self, meta, segments):
        """Rebuild from meta and an iterable of (k_from, k_to, rows) covering at least [k_lo, n)."""
        for key in ('D', 'cap', 'H', 'episodic'):
            assert meta[key] == getattr(self, key), (key, meta[key], getattr(self, key))
        assert self.n == 0, 'restore into a fresh buffer'
        self.n, self.k_lo, self.seg_from = meta['n'], meta['k_lo'], meta['seg_from']
        self.episodes = [list(e) for e in meta['episodes']]
        self.overwrites = meta['overwrites']
        covered = 0
        for a, b, rows in segments:
            lo = max(a, self.k_lo)
            if lo >= b:
                continue
            s = self.slot(np.arange(lo, b))
            for f in FIELDS:
                self.data[f][s] = rows[f][lo - a:]
            covered += b - lo
        assert covered == self.n - self.k_lo, f'segments cover {covered} of {self.n - self.k_lo} held rows'
