# ---- synthetic full-size checkpoint benchmark ----------------------------------------------------------
def bench_full(bench_dir, new_rows, obs_dim=53, act_dim=7, cap=5_000_000, demo=95_962, H=8):
    """Both buffers full at tool_hang size; time the first (full) write, an incremental (SIGTERM-path)
    write with `new_rows` new rows per buffer, and a full verified load."""
    import jax
    jax.config.update('jax_platforms', 'cpu')
    import flax
    import ml_collections
    from agents import acbsc
    from utils.replay_v1 import FIELDS, SeqReplay

    rs = np.random.default_rng(0)
    demos = {
        'observations': rs.standard_normal((demo, obs_dim), dtype=np.float32),
        'next_observations': rs.standard_normal((demo, obs_dim), dtype=np.float32),
        'actions': rs.uniform(-1, 1, (demo, act_dim)).astype(np.float32),
        'rewards': -np.ones(demo, np.float32), 'terminals': np.zeros(demo, np.float32),
        'masks': np.ones(demo, np.float32), 'num_chunks_to_timeout': np.ones(demo, np.float32),
        'num_chunks_to_success': np.ones(demo, np.float32)}
    cfg = acbsc.get_config()
    cfg.actor_type, cfg.horizon_length = 'best-of-n', H
    agent = acbsc.ACBSCAgent.create(0, demos['observations'][0], demos['actions'][0], cfg)
    agent_state = jax.device_get(flax.serialization.to_state_dict(agent))

    def fill(buf, n_rows, ep_len=700):
        """Bulk-append n_rows online rows (episodes of ep_len for the success buffer)."""
        R = buf.R
        if buf.episodic:
            n_ep = max(1, n_rows // ep_len)
            for _ in range(n_ep):
                while buf.n + ep_len - buf.k_lo > R:
                    s, L = buf.episodes.pop(0)
                    for f in FIELDS:  # as SeqReplay.add_episode: evicted slots are zeroed
                        buf.data[f][buf.slot(np.arange(s, s + L))] = 0
                    buf.k_lo += L
                    buf.overwrites += L
                buf.episodes.append([buf.n, ep_len])
                _bulk(buf, buf.n, ep_len)
                buf.n += ep_len
        else:
            _bulk(buf, buf.n, n_rows)
            buf.n += n_rows
            if buf.n - buf.k_lo > R:
                buf.overwrites += buf.n - R - buf.k_lo
                buf.k_lo = buf.n - R

    def _bulk(buf, k0, n):
        s = buf.slot(np.arange(k0, k0 + n))
        for f in FIELDS:
            shape = (n, *buf.data[f].shape[1:])
            buf.data[f][s] = rs.standard_normal(shape, dtype=np.float32) if buf.data[f].ndim > 1 else -1.0

    t = time.time()
    bufs = dict(main=SeqReplay(demos, cap, H, episodic=False), success=SeqReplay(demos, cap, H, episodic=True))
    for b in bufs.values():
        fill(b, b.R)
    print(f'BENCH built 2 x {cap} rows (obs {obs_dim}) in {time.time() - t:.1f}s; '
          f'occupancy main={bufs["main"].stats()["occupancy"]:.4f} success={bufs["success"].stats()["occupancy"]:.4f}',
          flush=True)

    root = os.path.join(bench_dir, 'ckpt')
    if os.path.exists(bench_dir):
        shutil.rmtree(bench_dir)
    store = CheckpointStore(root)
    store.load()

    def save():
        segs = {k: b.new_segment() for k, b in bufs.items()}
        state = dict(agent=agent_state, np_rng=np.random.get_state(), buffers={k: b.meta() for k, b in bufs.items()})
        t = time.time()
        info = store.save(state, segs, {k: b.k_lo for k, b in bufs.items()})
        return info, time.time() - t

    full, t_full = save()
    print(f'BENCH full write (all 2 x {cap} rows): {t_full:.1f}s, segments {full["segment_bytes"] / 1e9:.2f} GB, '
          f'state {full["state_bytes"] / 1e6:.0f} MB', flush=True)
    for b in bufs.values():
        fill(b, new_rows)
    inc, t_inc = save()
    print(f'BENCH incremental write ({new_rows} new rows per buffer, buffers full, old rows overwritten): '
          f'{t_inc:.1f}s (segments {inc["t_segments"]:.1f}s, state+commit {inc["t_state"]:.1f}s, '
          f'cleanup {inc["t_cleanup"]:.1f}s), wrote {inc["segment_bytes"] / 1e6:.0f} MB + state '
          f'{inc["state_bytes"] / 1e6:.0f} MB', flush=True)
    du = sum(os.path.getsize(p) for p in glob.glob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(p))
    print(f'BENCH checkpoint on disk: {du / 1e9:.2f} GB', flush=True)

    t = time.time()
    store2 = CheckpointStore(root)
    state, _ = store2.load()
    fresh = {k: SeqReplay(demos, cap, H, episodic=(k == 'success')) for k in bufs}
    for k, b in fresh.items():
        b.restore(state['buffers'][k], store2.iter_segments(k))
    t_load = time.time() - t
    same = all(np.array_equal(fresh[k].data[f], bufs[k].data[f]) for k in bufs for f in FIELDS)
    print(f'BENCH verified load + restore of both buffers: {t_load:.1f}s, bit-exact={same}', flush=True)
    shutil.rmtree(bench_dir)
    return dict(t_full=t_full, t_inc=t_inc, t_load=t_load, exact=same)


