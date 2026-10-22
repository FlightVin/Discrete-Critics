import os
os.chdir('/home/vineethb/ae/work/v1rec')
s = open('.ae-scratch/e2e_head.py').read()
old = """            p.send_signal(signal.SIGKILL)
            snap = list(lines)
            # Last committed generation: the last CKPT_DONE of this process, else the one it resumed from.
            gens = [l for l in snap if l.startswith(('CKPT_DONE', 'RESUME'))]
            m = re.search(r'gen=(\\d+)', gens[-1]) if gens else None
            out['last_done_gen'] = int(m.group(1)) if m else None
            ck = [l for l in snap if l.startswith(('CKPT_BEGIN', 'CKPT_DONE'))]
            out['ckpt_in_progress'] = bool(ck) and ck[-1].startswith('CKPT_BEGIN')
            out['killed'] = True"""
new = """            p.send_signal(signal.SIGKILL)
            out['killed'] = True"""
if old in s:
    s = s.replace(old, new)
old = """    out.update(rc=rc, lines=lines, fired=fired.is_set())\n"""
new = """    out.update(rc=rc, lines=lines, fired=fired.is_set())
    if out['killed']:  # all output printed before the kill has been read now
        # Last committed generation: the last CKPT_DONE of this process, else the one it resumed from.
        gens = [l for l in lines if l.startswith(('CKPT_DONE', 'RESUME'))]
        m = re.search(r'gen=(\\d+)', gens[-1]) if gens else None
        out['last_done_gen'] = int(m.group(1)) if m else None
        ck = [l for l in lines if l.startswith(('CKPT_BEGIN', 'CKPT_DONE'))]
        out['ckpt_in_progress'] = bool(ck) and ck[-1].startswith('CKPT_BEGIN')
"""
if "if out['killed']:" not in s:
    s = s.replace(old, new)
bench = open('.ae-scratch/bench_part.py').read()
main = '''if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/tmp/v1rec_e2e')
    p.add_argument('--sets', default='bsc,arb_qc,comp,det', help='subset of bsc,arb_qc,comp,det (bc always runs)')
    p.add_argument('--bench_full', action='store_true')
    p.add_argument('--bench_dir', default='/tmp/v1rec_bench')
    p.add_argument('--bench_new_rows', type=int, default=150_000)
    a = p.parse_args()
    if a.bench_full:
        bench_full(a.bench_dir, a.bench_new_rows)
    else:
        if os.path.exists(a.root):
            shutil.rmtree(a.root)
        sys.exit(0 if interruption_test(a.root, set(a.sets.split(','))) else 1)
'''
open('tests/e2e_v1_interrupt.py', 'w').write(s + bench + main)
print('ok')
