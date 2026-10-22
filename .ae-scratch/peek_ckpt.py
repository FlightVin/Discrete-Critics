import glob
import pickle
import sys

fs = glob.glob('/data/user_data/vineethb/Discrete-Critics/runs/dx-bsc-square/*/*/*/*/params_1000000.pkl')
print(fs)
d = pickle.load(open(fs[0], 'rb'))


def show(x, depth=0, maxd=3):
    if isinstance(x, dict) and depth < maxd:
        for k, v in x.items():
            print('  ' * depth + str(k), type(v).__name__, getattr(v, 'shape', ''))
            show(v, depth + 1, maxd)


show(d, 0, int(sys.argv[1]) if len(sys.argv) > 1 else 3)
import os
print(os.listdir(os.path.dirname(fs[0]))[:30])
