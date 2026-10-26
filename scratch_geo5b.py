import sys
sys.path.insert(0, 'tools')
sys.path.insert(0, 'tests')
sys.path.insert(0, '.')
import jax  # noqa
import numpy as np  # noqa
import test_geo5b_introspect as T  # noqa
from agents.acarb import ar_bin, canonical_digits  # noqa

ag = T.agent_of('E1')
b = T.fake_batch()
_, sr, rr, _ = jax.random.split(jax.random.PRNGKey(2), 4)
y, u, info = ag.td_target(b, sr, rr)
k = np.asarray(info['target_k'])
print(k[:6], np.asarray(u)[:6], np.asarray(ar_bin(u, 10, 3))[:6], np.asarray(canonical_digits(k, 20000, 10, 3))[:6])
