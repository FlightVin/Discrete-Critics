from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acbsc import ACBSCAgent
from agents.acarb import ACARBAgent
from agents.accat import ACCATAgent
from agents.achlg import ACHLGAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acbsc=ACBSCAgent,
    acarb=ACARBAgent,
    accat=ACCATAgent,
    achlg=ACHLGAgent,
)

# v1 protocol: the six critics, as (agent_name, overrides of that agent's get_config()). Every method uses the
# best-of-32 BC-flow actor, and its TD target's next chunk is best-of-32 ranked by the current critic
# (mean of 2 copies) and valued by the target critic.
_V1_COMMON = dict(actor_type='best-of-n', actor_num_samples=32, num_qs=2, q_agg='mean')
V1_METHODS = dict(
    qc=('acfql', dict(_V1_COMMON)),
    bsc=('acbsc', dict(_V1_COMMON, critic_target_bon=True)),
    arb_bsc=('acarb', dict(_V1_COMMON, critic_target_bon=True, arb_value='bsc')),
    arb_qc=('acarb', dict(_V1_COMMON, critic_target_bon=True, arb_value='qc', discount=0.99)),
    cat=('accat', dict(_V1_COMMON)),
    hlg=('achlg', dict(_V1_COMMON)),
)


def v1_config(method):
    """get_config() of the method's agent with the v1 overrides applied."""
    import importlib
    name, overrides = V1_METHODS[method]
    config = importlib.import_module(f'agents.{name}').get_config()
    config.update(overrides)
    return config
