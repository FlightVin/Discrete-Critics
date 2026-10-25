from collections import defaultdict

import jax
import numpy as np
from tqdm import trange, tqdm
from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
            
        observation_history = []
        action_history = []
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        while not done:
            
            action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            # print(info)
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                # print(info["gripper_contact"])
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def flatten_async(d, idx, parent_key='', sep='.'):
    """Flatten a vectorized info dict, extracting element at index idx."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            try:
                items.append((new_key, v[idx]))
            except (TypeError, IndexError):
                items.append((new_key, v))
    return dict(items)


def evaluate_parallel(
    agent,
    envs,
    num_eval_envs,
    eval_episodes,
    action_dim,
    return_episodes=False,
):
    """Evaluate the agent using a vectorized (parallel) environment.

    Args:
        agent: Agent.
        envs: AsyncVectorEnv with num_eval_envs parallel environments.
        num_eval_envs: Number of parallel environments.
        eval_episodes: Number of complete episodes to run per environment.
        action_dim: Per-step action dimension.
        return_episodes: also return per-episode results (see Returns).

    Returns:
        A tuple (stats, [], []) compatible with the sequential evaluate() signature. With return_episodes:
        (stats, [], [], episodes), episodes = dict of [num_eval_envs, eval_episodes] arrays: success (the
        final info's 'success', as averaged in stats), length (env steps until done) and first_obs
        ([..., obs_dim], the observation returned by reset).
    """
    total_episodes = num_eval_envs * eval_episodes
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    stats = defaultdict(list)
    pbar = tqdm(total=total_episodes, desc="Parallel Eval", dynamic_ncols=True)
    ep_success = np.zeros((num_eval_envs, eval_episodes))
    ep_length = np.zeros((num_eval_envs, eval_episodes), dtype=np.int64)
    first_obs = []

    for rollout_idx in range(eval_episodes):
        observations, _ = envs.reset()
        if return_episodes:
            first_obs.append(np.array(observations))

        action_queues = [[] for _ in range(num_eval_envs)]
        active_envs = np.ones(num_eval_envs, dtype=bool)

        while np.any(active_envs):
            # Collect envs that need new action chunks
            envs_needing_actions = [i for i, q in enumerate(action_queues) if not q and active_envs[i]]
            if envs_needing_actions:
                obs_to_act = observations[envs_needing_actions]
                n_need = len(envs_needing_actions)

                # Pad batch to fixed size to avoid JIT recompilation as envs finish
                if n_need < num_eval_envs:
                    pad = np.zeros((num_eval_envs - n_need,) + obs_to_act.shape[1:], dtype=obs_to_act.dtype)
                    obs_padded = np.concatenate([obs_to_act, pad], axis=0)
                    action_chunks = np.array(actor_fn(observations=obs_padded))
                    action_chunks = action_chunks.reshape(num_eval_envs, -1, action_dim)[:n_need]
                else:
                    action_chunks = np.array(actor_fn(observations=obs_to_act))
                    action_chunks = action_chunks.reshape(n_need, -1, action_dim)

                for i, env_idx in enumerate(envs_needing_actions):
                    for a in action_chunks[i]:
                        action_queues[env_idx].append(a)

            # Build step actions (dummy zeros for inactive/waiting envs)
            step_actions = []
            for i in range(num_eval_envs):
                if active_envs[i] and action_queues[i]:
                    step_actions.append(action_queues[i].pop(0))
                else:
                    step_actions.append(np.zeros(action_dim))

            observations, rewards, terminations, truncations, infos = envs.step(np.array(step_actions))
            dones = np.logical_or(terminations, truncations)
            ep_length[:, rollout_idx] += active_envs

            for i in range(num_eval_envs):
                if active_envs[i] and dones[i]:
                    info_i = flatten_async(infos, i)
                    add_to(stats, info_i)
                    ep_success[i, rollout_idx] = info_i.get('success', np.nan)
                    active_envs[i] = False
                    pbar.update(1)

    pbar.close()

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if return_episodes:
        episodes = dict(success=ep_success, length=ep_length, first_obs=np.stack(first_obs, axis=1))
        return stats, [], [], episodes
    return stats, [], []
