from jax import numpy as jnp


def sample_episodes(env, dataset, num_episodes, max_steps):
    for _ in range(num_episodes):
        (state, _), done, step = env.reset(), False, 0
        while not done and step < max_steps:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            dataset.add(state, action, reward, next_state)
            state = next_state
            step = step + 1
    return dataset


def sample_trajectory(env, max_steps):
    (state, _), done, steps, trajectory = env.reset(), False, 0, []
    while not done and steps < max_steps:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        trajectory.append((state, action, reward, next_state))
        state = next_state
        steps = steps + 1
    states, actions, rewards, next_states = zip(*trajectory)
    return jnp.array(states), jnp.array(actions), jnp.array(rewards), jnp.array(next_states)
