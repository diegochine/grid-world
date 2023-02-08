from environment import GridWorld
from agent import MonteCarloAgent


if __name__ == '__main__':
    n_episodes = 300
    max_steps = 10
    env = GridWorld(size=4, seed=42)
    agent = MonteCarloAgent()

    for episode in range(n_episodes):
        print(f'[{episode}/{n_episodes}]')
        done = False
        obs = env.reset((0, 0), (3, 3))
        states = []
        actions = []
        rewards = []
        step = 0
        while not done and step < max_steps:
            env.render()
            feasible_actions = env.get_feasible_actions()
            action = agent.act(obs, feasible_actions)
            next_obs, reward, done, _ = env.step(action)
            states.append(obs.tobytes())
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            step += 1
        if step == max_steps:
            rewards[-1] = -100
        agent.train(states, actions, rewards)

    print('TESTING')
    done = False
    obs = env.reset((0, 0), (3, 3))
    states = []
    actions = []
    rewards = []
    step = 0
    while not done and step < max_steps:
        env.render()
        feasible_actions = env.get_feasible_actions()
        action = agent.act(obs, feasible_actions, training=False)
        next_obs, reward, done, _ = env.step(action)
        states.append(obs.tobytes())
        actions.append(action)
        rewards.append(reward)
        obs = next_obs
        step += 1
    if step == max_steps:
        print('failed')
    else:
        print('victory!')
