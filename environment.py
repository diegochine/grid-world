import numpy as np


class GridWorld:

    def __init__(self, size=3, seed=None):
        if isinstance(size, int):
            size = (size, size)
        assert isinstance(size, tuple), f'{size}'

        if seed is not None:
            np.random.seed(seed)

        self.size = size
        self.world = None
        self.target = (size[0] - 1, size[1] - 1)
        self.agent_pos = (0, 0)

    def reset(self, target=None, agent_pos=None):
        self.world = np.zeros(self.size)
        self.target = target if target is not None else (np.random.randint(self.size[0]), np.random.randint(self.size[1]))
        self.agent_pos = agent_pos if agent_pos is not None else (np.random.randint(self.size[0]), np.random.randint(self.size[1]))
        self.world[self.target] = 1
        self.world[self.agent_pos] = -1
        return self.world.copy('K')

    def render(self):
        print(f'Agent: {self.agent_pos}, target: {self.target}')

    def get_feasible_actions(self):
        feasibles = []
        if self.agent_pos[0] > 0:
            feasibles.append('n')
        if self.agent_pos[0] < self.size[0] - 1:
            feasibles.append('s')
        if self.agent_pos[1] > 0:
            feasibles.append('w')
        if self.agent_pos[1] < self.size[1] - 1:
            feasibles.append('e')
        return feasibles

    def step(self, action):
        assert action in ('n', 's', 'w', 'e')
        if action == 'n':
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 's':
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 'w':
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        else:  # action == 'e'
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)

        assert all(c >= 0 and c < self.size[0] for c in new_pos), f'{new_pos}'

        self.world[new_pos] = -1
        self.world[self.agent_pos] = 0
        self.agent_pos = new_pos
        if new_pos == self.target:
            done = True
            reward = 0
        else:
            done = False
            reward = -1

        return self.world.copy('K'), reward, done, {}
