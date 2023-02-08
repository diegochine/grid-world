from collections import defaultdict
import numpy as np


class MonteCarloAgent:

    actions = ('n', 's', 'w', 'e')

    def __init__(self, eps=1, discount=0.99):
        self.q_vals = defaultdict(lambda: 10)
        self.counter = {a: 0 for a in self.actions}
        self.discount = discount
        self.eps = eps

    def act(self, state, feasible_actions, training=True):
        self.eps = max(0.01, self.eps * 0.95)
        if training and np.random.rand() < self.eps:
            return np.random.choice(feasible_actions)
        else:
            state = state.tostring()
            state_q_vals = {action: self.q_vals[(state, action)] for action in self.actions if action in feasible_actions}
            return max(state_q_vals, key=state_q_vals.get)

    def train(self, states, actions, rewards):
        for t, (s_t, a_t, r_t) in enumerate(zip(states, actions, rewards)):
            if s_t not in states[:t]:
                self.counter[a_t] += 1
                self.q_vals[(s_t, a_t)] += 1/self.counter[a_t] * (r_t - self.q_vals[(s_t, a_t)])
