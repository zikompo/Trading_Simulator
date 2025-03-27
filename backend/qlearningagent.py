# qlearning_agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = {}

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        return self.Q[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        Qs = self.get_Q(state)
        return self.actions[np.argmax(Qs)]

    def learn(self, state, action, reward, next_state, done):
        a_index = self.actions.index(action)
        Qs = self.get_Q(state)
        q_predict = Qs[a_index]
        if not done:
            q_target = reward + self.gamma * np.max(self.get_Q(next_state))
        else:
            q_target = reward
        self.Q[state][a_index] += self.alpha * (q_target - q_predict)
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
