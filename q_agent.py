import numpy as np
class DeliveryQAgent:

    def __init__(self, n_actions, epsilon=0.5):
        self.Q = np.zeros(
            (n_actions,))  # Initialisez votre tableau Q avec des zéros ou utilisez une autre méthode selon vos besoins
        self.epsilon = epsilon
        self.actions_size = n_actions
        self.reset_memory()

    def act(self, s):
        q = np.copy(self.Q)
        q[self.states_memory] = -np.inf
        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])
        return a

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []