import random
from collections import namedtuple

##https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

        self.cache = dict()
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def cache_state(self, entity_id, timestep, state, action, state_value):
        if entity_id not in self.cache:
            self.cache[entity_id] = dict()
        self.cache[entity_id][timestep] = (state, action, state_value)

    def push_cache(self):
        for entity_id in self.cache.keys():
            for timestep in self.cache[entity_id].keys():
                if timestep + 1 in self.cache[entity_id]:
                    current_state = self.cache[entity_id][timestep]
                    next_state = self.cache[entity_id][timestep + 1]
                    self.push(current_state[0], current_state[1], next_state[0], next_state[2])

        self.cache = dict()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
