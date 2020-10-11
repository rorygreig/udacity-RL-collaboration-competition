from collections import deque, namedtuple
import random
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor([e.states for e in experiences if e is not None]).float()
        actions = torch.tensor([e.actions for e in experiences if e is not None]).float()
        rewards = torch.tensor([e.rewards for e in experiences if e is not None]).float()
        next_states = torch.tensor([e.next_states for e in experiences if e is not None]).float()
        dones = torch.tensor([e.dones for e in experiences if e is not None]).float()

        return states, actions, rewards, next_states,  dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
