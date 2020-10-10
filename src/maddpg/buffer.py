
import random
import torch
from collections import deque, namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=int(buffer_size))  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["obs", "obs_full", "actions_for_env", "rewards",
                                                                "next_obs", "next_obs_full", "dones"])
        random.seed(seed)

    def add(self, obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones):
        """Add a new experience to memory."""
        e = self.experience(obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        transitions = random.sample(self.memory, k=batch_size)

        obs = torch.stack([e.obs for e in transitions if e is not None]).float()
        obs_full = torch.stack([e.obs_full for e in transitions if e is not None]).float()
        actions_for_env = torch.tensor([e.actions_for_env for e in transitions if e is not None]).float()
        rewards = torch.tensor([e.rewards for e in transitions if e is not None]).float()
        next_obs = torch.stack([e.next_obs for e in transitions if e is not None]).float()
        next_obs_full = torch.stack([e.next_obs_full for e in transitions if e is not None]).float()
        dones = torch.tensor([e.dones for e in transitions if e is not None]).float()

        return obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# class ReplayBuffer:
#     def __init__(self,size):
#         self.size = size
#         self.deque = deque(maxlen=self.size)
#
#     def push(self, transition):
#         """push into the buffer"""
#
#         input_to_buffer = transpose_list(transition)
#
#         for item in input_to_buffer:
#             self.deque.append(item)
#
#     def sample(self, batchsize):
#         """sample from the buffer"""
#         samples = random.sample(self.deque, batchsize)
#
#         # transpose list of list
#         return transpose_list(samples)
#
#     def __len__(self):
#         return len(self.deque)



