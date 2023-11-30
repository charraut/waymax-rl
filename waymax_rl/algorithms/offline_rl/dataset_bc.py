import collections
import numpy as np

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions'])

class Dataset(object):

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx]
                     )