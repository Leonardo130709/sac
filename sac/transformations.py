import numpy as np
import torch


class Transformations:
    def __init__(self, *transformations_list):

        if isinstance(transformations_list, list):
            self.T = transformations_list
        else:
            self.T = list(transformations_list)

    def __call__(self, trajectory):
        for T in self.T:
            trajectory = T(trajectory)
        return trajectory


class Next:
    # keep in mind last value gives wrong value
    def __init__(self, keys=['states'], roll=-1, axis=0):
        self._axis = axis
        self._roll = roll
        self._keys = keys

    def __call__(self, trajectory):
        for k in self._keys:
            trajectory[f'next_{k}'] = np.roll(trajectory[k], self._roll, axis=self._axis)
        return trajectory


class Truncate:
    def __init__(self, amount, axis):
        self.amount = amount
        self.axis = axis

    def __call__(self, trajectory):
        for k, v in trajectory.items():
            v = np.swapaxes(v, 0, self.axis)[:-self.amount]
            trajectory[k] = np.swapaxes(v, 0, self.axis)
        return trajectory


class Concatenate:
    def __init__(self, axis=0):
        self.axis = axis
        self._op = None

    def __call__(self, trajectory):
        if self._op is None:
            self._build(trajectory)

        for k, v in trajectory.items():
            trajectory[k] = self._op(v, axis=self.axis)
        return trajectory

    def _build(self, trajectory):
        if torch.is_tensor(trajectory['actions']):
            self._op = torch.cat
        else:
            self._op = np.concatenate

class Transpose:
    def __init__(self, d0, d1):
        self._d0 = d0
        self._d1 = d1
        self._op = None

    def __call__(self, trajectory):
        if self._op is None:
            self._build(trajectory)

        for k, v in trajectory.items():
            trajectory[k] = self._op(v, self._d0, self._d1)
        return trajectory

    def _build(self, trajectory):
        if torch.is_tensor(trajectory['actions']):
            self._op = torch.transpose
        else:
            self._op = np.moveaxis
