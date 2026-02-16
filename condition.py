"""
condition.py

Implementation of the BaseCondition, OriginCondition and AxisCondition classes.
"""

import numpy as np
from condevo.es.guidance import Condition
from torch import Tensor, float64, ones, tensor


class BaseCondition(Condition):
    """
    BaseCondition class docstring
    """

    def __init__(self, n_atoms, target=1.0, kwargs=None):
        Condition.__init__(self)
        self.n_atoms = n_atoms
        self.target = target
        self.kwargs = kwargs

    def condition(self, x):
        # Return default array filled with True values
        g0 = np.full(x.shape, True)
        return g0

    def evaluate(self, charles_instance, x, *args, **kwargs):
        rc = np.zeros(len(x))

        # Get positions (x,y,z) for every atom of every cluster
        x = np.reshape(x, (-1, 3))

        # Apply specific condition to atom positions
        g0 = self.condition(x)
        # Group the atoms together to form the clusters
        g0 = np.reshape(g0, (-1, self.n_atoms))
        # Get only one bool value for every cluster
        g0 = np.all(g0, axis=1)

        # Set target values
        rc[g0[:]] = 1.0
        rc[~g0[:]] = -1.0

        if isinstance(x, Tensor):
            return tensor(rc, device=x.device, dtype=float64)
        return rc

    def sample(self, charles_instance, num_samples):
        return ones(num_samples, dtype=float64) * self.target

    def to_dict(self):
        return {"target": self.target, **Condition.to_dict(self)}

    def __str__(self):
        return f"BaseCondition(n_atoms={self.n_atoms}, target={self.target})"

    def __repr__(self):
        return self.__str__()


class OriginCondition(BaseCondition):
    """
    OriginCondition class docstring
    """

    def condition(self, x):
        # Calculate distance from origin for every atom of every cluster
        g0 = np.linalg.norm(x, axis=1) < self.kwargs["cond_threshold"]
        return g0

    def __str__(self):
        return f"OriginCondition(n_atoms={self.n_atoms}, target={self.target}, kwargs={self.kwargs})"


class AxisCondition(BaseCondition):
    """
    AxisCondition class docstring
    """

    def condition(self, x):
        # Calculate position on axis for every atom of every cluster
        z = np.array(x[:, self.kwargs["cond_axis"]])
        g0 = (z >= self.kwargs["cond_lower_threshold"]) & (
            z < self.kwargs["cond_upper_threshold"]
        )
        return g0

    def __str__(self):
        return f"AxisCondition(n_atoms={self.n_atoms}, target={self.target}, kwargs={self.kwargs})"
