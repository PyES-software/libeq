from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from libeq.consts import Flags
from libeq.data_structure import SolverData


FArray = NDArray[np.float32 | np.float64]


class Bridge(ABC):
    def __init__(self, data: SolverData, reporter: Callable) -> None:
        assert hasattr(self, "_exp_data_handler")
        self._titration_list = getattr(data, self._exp_data_handler).titrations
        self._data = data
        self._reporter = reporter

        self._freeconcentration: FArray | None = None

        self._stoich = self._stoichiometry(extended=False)
        self._stoichx = self._stoichiometry(extended=True)
        self._nspecies, self._ncomponents = self._stoich.shape
        self._ntitrations = len(data.potentiometry_opts.titrations)
        self._experimental_points = [ len(t.get_titre) for t in self._titrations() ]
        self._total_points = sum(self._experimental_points)
        self._chargesx = np.sum(self._stoichx*data.charges, axis=1)

        # calculate degrees of freedom related to equilibrium constants
        self._dof_beta = sum(1 for _ in data.potentiometry_opts.beta_flags if _ == Flags.REFINE)

        # define variables from the children class
        self._variable: FArray
        self._step: FArray
        self._previous_values: FArray

    def accept_step(self) -> None:
        """Commit the current trial step and reset the increment buffer to zero."""
        self._previous_values[:] = self._variables
        self._variables += self._step
        self._step[...] = 0.0

    @abstractmethod
    def final_result(self) -> dict:
        """Compile and return the full refinement result as a dictionary.

        Returns
        -------
        dict
            Key–value pairs describing the converged parameter values,
            concentrations, residuals, and diagnostic information.
        """
        ...

    @abstractmethod
    def matrices(self) -> tuple[FArray, FArray]:
        """Compute and return the Jacobian and residual arrays for the accepted step.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            ``(J, r)`` where *J* is the Jacobian of shape
            ``(n_points, n_params)`` and *r* is the residual vector of shape
            ``(n_points,)``.
        """
        ...

    def reject_step(self) -> None:
        """Discard the current trial step and reset the increment buffer to zero."""
        self._step[...] = 0.0

    @abstractmethod
    def size(self) -> tuple[int, int]:
        """Return the dimensions of the fitting problem.

        Returns
        -------
        tuple of (int, int)
            ``(n_points, n_params)`` — total number of experimental
            observations and number of refinable parameters.
        """
        ...

    @abstractmethod
    def trial_step(self, increments: FArray) -> None:
        """Store parameter increments for the next trial step.

        Parameters
        ----------
        increments : numpy.ndarray
            Proposed parameter increments, shape ``(n_params,)``.
        """
        ...

    @abstractmethod
    def tmp_residual(self) -> FArray:
        """Compute and return the residual vector for the current trial step.

        Returns
        -------
        numpy.ndarray
            Residual vector of shape ``(n_points,)``.
        """
        ...

    @abstractmethod
    def weights(self) -> FArray:
        """Return the weights matrix used in the objective function.

        Returns
        -------
        numpy.ndarray
            Square diagonal weights matrix of shape ``(n_points, n_points)``.
        """
        ...

    def _stoichiometry(self, extended=False):
        "Get stoichiometry array."
        number_components = self._data.stoichiometry.shape[0]
        if extended:
            return np.vstack((np.eye(number_components, dtype=int),
                              np.array(self._data.stoichiometry.T)))
        return self._data.stoichiometry.T

    def _titrations(self):
        """
        Iterate over the titrations.
        """
        yield from iter(self._titration_list)


def ravel(x, y, flags):
    """Update values from one iterable with other iterable according to flags.

    This function does the opposite action than :func:`unravel`.

    Parameters:
        x (iterable): the original array values
        y (iterable): the updated values to be plugged into *x*.
        flags (sequence): flags indicating how to update *x* with *y*. Accepted
            values are:

            * 0: value is to be kept constant
            * 1: value is to be refined and the corresponding value from x
              will be substituted by the corresponding value from y.
            * >2: value is restrained. All places with the same number are
              refined together and the ratio between them is maintained.

    Yields:
        float: Raveled values.
    """
    # indices of the reference parameter for constraining
    ref_index = {i: flags.index(i) for i in range(2, 1 + max(flags))}
    ref_val = {}

    ity = iter(y)
    for i, f in enumerate(flags):
        if f == Flags.REFINE:  # refinable: return new value
            yield next(ity)
        elif f == Flags.CONSTANT:  # constant: return old value
            yield x[i]
        else:  # constrained: return or compute
            if i == ref_index[f]:
                val = next(ity)  # reference value: return new value
                ref_val[f] = val  # and store ref value
                yield val
            else:  # other: compute proportional value
                yield x[i] * ref_val[f] / x[ref_index[f]]
