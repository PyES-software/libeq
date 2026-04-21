"""
Optimizer bridge utilities.

This module defines the :class:`~libeq.optimizers.bridge.Bridge` abstract base
class used by optimizers to interact with experimental datasets and the solver
state in a uniform way. Concrete bridge implementations adapt a specific
experimental technique (e.g., potentiometry) to the generic needs of a
least-squares / iterative refinement engine by providing:

- the current parameter vector and trial increments,
- residual vectors for accepted and trial steps,
- Jacobian and residual matrices for linearized updates,
- weights used in the objective function,
- problem dimensions and final result packaging.

The module also provides :func:`ravel`, a helper to reconstruct a full parameter
array from a compact vector of refinable parameters according to per-parameter
flags (constant, refine, or constrained groups).

Notes
-----
Bridges are expected to be subclassed. Subclasses must define the attribute
``_exp_data_handler`` (the name of the attribute in :class:`~libeq.data_structure.SolverData`
that holds the experimental dataset), and implement the abstract methods
declared by :class:`Bridge`.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from libeq.consts import Flags
from libeq.data_structure import SolverData


FArray = NDArray[np.float32 | np.float64]


class Bridge(ABC):
    """
    Abstract adapter between a refinement algorithm and an experimental dataset.

    A ``Bridge`` encapsulates the bookkeeping required to run an iterative fit:
    it exposes residuals, Jacobians, weights, and the total problem size, and it
    manages the distinction between *accepted* parameter values and a *trial*
    step (stored as increments in ``_step``).

    The constructor extracts and precomputes common quantities from
    :class:`~libeq.data_structure.SolverData`, including stoichiometry matrices,
    counts of species/components/titrations/points, charges for the extended
    stoichiometry, and the number of refinable equilibrium constants.

    Subclass contract
    -----------------
    Subclasses must:

    - define ``_exp_data_handler`` before calling ``Bridge.__init__``; it must
      be the name of a ``SolverData`` attribute whose value provides a
      ``titrations`` iterable,
    - define and maintain the following arrays used by the base methods:

      * ``_variables`` (current accepted parameter values),
      * ``_previous_values`` (buffer for rollback/diagnostics),
      * ``_step`` (trial increments to apply on accept),

    - implement all abstract methods: :meth:`matrices`, :meth:`tmp_residual`,
      :meth:`trial_step`, :meth:`weights`, :meth:`size`, and :meth:`final_result`.

    Parameters
    ----------
    data : SolverData
        Full solver state and experimental options.
    reporter : Callable
        Callback used by higher-level optimizers to report progress and/or
        diagnostics.

    Notes
    -----
    :meth:`accept_step` commits ``_step`` into ``_variables`` and resets the
    increment buffer to zero. :meth:`reject_step` only resets the increment
    buffer.
    """
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
        self._variables: FArray
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
