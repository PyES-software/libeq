"""General functions for nonlinear fitting.

Author: Salvador Blasco <salvador.blasco@uv.es>
"""

import enum
import math
from typing import Dict, Final

import numpy as np
from numpy.typing import NDArray

from libeq import consts
from libeq import excepts
from .libmath import fit_sigma


FArray = NDArray[float]


class Exec(enum.IntEnum):
    "Flag class for execution and finalisation status."
    INITIALISING = enum.auto()
    RUNNING = enum.auto()
    NORMAL_END = enum.auto()
    TOO_MANY_ITERS = enum.auto()
    ABNORMAL_END = enum.auto()
    SINGULAR_MATRIX = enum.auto()


def levenberg_marquardt(bridge, **kwargs) -> tuple[Exec, dict]:
    r"""
    Non linear fitting by means of the Levenberg-Marquardt method.

    Parameters
    ----------
    bridge
        A class that handles the data behind the scene. It must implement the following
        methods:
        - accept_step()
        - reject_step()
        - trial_step(increments)
        - matrices()
        - size()
        - weights()

    Returns:
        tuple:
        - :class:`numpy.ndarray`: The refined constants in natural logarithmic
            units
        - :class:`numpy.ndarray`: The free concentrations
        - dict: Extra optional parameters

    Raises:
        ValueError: If invalid parameters are passed.
    """

    DAMPING_UPPER: Final[float] = 1e10
    DAMPING_LOWER: Final[float] = 1e-12
    DAMPING_UPF: Final[float] = 8.0
    DAMPING_LOWF: Final[float] = 3.0
    DAMPING0: Final[float] = 1e-2
    SIGMA_THRESHOLD: Final[float] = kwargs.pop('chisq_threshold', 1e-2)
    GRAD_THRESHOLD: Final[float] = kwargs.pop('grad_threshold', 1e-4)
    STEP_THRESHOLD: Final[float] = kwargs.pop('step_threshold', 1e-4)
    RHO_THRESHOLD: Final[float] = kwargs.pop('rho_threshold', 1e-4)
    MAX_ITERATIONS: Final[int] = kwargs.pop('max_iterations', 20)

    damping: float = kwargs.pop('damping', DAMPING0)

    def _gather_info():
        return {'iteration':iteration, 
                'increment':dx,
                'damping':damping, 
                'chisq':chisq,
                'sigma':sigma,
                'rho': rho,
                'gradient_norm':gradient_norm,
                'exit_sigma':exit_sigma,
                'exit_gradient_value':exit_gradient_value,
                'exit_gradient':exit_gradient,
                'exit_step_value':exit_step_value,
                'exit_step':exit_step}

    iteration: int = 0
    gradient_norm: float = 0.0
    execution_status: Exec = Exec.INITIALISING

    n_points, n_vars = bridge.size()
    dx = np.zeros(n_vars)
    bridge.trial_step(dx)
    J, resid = bridge.matrices()
    W: FArray = bridge.weights()
    gradient: FArray = 2*J.T @ W @ resid
    gradient_norm: float = float(np.linalg.norm(gradient))
    INITIAL_GRADIENT_NORM: Final[float] = gradient_norm
    M: FArray = J.T @ W @ J
    D: FArray = np.diag(np.diag(M))
    chisq = float(resid.T @ W @ resid)

    execution_status: Exec = Exec.RUNNING

    while iteration < MAX_ITERATIONS:
        dx = np.linalg.solve(M+damping*D, -gradient)    # it may raise np.linalg.LinAlgError
        bridge.trial_step(dx)
        trial_resid = bridge.tmp_residual()
        trial_chisq = float(trial_resid.T @ W @ trial_resid)

        actual = chisq - trial_chisq
        predicted = -(dx @ gradient) - 0.5 * (dx @ (damping * D @ dx))
        rho = actual/predicted if predicted > 1e-12 else 0.0

        if rho > RHO_THRESHOLD:
            # step ACCEPTED
            bridge.accept_step()

            chisq = trial_chisq
            J, resid = bridge.matrices()
            M = J.T @ W @ J
            D = np.diag(np.diag(M))
            gradient = 2*J.T @ W @ resid
            gradient_norm = float(np.linalg.norm(gradient))

            sigma: float = fit_sigma(resid, np.diag(W), n_points, n_vars)

            # exit criteria check
            exit_sigma = sigma  < SIGMA_THRESHOLD
            exit_gradient_value = abs(max(gradient))
            exit_gradient = gradient_norm  < GRAD_THRESHOLD
            exit_step_value = max(np.abs(r) for r in bridge.relative_change(dx))
            exit_step = exit_step_value < STEP_THRESHOLD

            bridge.report_step(**_gather_info())                               

            # Adaptive damping (very effective)
            if rho > 0.75:
                damping = max(damping / 2*DAMPING_LOWF, DAMPING_LOWER)
            else:
                damping = max(damping / DAMPING_LOWF, DAMPING_LOWER)
        else:
            # step REJECTED 
            bridge.reject_step()
            damping = damping*DAMPING_UPF
            if damping > DAMPING_UPPER:
                execution_status = Exec.ABNORMAL_END
                break
            continue

        if any((exit_sigma, exit_gradient, exit_step)):
            execution_status = Exec.NORMAL_END
            break

        iteration += 1
    else:
        raise excepts.TooManyIterations(msg=("Maximum number of iterations reached"),
                                        last_value=_gather_info())

    if execution_status == Exec.ABNORMAL_END:
        raise excepts.UnstableIteration(msg=("The iteration is not stable"),
                                        last_value=_gather_info())

    return execution_status, _gather_info()
