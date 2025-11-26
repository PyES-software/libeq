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
from . import libmath


FArray = NDArray[float]


class Exec(enum.IntEnum):
    "Flag class for execution status."
    INITIALISING = enum.auto()
    RUNNING = enum.auto()
    NORMAL_END = enum.auto()
    TOO_MANY_ITERS = enum.auto()
    ABNORMAL_END = enum.auto()
    SINGULAR_MATRIX = enum.auto()


def levenberg_marquardt(bridge, **kwargs) -> Dict[str, np.ndarray]:
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
    CHISQ_THRESHOLD: Final[float] = kwargs.pop('chisq_threshold', 1e-2)
    GRAD_THRESHOLD: Final[float] = kwargs.pop('grad_threshold', 1e-4)
    STEP_THRESHOLD: Final[float] = kwargs.pop('step_threshold', 1e-4)
    RHO_THRESHOLD: Final[float] = kwargs.pop('rho_threshold', 1e-4)
    MAX_ITERATIONS: Final[int] = kwargs.pop('max_iterations', 20)

    damping: float = kwargs.pop('damping', DAMPING0)
    debug: bool = kwargs.pop('debug', False)

    n_points, n_vars = bridge.size()

    iteration: int = 0
    gradient_norm: float = 0.0
    W: FArray = bridge.weights()
    sigma: float = math.inf
    execution_status: Exec = Exec.INITIALISING
    J: FArray
    M: FArray = np.array([])
    D: FArray = np.array([])

    dx = np.zeros(n_vars)
    bridge.trial_step(dx)
    J, resid = bridge.matrices()
    gradient: FArray = 2*J.T @ W @ resid
    gradient_norm: float = float(np.linalg.norm(gradient))
    M: FArray = J.T @ W @ J
    D: FArray = np.diag(np.diag(M))
    chisq = float(resid.T @ W @ resid)

    execution_status: Exec = Exec.RUNNING

    while iteration < MAX_ITERATIONS:
        try:
            dx = np.linalg.solve(M+damping*D, -gradient)
        except np.linalg.LinAlgError as exc:
            execution_status = Exec.SINGULAR_MATRIX
            exception_thrown = exc
            break

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

            sigma = fit_sigma(resid, np.diag(W), n_points, n_vars)

            # exit criteria check
            exit_chi_value = chisq/bridge.degrees_of_freedom
            exit_chi = exit_chi_value  < CHISQ_THRESHOLD
            exit_gradient_value = abs(max(gradient))
            exit_gradient = gradient_norm  < GRAD_THRESHOLD
            exit_step_value = max(np.abs(r) for r in bridge.relative_change(dx))
            exit_step = exit_step_value < STEP_THRESHOLD

            bridge.report_step(iteration=iteration, 
                               increment=dx,
                               damping=damping, 
                               chisq=exit_chi_value,
                               sigma=sigma,
                               gradient_norm=gradient_norm,
                               exit_chi=exit_chi,
                               exit_gradient_value=exit_gradient_value,
                               exit_gradient=exit_gradient,
                               exit_step_value=exit_step_value,
                               exit_step=exit_step)

            if debug:
                print(f"iteration={iteration-1}, {damping=:.2e}, {rho=:.4e}, {sigma=:.4e}, {chisq=:.4e}")
                print(f"\t{dx=}\n\tx={bridge._variables}")


            # Adaptive damping (very effective)
            if rho > 0.75:
                damping = max(damping / 3.0, DAMPING_LOWER)
            elif rho > 0.25:
                damping = max(damping / 2.0, DAMPING_LOWER)
        else:
            # step REJECTED 
            bridge.reject_step()
            damping = damping*DAMPING_UPF
            if damping > DAMPING_UPPER:
                execution_status = Exec.ABNORMAL_END
                break
            continue

            if math.isnan(rho) or any(np.isnan(dx)):
                execution_status = Exec.ABNORMAL_END
                break

        if exit_chi:
            execution_status = Exec.NORMAL_END
            if debug:
                print(f"END: threshold   {chisq/bridge.degrees_of_freedom}<{chisq_threshold}")
                print(f"\t{dx=}\n\tx={bridge._variables}")
            # bridge.report_raw(f" refinent finished on threshold criteria [{rho}<{chisq_threshold}]\n")
            break

        if exit_gradient:
            execution_status = Exec.NORMAL_END
            if debug:
                print(f"END: gradient   {gradient_norm}<{grad_threshold}")
                print(f"\tx={bridge._variables}")
            # bridge.report_raw(f"refinent finished on gradient criteria [{gradient_norm}<{grad_threshold}]\n")
            break

        if exit_step:
            execution_status = Exec.NORMAL_END
            if debug:
                print(f"END: step   {step_size}<{step_threshold}")
                print(f"\tx={bridge._variables}")
            # bridge.report_raw(f"refinent ended on small step criteria [{step_size}<{step_threshold}]\n")
            break

        iteration += 1
    else:
        execution_status = Exec.TOO_MANY_ITERS

    ret = {'jacobian': J,
           'residuals': resid,
           'damping': damping,
           'iterations': iteration}
    if execution_status == Exec.TOO_MANY_ITERS:
        raise excepts.TooManyIterations(msg=("Maximum number of iterations reached"),
                                        last_value=ret)
    if execution_status == Exec.ABNORMAL_END:
        raise excepts.UnstableIteration(msg=("The iteration is not stable"),
                                        last_value=ret)
    if execution_status == Exec.SINGULAR_MATRIX:
        raise exception_thrown
    return ret


def fit_sigma(residuals: np.ndarray, weights: np.ndarray, npoints: int, nparams: int) -> float:
    """Calculate the fit's sigma value for a given set of residuals and weights."""
    return np.sum(weights*residuals**2)/(npoints-nparams)


def is_near_singular_lstsq(matrix, thresh=1e-3):
    """
    Test whether the matrix is near singular.
    """
    result = np.linalg.lstsq(matrix, np.eye(matrix.shape[1]), rcond=thresh)
    rank = result[2]                   # third return value = rank in NumPy ≥ 2.0
    s = result[3]                      # singular values
    rcond_est = s[-1] / s[0] if len(s) > 1 else 0.0
    print(f">>> {rcond_est}  {rank}")
    return rcond_est < thresh or rank < min(matrix.shape)

