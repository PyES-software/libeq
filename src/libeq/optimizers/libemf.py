"""Routines for constant fitting from potentiometric data.

Module :mod:`libemf` contains the routines needed for fitting equilibrium
constants from potentiometric data. The main function to invoke here is
:func:`emffit` which handles most of the work. This is the only public
function for this module.

Author: Salvador Blasco <salvador.blasco@uv.es>
"""


import numpy as np
from numpy.typing import NDArray

NERNST = 25.6926      # mV
RoverF = 0.086173424  # mV/K
FArray = NDArray[float]


def hselect(array: FArray, hindices: list[int]) -> FArray:
    """Select columns that correspond to the electroactive species.

    Given the concentrations array, selects the columns that correspond
    to the electroactive species.

    Parameters:
        array (:class:`numpy.ndarray`): The :term:`free concentrations array`
        hindices (list[int]): Indices of the electroactive specie(s).

    Returns:
        The part of C which is electroactive
    """
    return array[np.arange(len(hindices)),hindices]


def nernst(electroactive_conc: FArray, emf0: FArray, slope: FArray | float = 1.0,
           joint: FArray | float=0.0) -> FArray:
    r"""Calculate the calculated potential.

    Apply Nernst's equation to calculate potential according to
    .. math ::

    E = E^0 + f\frac{nF}{RT}\ln[C] + J

    Parameters:
        electroactive_conc (:class:`numpy.ndarray`): a 1D array of floats
            representing the free concentrations of the electroactive species.
        emf0 (:class:`numpy.ndarray`): The :term:`standard potential`
        slope (:class:`numpy.ndarray`): The slope for :term:`Nernst's equation`
        joint (:class:`numpy.ndarray`): The liquid joint contribution for :term:`Nernst's equation`
        temperature (float): the absolute temperature
    Returns:
        :class:`numpy.ndarray`: an array of floats containing the calculated values
    """
    return emf0 + slope*np.log(electroactive_conc) + joint


def emf_jac_beta(dlogc_dlogbeta: FArray, slope=1.0, temperature: float=298.15) -> FArray:
    r"""Calculate the jacobian part related to equilibrium constants.

    The calculation is done according to equation
    .. math ::

    \frac{\partial E_n}{\partial\log\beta_b}=\frac{fRT}{nF\log e}\frac{\partial\log c_{nh}}
    {\partial\log\beta_b} 

    Parameters:
        dlogc_dlogbeta (:class:`numpy.ndarray`): the derivative values. They can be obtained
            from :func:`libeq.jacobian.dlogcdlogbeta`.
        slope (:class:`numpy.ndarray`): The slope for :term:`Nernst's equation`
        temperature (float): the absolute temperature
    Returns:
        :class:`numpy.ndarray`: an array of floats containing the calculated values
    """
    nernstian_slope = slope*RoverF*temperature
    return nernstian_slope*dlogc_dlogbeta


def emf_jac_init(dlogc_dt: FArray, slope=1.0, temperature=298.15) -> FArray:
    r"""Calculate the jacobian part related to the initial amount.

    The calculation is done according to equation
    .. math ::

    \frac{\partial E_n}{\partial t_i} = f\frac{RT}{nF}\frac{\partial\log c_{nh}}{\partial t_i}

    Parameters:
        dlogc_dt (:class:`numpy.ndarray`): the derivative values. They can be obtained
            from :func:`libeq.jacobian.dlogcdt`.
        slope (:class:`numpy.ndarray`): The slope for :term:`Nernst's equation`
        temperature (float): the absolute temperature
    Returns:
        :class:`numpy.ndarray`: an array of floats containing the calculated values
    """
    nernstian_slope = slope*RoverF*temperature
    return nernstian_slope*dlogc_dt


def emf_jac_buret(dlogc_db: FArray, slope=1.0, temperature=298.15) -> FArray:
    r"""Calculate the jacobian part related to the buret concentration.

    The calculation is done according to equation
    .. math ::

    \frac{\partial E_n}{\partial b_i} = f\frac{RT}{nF}\frac{\partial\log c_{nh}}{\partial b_i}

    Parameters:
        dlogc_db (:class:`numpy.ndarray`): the derivative values. They can be obtained
            from :func:`libeq.jacobian.dlogcdb`.
        slope (:class:`numpy.ndarray`): The slope for :term:`Nernst's equation`
        temperature (float): the absolute temperature
    Returns:
        :class:`numpy.ndarray`: an array of floats containing the calculated values
    """
    nernstian_slope = slope*RoverF*temperature
    return nernstian_slope*dlogc_db


def emf_jac_e0(size: int) -> FArray:
    r"""Calculate the jacobian part related to the standard potential.

    It returns ones based on the size according to equation
    .. math ::

    \frac{\partial E_n}{\partial E^0} = 1

    Parameters:
        size (int): the number of ones to return
    Returns:
        :class:`numpy.ndarray`: an array of floats containing the calculated values
    """
    return np.ones(size)


def emf_weights(titre: FArray, titre_error: float, emf: FArray, emf_error: float) -> FArray:
    """Compute per-point weights using the Gans *et al.* propagation formula.

    The weight at each point is defined as

    .. math::

        w_n = \\frac{1}{\\sigma_E^2 + \\left(\\frac{\\partial E}{\\partial V}\\right)_n^2 \\sigma_V^2}

    where the gradient is estimated numerically from the data.

    Parameters
    ----------
    titre : numpy.ndarray
        Titrant volumes at each point, shape ``(n_points,)``.
    titre_error : float
        Standard deviation of the titre measurement (mL).
    emf : numpy.ndarray
        Measured EMF values, shape ``(n_points,)``.
    emf_error : float
        Standard deviation of the EMF measurement (mV).

    Returns
    -------
    numpy.ndarray
        1-D weight array of shape ``(n_points,)``.
    """
    gradient = np.gradient(emf, titre, axis=0)
    return 1/(emf_error**2 + gradient**2 * titre_error**2)


def residual_jacobian(emf: FArray, calc_emf: FArray, weights: FArray, demfdx) -> FArray:
    """Compute the gradient of the weighted sum-of-squares with respect to parameters.

    Evaluates :math:`-2 \\sum_n w_n (E_n - \\hat{E}_n) E_n \\frac{\\partial \\hat{E}}{\\partial x}`,
    which is used as the right-hand side in the Gauss–Newton normal equations.

    Parameters
    ----------
    emf : numpy.ndarray
        Experimentally measured EMF values, shape ``(n_points,)``.
    calc_emf : numpy.ndarray
        Calculated (model) EMF values, shape ``(n_points,)``.
    weights : numpy.ndarray
        Per-point weights, shape ``(n_points,)``.
    demfdx : numpy.ndarray
        Jacobian of the calculated EMF with respect to the parameters,
        shape ``(n_points, n_params)`` or ``(n_points, p, q)``.

    Returns
    -------
    numpy.ndarray
        Gradient array with the same shape as the last dimensions of
        *demfdx* after summing over the first (points) dimension.
    """
    # breakpoint()
    aux = np.sqrt(weights)*(emf - calc_emf)*emf
    return -2*np.sum(aux[:,None,None]*demfdx, axis=0)

