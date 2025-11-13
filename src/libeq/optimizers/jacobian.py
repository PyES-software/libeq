import numpy as np

from . import fobj


def jacobian(concentration, stoichiometry, logc=False):
    r"""Compute the jacobian array.

    This function computes the jacobian for the function :func:`fobj`,
    which is defined as

    .. math:: J = \left( \begin{array}{ccc}
       \frac{\partial f_0}{\partial c_0} & \cdots &
       \frac{\partial f_0}{\partial c_S} \\
       \vdots  & \ddots & \vdots  \\
       \frac{\partial f_N}{\partial c_0} & \cdots &
       \frac{\partial f_N}{\partial c_S} \\
      \end{array} \right)

    where :math:`f_i = c_i + \sum_{j=1}^E p_{ij}c_{j+S} - T_i`
    and therefore

    .. math:: J_{ij} = \delta_{ij} + c_j^{-1} \sum_{k=1}^E {p_{ki} p_{kj}
       c_{k+S}}

    Parameters:
        concentration (:class:`ndarray`): the :term:`free concentrations array`
            for every component.  It must be an (*N*, *E* + *S* )-sized array
            of floats.
        stoichiometry (:class:`ndarray`): The :term:`stoichiometry array`.
            It must be an (*E*, *S*)-sized array.
        log (bool): If True, the returned result will be
            :math:`J_{ij} = \frac{\partial f_i}{\partial\log c_j}`. If False
            (default) the returned result will be
            :math:`J_{ij} = \frac{\partial f_i}{\partial c_j}`
    Returns:
        :class:`ndarray`: An (*E*, *E*)-sized array which is the jacobian
            matrix.
    """
    n_species = stoichiometry.shape[1]
    aux1 = np.einsum('ij,ik,li->ljk', stoichiometry, stoichiometry,
                     concentration[:, n_species:])
    aux2 = np.eye(n_species)
    aux3 = concentration[:, np.newaxis, :n_species]
    if logc:
        return aux2 * aux3 + aux1
    else:
        return aux2 + aux1/aux3


def jacobian_solid(concentration, stoichiometry, solubility_stoich, solubility_product):
    jf1 = jacobian(concentration, stoichiometry)
    jf2 = jacobian_f_solid(solubility_stoich)
    jg1 = jacobian_g_solid(concentration, solubility_stoich, solubility_product)
    z = np.zeros((jf1.shape[0], jf2.shape[1], jg1.shape[2]))
    return np.block([[jf1, jf2],[jg1, z]])


def jacobian_f_solid(solubility_stoich):
    return solubility_stoich.T


def jacobian_g_solid(concentration, solubility_stoich, solubility_product):
    g = fobj.gobj(concentration, solubility_stoich, solubility_product)
    return (1 + g[..., None])*solubility_stoich[None,...]/concentration[:,None,:]


def dlogcdlogbeta(Amatrix, concentration, stoichiometry):
    r"""Return ∂logc_k/∂logβ_i.

    It returns the solution of the lineal system:
    .. math :: \sum_{k=1}^S \left(
                 c_k\delta_{ki} + \sum_{j=1}^E {
                  p_{ji} p_{jk} c_{j+S}
                 }
                \right) \frac{\partial\log c_k}{\partial \log\beta_b}
                = -p_{bi}c_{b+S}
    """
    n_species = stoichiometry.shape[1]
    B = -stoichiometry[np.newaxis, ...] *  concentration[:, n_species:, None]
    return np.linalg.solve(Amatrix, B.swapaxes(-1,-2))


def extended_dlogcdlogbeta(dlcdlb, stoichiometry):
    r"""Return ∂logc_k/∂logβ_i for the extended species E->E+S.

    It returns the values of:
    .. math :: \frac{\partial\log c_{i+S}}{\partial\log\beta_k} =
               \delta_{ik} + \sum_{j=1}^S p_{ij}
               \frac{\partial\log c_j}{\partial\log\beta_k}
    """
    n_equil = stoichiometry.shape[0]
    return np.eye(n_equil) + np.einsum('ijk,lj->ikl', dlcdlb, stoichiometry)


def dlogcdt(Amatrix, vol, vol0):
    r"""Return ∂logc_k/∂t_i.

    The definition is the following
    .. math::

        \sum_{k=1}^S \left(
         c_k\delta_{ki} + \sum_{j=1}^E p_{ji}p_{jk} c_{j+S}
        \right) \frac{\partial\log c_k}{\partial t_j} = \frac{v_0\cdot \delta_{ij}}{v+v_0}

    The matrix **A** can be obtained from :func:`matrix_a`.

    Parameters:
        Amatrix (:class:`numpy.ndarray`): the matrix **A** which is a (N, S, S) float array.
        vol (:class:`numpy.ndarray`): the titre
        vol0 (float): the starting volume:
    Returns:
        :class:`numpy.ndarray`: an (N, S, S) array
    """
    n_points, n_species, *_ = Amatrix.shape
    B = np.eye(n_species)[np.newaxis, ...] / (vol0 + vol[:, np.newaxis, np.newaxis])
    return np.squeeze(np.linalg.solve(Amatrix, B))


def dlogcdb(Amatrix, vol, vol0):
    r"""Return ∂logc_k/∂b_i.

    The definition is the following
    .. math::

        \sum_{k=1}^S \left(
         c_k\delta_{ki} + \sum_{j=1}^E p_{ji}p_{jk} c_{j+S}
        \right) \frac{\partial\log c_k}{\partial b_j} = \frac{v\cdot\delta_{ij}}{v+v_0}

    The matrix **A** can be obtained from :func:`matrix_a`.

    Parameters:
        Amatrix (:class:`numpy.ndarray`): the matrix **A** which is a (N, S, S) float array.
        vol (:class:`numpy.ndarray`): the titre
        vol0 (float): the starting volume:
    Returns:
        :class:`numpy.ndarray`: an (N, S, S) array
    """
    n_points, n_species, *_ = Amatrix.shape
    B = vol[:, np.newaxis, np.newaxis] * np.eye(n_species)[np.newaxis, ...] / (vol0 + vol[:, np.newaxis, np.newaxis])
    return np.squeeze(np.linalg.solve(Amatrix, B))


def amatrix(concentration, stoichiometryx):
    r"""Calculate the matrix **A**.

    **A** is a matrix that apperars commonly in many equations for the elaboration
    of the jacobian. It is defined  as follows:

    .. math::

        A_{nij} = c_{nk}\delta_{ki} + \sum_{j=1}^E p_{ki}p_{jk}c_{n,j+S}
    """
    return np.einsum('ji,jk,...j->...ik', stoichiometryx, stoichiometryx, concentration)


def bmatrix_t(vol, vol0, n_species):
    B = np.eye(n_species)[np.newaxis, ...] * vol0 / (vol0 + vol[:, np.newaxis, np.newaxis])
    return B


def bmatrix_b(vol, vol0, n_species):
    B = vol[:, np.newaxis, np.newaxis] * np.eye(n_species)[np.newaxis, ...] / (vol0 + vol[:, np.newaxis, np.newaxis])
    return B


def solve_xmatrix(amatrix, bmatrix):
    return np.squeeze(np.linalg.solve(amatrix, bmatrix))
