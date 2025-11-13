import numpy as np

from .cpluss import cpluss, mass_action_solid


def fobj(concentration, stoichiometry, analyticalc, beta=None):
    r"""Return the value of the function to be minimized.

    This function is defined as
    :math:`f_i = c_i + \sum_{j=1}^E p_{ij}c_{j+S} - T_i`

    Parameters:
        concentration (:class:`numpy.ndarray`): free concentrations for all the
            species in mmol/mL. It must be an array of
            (*N*, *S* + *E* ) floats.
        stoichiometry (:class:`numpy.ndarray`): The stoichiometric coefficients
            array
        analyticalc (:class:`numpy.ndarray`): The total concentrations of the
            free components in mmol/mL. It must be an array of (*N*, *S*)
            floats.

    Returns:
        :class:`numpy.ndarray`: array of (*N*, *S*) floats containing the
            values of the function.
    """
    if np.ma.is_masked(concentration):
        np_sum = np.ma.sum
    else:
        np_sum = np.sum

    n_species = stoichiometry.shape[1]
    if concentration.shape[1] == n_species:
        c1 = concentration
        c2 = cpluss(concentration, beta, stoichiometry)
    else:
        c1 = concentration[..., :n_species]
        c2 = concentration[..., n_species:]

    return c1 + np_sum(c2[..., np.newaxis]*stoichiometry[np.newaxis, ...],
                       axis=1) - analyticalc


def fobj_solid(concentration, stoich_soln, stoich_solid, analyticalc, beta, solubility_product):
    r"""Return the value of the objective function to be minimized for solids.

    This objective function contains also information for solids.

    
    Parameters:
        concentration (:class:`numpy.ndarray`): free concentrations for all the
            species in mmol/mL. It must be an array of
            (*N*, *S* + *E1* + *E2* ) floats.
        stoichiometry (:class:`numpy.ndarray`): The stoichiometric coefficients
            array. It must be an array of (*E1*+*E2*,*S*). 
        analyticalc (:class:`numpy.ndarray`): The total concentrations of the
            free components in mmol/mL. It must be an array of (*N*, *S*)
            floats.

    Returns:
        :class:`numpy.ndarray`: array of (*N*, *S*) floats containing the
            values of the function.

    """
    n_points, n_components = stoich_soln.shape
    n_solids = len(solubility_product)
    n_equils = len(beta)

    c_comps = concentration[:,:n_components]
    c_solut = concentration[:,:(n_components+n_equils)]
    c_solid = concentration[:,(n_components+n_equils+1):]

    raw_f = fobj(c_solut, stoich_soln, analyticalc)
    solid_f = solid_factor(c_solid, stoich_solid)
    raw_g = gobj(c_comps, stoich_solid, solubility_product)

    return np.vstack((raw_f + solid_f, raw_g))


def gobj(concentration, stoichiometry, solubility_product):
    r"""Return the value of the function to be minimized for solids.

    This function is defined as
    :math:`g_i = \prod_k c_k^(q_{ik}) - Ks_i`

    Parameters:
        concentration (:class:`numpy.ndarray`): free concentrations for all the
            species in mmol/mL. It must be an array of
            (*N*, *S*) floats.
        stoichiometry (:class:`numpy.ndarray`): The stoichiometric coefficients
            array for precipitation equilibria.
        solubility_product (:class:`numpy.ndarray`): an array with values
            for the solubility_product.

    Returns:
        :class:`numpy.ndarray`: array of (*N*, *S*) floats containing the
            values of the function.
    """
    aux = mass_action_solid(concentration, stoichiometry)
    return aux / solubility_product - 1


def solid_factor(conc_solid, stoich_solid):
    """Factor containing the contribution of the precipitated solid.

    This term is defined as :math:`\sum_j^{E_2} q_{ji}C_j`
    """
    return np.dot(conc_solid, stoich_solid)
