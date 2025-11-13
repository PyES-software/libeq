import numpy as np

#from libaux import assert_array


def cpluss(concentration, beta, stoichiometry, full=False, logc=False):
    r"""Compute the free concentrations for the extended components.

    The concentration of the complexes is calculated by means of the
    equilibrium condition.

    .. math::`c_{i+S} = \beta_i \prod_{j=1}^E c_j^p_{ji}`

    This is an auxiliary function.

    Parameters:
        concentration (:class:`numpy.ndarray`): The free concentrations of the
            free components. It must be an (*S*,) or (*N*, *S*)-sized array
            where *S* is the number of free components. This parameter can be
            a masked array. In this case, the return concentration matric will
            also be masked.
        stoichiometry (:class:`numpy.ndarray`): The stoichiometric coefficient
            matrix. It must be (*E*, *S*)-sized where E is the number of
            equilibria.
        beta (:class:`numpy.ndarray`): The equilibrium constants. The last
            dimmension must be E-sized and the rest of the dimmensions must be
            compatible with those of **concentration**.
        full (bool): If set, the return array will be the full (*N*, *S* + *E*)
            array. If unset only the extra calculated array (*N*, *E*) will be
            returned.
        logc (bool): If True, the natural logarithms of the concentrations are
            expected. Otherwise, work with regular values for the
            concentration.
    Returns:
        :class:`numpy.ndarray`: array of size (*N*, *E*) containing the
            extended concentrations

    Raises:
        ValueError: If any parameter is incorrect.
    """
    # remove because this routine is called many many times
    #assert_array(concentration, beta, stoichiometry)

    if np.ma.is_masked(concentration):
        np_log = np.ma.log
        np_dot = np.ma.dot
        np_exp = np.ma.exp
        np_concatenate = np.ma.concatenate
    else:
        np_log = np.log
        np_dot = np.dot
        np_exp = np.exp
        np_concatenate = np.concatenate

    # concentration[concentration <= 0] = sys.float_info.min
    if logc:
        _c = concentration
    else:
        _c = np_log(concentration)
    cext = np_log(beta) + np_dot(_c, stoichiometry.T)

    if full:
        p = np_concatenate((_c, cext), axis=1)
    else:
        p = cext

    # if logc:
    #     return p
    # else:
    #     # return np.nan_to_num(np.exp(p))     # replace NaN with 0
    #     return np_exp(p)

    return p if logc else np_exp(p)


def mass_action_solid(concentration, solubility_stoich):
    logc = np.log(concentration)
    return np.exp(np.dot(logc, solubility_stoich.T))
