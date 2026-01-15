import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def species_concentration(concentration, log_beta, stoichiometry, full=False):
    r"""
    Calculate the species concentrations through the mass action law.

    $$
    S_{i} = \beta_i \prod_{j=1}^{N_c} C_j^{p_{ij}}
    $$

    With $S_i$ being the concentration of the species $i$, $\beta_i$ the equilibrium constant of the species $i$,
    $C_j$ the concentration of the component $j$, and $p_{ij}$ the stoichiometric coefficient of the component $j$ in the species $i$.

    Parameters
    ----------
    concentration : numpy.ndarray
        The concentration array of shape (n, c+p), where n is the number of points c is the number of components and p is the number of solid species.
    log_beta : numpy.ndarray
        The logarithm of the equilibrium constants with shape (n, s), where s is the number of solid species.
    stoichiometry : numpy.ndarray
        The stoichiometry matrix with shape (n, s), where s is the number of soluble species.
    full : bool, optional
        If True, return the concentrations of all species including the original concentrations.
        If False, return only the concentrations of the new species.

    Returns
    -------
    numpy.ndarray
        The calculated species concentrations.

    """
    nc = stoichiometry.shape[0]
    _c = np.log10(concentration[:, :nc])

    cext = 10 ** (log_beta + _c @ stoichiometry)

    if full:
        p = np.concatenate((concentration, cext), axis=1)
    else:
        p = cext

    return p


def percent_distribution(concentration, stoichiometry, analytc, reference):
    """Transform free concentrations to relative concentrations.

    This function converts absolute concentrations to relative percent of
    concentrations with respect to a given species.

    Parameters
    ----------
    concentration: numpy.ndarray
        The extended free concentrations array.
    stoichiometry: numpy.ndarray
        Stoichimetric coefficients
    analytc: numpy.ndarray
        The total concentration array
    reference: int
        the index of the reference species

    Returns
    -------
    numpy.ndarray: The concentrations expressed as percent with
        respect to the reference species. The first dimmension is
        unchanged with respect to that of **C** but the second one is
        shorter as all components whose stoichiometric coefficient with
        respect to the reference one is zero have been removed.
    """
    stoich_ref = stoichiometry[:, reference].copy()
    stoich_ref[stoich_ref == 0] = np.inf

    analytc_ref = analytc[:, reference]
    new_c = concentration / analytc_ref[None, :]
    return 100*new_c*stoich_ref
