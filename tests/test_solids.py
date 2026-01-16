import numpy as np
import numpy.testing as npt

from libeq import SolverData, EqSolver, species_concentration
from libeq.utils import objective_function


def test_solid_aluminium():
    FOBJ_TOLERANCE = 1e-7
    data = SolverData.load_from_pyes('tests/data/aluminium_alkaline.json')
    freec, log_beta, log_ks, saturation_index, analc = EqSolver(data, mode="distribution")

    c_soluble = freec[:,:2]
    c_solids  = freec[:,-1][:, None]

    number_components = 2
    stoichx = np.vstack((np.eye(number_components, dtype=int), data.stoichiometry.T))
    freecx = species_concentration(c_soluble, data.log_beta, data.stoichiometry, full=True)
    f = objective_function(analc, freecx, stoichx, c_solids, data.solid_stoichiometry.T, log_ks)
    npt.assert_allclose(f, np.zeros_like(f), atol=FOBJ_TOLERANCE)   
