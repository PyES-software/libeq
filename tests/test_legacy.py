"Test collection for legacy file import"

import numpy as np
import numpy.testing as npt

from libeq import SolverData, Flags

def test_import_superquad():
    parsed_data = SolverData.load_from_superquad("tests/data/hedtac.sup")
    assert parsed_data.components[0] == 'EDTA'
    assert parsed_data.components[1] == 'PROTON'
    assert parsed_data.temperature == 298.15

    test_stoich = np.array([[1,1,1,1,0],[1,2,3,4,-1]], dtype=int)
    npt.assert_equal(parsed_data.stoichiometry, test_stoich)
    npt.assert_equal(parsed_data.solid_stoichiometry, np.array([[],[]], dtype=int))
    npt.assert_equal(parsed_data.log_beta, np.array([11.0, 17.0, 20.0, 22.0, -13.73]))
    npt.assert_equal(parsed_data.log_ks, np.array([]))

    popts = parsed_data.potentiometry_opts
    assert popts.beta_flags == [Flags.REFINE, Flags.REFINE, Flags.REFINE,
                                Flags.REFINE,Flags.CONSTANT]
    assert popts.weights == 'calculated'


def test_import_bstac():
    data = SolverData.load_from_bstac("tests/data/namototc.mis")
