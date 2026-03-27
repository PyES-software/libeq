import numpy as np
import numpy.testing as npt

from libeq.optimizers.jacobian import amatrix

from .fixtures import synthetic_simple1_wosld

def test_amatrix1(synthetic_simple1_wosld):
    # analyticalc = np.array([[1.0, 0.0],[0.3, 0.7], [0.5, 0.5],[0.7,0.3],[0.0,1.0]])
    # stoichiometry = np.array([[1,1]], dtype=int)
    # solid_stoichiometry = np.array([[]], dtype=int)
    # log_beta = np.log10(np.array([1.0]))
    # log_ks = np.array([])
    # #                       c1           c2           c3         
    # real_freec = np.array([[1.000000000, 0.000000000, 0.000000000],
    #                        [0.188819442, 0.588819442, 0.111180558],
    #                        [0.366025404, 0.366025404, 0.133974596],
    #                        [0.588819442, 0.188819442, 0.111180558],
    #                        [0.000000000, 1.000000000, 0.000000000]])

    stoichx = np.vstack([np.eye(2), synthetic_simple1_wosld.stoichiometry])
    reala = np.array([[[1.00000000, 0.00000000], [0.00000000, 0.00000000]],
                      [[0.70000000, 0.11118056], [0.11118056, 0.30000000]],
                      [[0.50000000, 0.13397460], [0.13397460, 0.50000000]],
                      [[0.30000000, 0.11118056], [0.11118056, 0.70000000]],
                      [[0.00000000, 0.00000000], [0.00000000, 1.00000000]]])

    calca = amatrix(synthetic_simple1_wosld.real_freec, stoichx)
    npt.assert_allclose(reala, calca)

