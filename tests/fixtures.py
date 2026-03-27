from collections import namedtuple

import numpy as np
import pytest


data = namedtuple('data', ['analyticalc', 'stoichiometry', 'log_beta',
                           'solid_stoichiometry', 'log_ks',
                           'real_freec', 'real_jac', 'gfun'])

@pytest.fixture
def synthetic_simple1_wsld():
    analyticalc = np.array([[1.0, 0.0],[0.7, 0.3], [0.5, 0.5],[0.3,0.7],[0.0,1.0]])
    stoichiometry = np.array([[1,1]], dtype=int)
    log_beta = np.log10(np.array([1.0]))
    solid_stoichiometry = np.array([[1,1]], dtype=int)
    log_ks =   np.log10(np.array([0.12]))
    #                       c1           c2           c3           s1
    real_freec = np.array([[1.000000000, 0.000000000, 0.000000000, 0.000000000],
                           [0.588819442, 0.188819442, 0.111180558, 0.000000000],
                           [0.346410162, 0.346410162, 0.120000000, 0.033589838],
                           [0.188819442, 0.588819442, 0.111180558, 0.000000000],
                           [0.000000000, 1.000000000, 0.000000000, 0.000000000]])
    real_jac = np.array([[[1.        , 0.        , 0.],
                         [0.        , 0.        , 0.],
                         [0.        , 0.        , 0.]],
                        [[0.7       , 0.11118056, 0.],
                         [0.11118056, 0.3       , 0.],
                         [0.        , 0.        , 0.]],
                        [[0.46641016, 0.12      , 1.],
                         [0.12      , 0.46641016, 1.],
                         [1.        , 1.        , 0.]],
                        [[0.3       , 0.11118056, 0.],
                         [0.11118056, 0.7       , 0.],
                         [0.        , 0.        , 0.]],
                        [[0.        , 0.        , 0.],
                         [0.        , 1.        , 0.],
                         [0.        , 0.        , 0.]]])
    gfun = np.array([[0.0, 0.0, 2.80722e-9, 0.0, 0.0]]).T
    return data(analyticalc, stoichiometry, log_beta, solid_stoichiometry, log_ks,
                real_freec, real_jac, gfun)


@pytest.fixture
def synthetic_simple1_wosld():
    analyticalc = np.array([[1.0, 0.0],[0.7, 0.3], [0.5, 0.5],[0.3,0.7],[0.0,1.0]])
    stoichiometry = np.array([[1,1]], dtype=int)
    log_beta = np.log10(np.array([1.0]))
    solid_stoichiometry = np.array([], dtype=int)
    log_ks = np.array([])
    #                       c1           c2           c3         
    real_freec = np.array([[1.000000000, 0.000000000, 0.000000000],
                           [0.588819442, 0.188819442, 0.111180558],
                           [0.366025404, 0.366025404, 0.133974596],
                           [0.188819442, 0.588819442, 0.111180558],
                           [0.000000000, 1.000000000, 0.000000000]])
    real_jac = np.array([[[1.        , 0.        ],
                         [0.        , 0.        ]],
                        [[0.7       , 0.11118056],
                         [0.11118056, 0.3       ]],
                        [[0.5       , 0.1339746 ],
                         [0.1339746 , 0.5       ]],
                        [[0.3       , 0.11118056],
                         [0.11118056, 0.7       ]],
                        [[0.        , 0.        ],
                         [0.        , 1.        ]]])
    gfun = np.array([[]]).T
    return data(analyticalc, stoichiometry, log_beta, solid_stoichiometry, log_ks,
                real_freec, real_jac, gfun)
