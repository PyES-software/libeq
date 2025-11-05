"Test collection for potentiometry data fitting."

import numpy as np
from libeq.data_structure import SolverData

def test_first():
    sd = SolverData(
        components = ['EDTA', 'Zn', 'H'],
        stoichiometry = np.array([[1,0, 1],
                                  [1,0, 2],
                                  [1,0, 3],
                                  [1,0, 4],
                                  [1,0, 5],
                                  [0,1,-1],
                                  [0,1,-2],
                                  [0,1,-3],
                                  [0,1,-4],
                                  [0,2,-1],
                                  [0,2,-6],
                                  [1,1, 0],
                                  [1,1, 1],
                                  [1,1,-1],
                                  [0,0,-1]], dtype=int),
        solid_stoichiometry = np.array([[]], dtype=int)
    )
