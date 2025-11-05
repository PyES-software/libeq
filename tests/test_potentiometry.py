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
        solid_stoichiometry = np.array([[]], dtype=int),
        log_beta = np.array([ 10.19, 
                              16.32, 
                              19.01, 
                              21.01, 
                              22.51, 
                              -9.15, 
                             -17.1, 
                             -28.39, 
                             -40.71, 
                              -8.89, 
                             -57.53, 
                              16.25, 
                              19.25, 
                               4.65, 
                              -13.78])
    )
