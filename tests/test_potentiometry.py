"Test collection for potentiometry data fitting."

import numpy as np
from libeq.data_structure import SolverData

def test_first():
    sd = SolverData(
        components = ['EDTA', 'Zn', 'H']
    )
