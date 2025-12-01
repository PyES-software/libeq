"Test collection for potentiometry data fitting."

import copy
import sys
print(sys.path)

import numpy as np
import numpy.testing as npt
from libeq.data_structure import SolverData, PotentiometryOptions, PotentiometryTitrationsParameters
from libeq import PotentiometryOptimizer
from libeq.optimizers.potentiometry import Flags


class Test_ZnEDTA:
    def __init__(self):
        from .data import data_znedta
        self.data = data_znedta.load_data()

    def test_first():
        sd = copy.deepcopy(self.data)
        result = PotentiometryOptimizer(sd, reporter=self.__dummy_reporter)
        true_beta = np.array([ 10.19, 16.32, 19.01, 21.01, 22.51, -9.15, -17.1, -28.39, -40.71, -8.89, -57.53, 16.25, 19.25, 4.65, -13.78])
        npt.assert_allclose(result['final_beta'], true_beta, rtol=0.02)
        
    def __dummy_reporter(**kwargs):
        pass


