"Test collection for potentiometry data fitting."

import copy
import random
import sys
import os

import pytest
import numpy as np
import numpy.testing as npt
from libeq.data_structure import SolverData, PotentiometryOptions, PotentiometryTitrationsParameters
from libeq import PotentiometryOptimizer
from libeq.optimizers.potentiometry import Flags


class Test_ZnEDTA:
    """
    Test potentiometry optimisation using synthetic Zn/EDTA data
    """

    def setup_method(self):
        from .data import data_znedta
        self.data = data_znedta.load_data()
        # ensure weighting mode; only passes for calculated weights
        self.data.potentiometry_opts.weights = "calculated"
        self.true_beta = np.array([ 16.25, 19.25, 4.65])

    def copy_data(self):
        return copy.deepcopy(self.data)

    def test_first(self):
        sd = self.copy_data()
        result = PotentiometryOptimizer(sd, reporter=_dummy_reporter)
        calc_beta = result['final log beta'][-4:-1]
        npt.assert_allclose(calc_beta, self.true_beta, rtol=0.01)
        
    @pytest.mark.parametrize("error", [0.1, 0.2, 0.5, 1.0])
    def test_noisy_beta(self, error):
        sd = self.copy_data()
        sd.log_beta[-4:-1] += error*2*(np.random.rand(3)-0.5)
        result = PotentiometryOptimizer(sd, reporter=_dummy_reporter)
        calc_beta = result['final log beta'][-4:-1]
        npt.assert_allclose(calc_beta, self.true_beta, rtol=0.01)

    @pytest.mark.parametrize("error", [0.0002, 0.0005, 0.001, 0.002])
    def test_noisy_c0(self, error):
        sd = self.copy_data()
        true_c0 = []
        for tit in sd.potentiometry_opts.titrations:
            true_c0.append(tit.c0.copy())
            tit.c0_flags = [Flags.CONSTANT, Flags.CONSTANT, Flags.REFINE]
            tit.c0[-1] += random.uniform(-error/2, error/2) 
        result = PotentiometryOptimizer(sd, reporter=_dummy_reporter)
        calc_beta = result['final log beta'][-4:-1]
        npt.assert_allclose(calc_beta, self.true_beta, rtol=0.01)
        for (calc_c0, _), _true_c0 in zip(result['final titration parameters'], true_c0):
            npt.assert_allclose(calc_c0, _true_c0, atol=1e-5)


def _list_json(path):
    return [os.path.join(path, f) 
            for f in os.listdir(path)
            if f.endswith('json')]


@pytest.mark.parametrize("filename", _list_json('./tests/data/test_files/Glycine_test/'))
def test_gly(filename):
    data = SolverData.load_from_pyes(filename)
    result = PotentiometryOptimizer(data, reporter=_dummy_reporter)
    if 'Hfix' in filename:
        true_beta = np.array([ 9.58, 12.00 ])
        calc_beta = result['final variables']
        npt.assert_allclose(calc_beta, true_beta, atol=1e-2)
    else:
        true_beta = np.array([ 9.49, 11.83 ])
        true_c0 = [ 0.01483, 0.02989, 0.02103, 0.00876]
        calc_beta = result['final variables'][:2]
        npt.assert_allclose(calc_beta, true_beta, atol=1e-2)
        calc_c0 = result['final variables'][2:]
        npt.assert_allclose(calc_c0, true_c0, atol=1e-2)
    

def _dummy_reporter(**kwargs):
    pass
