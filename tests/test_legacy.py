"Test collection for legacy file import"

import numpy as np
import numpy.testing as npt

from libeq import SolverData, Flags
from libeq.data_structure import PotentiometryTitrationsParameters


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

    titration: PotentiometryTitrationsParameters = parsed_data.potentiometry_opts.titrations[0]
    print(titration.electro_active_compoment)
    assert parsed_data.components[titration.electro_active_compoment] == 'PROTON'
    assert titration.e0 == 364.50
    assert titration.e0_sigma == 0.3
    assert titration.v0 == 30.407
    assert titration.v0_sigma == 0.003

    true_titre = np.arange(0.0, 5.14, 0.02) 
    npt.assert_allclose(titration.v_add, true_titre)

    true_emf = np.array([ 242.09, 241.94, 241.76, 241.57, 241.39, 241.20, 241.01, 240.83, 240.64, 240.44, 240.26, 240.07, 239.89, 239.70, 239.49, 239.29, 239.11, 238.91, 238.71, 238.51, 238.30, 238.12, 237.92, 237.70, 237.49, 237.28, 237.08, 236.89, 236.68, 236.44, 236.24, 236.01, 235.80, 235.60, 235.35, 235.14, 234.92, 234.69, 234.47, 234.24, 234.00, 233.79, 233.55, 233.32, 233.08, 232.84, 232.60, 232.37, 232.14, 231.89, 231.62, 231.38, 231.15, 230.92, 230.64, 230.38, 230.13, 229.87, 229.60, 229.34, 229.08, 228.82, 228.55, 228.29, 228.00, 227.74, 227.46, 227.21, 226.92, 226.61, 226.32, 226.07, 225.78, 225.48, 225.20, 224.93, 224.64, 224.34, 224.00, 223.70, 223.38, 223.07, 222.76, 222.44, 222.12, 221.77, 221.47, 221.15, 220.81, 220.50, 220.16, 219.84, 219.47, 219.08, 218.72, 218.36, 218.00, 217.60, 217.23, 216.89, 216.50, 216.11, 215.71, 215.28, 214.89, 214.51, 214.13, 213.74, 213.35, 212.95, 212.54, 212.12, 211.72, 211.29, 210.88, 210.44, 210.00, 209.55, 209.11, 208.63, 208.14, 207.65, 207.17, 206.67, 206.15, 205.64, 205.13, 204.60, 204.06, 203.53, 202.97, 202.42, 201.85, 201.24, 200.63, 200.00, 199.36, 198.72, 198.07, 197.39, 196.69, 196.01, 195.31, 194.60, 193.86, 193.76, 193.68, 193.57, 193.42, 193.21, 192.99, 192.73, 192.17, 191.48, 190.72, 189.90, 189.03, 188.17, 187.24, 186.32, 185.37, 184.38, 183.36, 182.31, 181.21, 180.12, 178.95, 177.75, 177.28, 176.99, 176.54, 176.03, 175.08, 173.89, 172.82, 171.47, 170.64, 169.61, 168.23, 166.87, 165.19, 163.60, 161.77, 159.94, 157.75, 155.31, 152.54, 149.42, 145.86, 141.82, 137.07, 131.40, 124.52, 115.77, 104.56, 90.71, 76.39, 63.94, 53.65, 44.84, 37.30, 30.48, 24.39, 18.75, 13.55, 8.52, 3.76, -1.02, -6.00, -11.03, -16.19, -21.59, -27.71, -34.48, -42.20, -51.52, -63.85, -82.53, -114.12, -148.92, -172.18, -184.00, -192.23, -198.69, -204.31, -209.36, -213.78, -217.87, -221.54, -225.10, -228.41, -231.42, -234.13, -236.66, -239.31, -241.59, -243.94, -246.01, -248.11, -250.05, -251.89, -253.86, -255.59, -257.14, -258.68, -260.21, -261.68, -263.07, -264.50, -265.84, -267.19, -268.47, -269.65, -270.79, -271.92, -273.02, -274.06])
    npt.assert_allclose(true_emf, titration.emf)

    true_c0 = np.array([0.03558/30.407, 0.39203/30.407])
    npt.assert_allclose(true_c0, titration.c0)

    true_b = np.array([0.0, -0.081])
    npt.assert_allclose(true_b, titration.ct)


def test_import_bstac():
    data = SolverData.load_from_bstac("tests/data/namototc.mis")

