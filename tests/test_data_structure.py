import numpy as np
import pytest

from libeq.data_structure import PotentiometryTitrationsParameters, SolverData

@pytest.fixture
def znedta():
    return SolverData.load_from_pyes("tests/data/znedta.json")


@pytest.fixture
def ptp1():
    return PotentiometryTitrationsParameters(
        electro_active_compoment=0,
        e0=400.0,
        e0_sigma=0.1,
        slope=59.0,
        v0=25.0,
        v0_sigma=0.1,
        v_add=np.linspace(0.0, 1.0, 56),
        emf=np.linspace(341.0, -308, 56),
        px_range=[[2.0, 3.0], [7.0, 9.0]]
    )


def test_pxrange1(ptp1):
    for upx in ptp1.pX:
        assert 2.0 <= upx <= 3.0 or 7.0 <= upx <= 9.0


def test_pxrange2(znedta):
    tit0 = znedta.potentiometry_opts.titrations[0]
    tit0.px_range=[[2.0, 3.0], [8.0, 11.0]]
    for upx in tit0.pX:
        assert 2.0 <= upx <= 3.0 or 8.0 <= upx <= 11.0


def test_pxcombined(ptp1):
    igneven = np.array(28*[True, False], dtype=bool)
    ptp1.ignored = igneven
    for upx in ptp1.pX:
        assert 2.0 <= upx <= 3.0 or 7.0 <= upx <= 9.0
