import numpy as np
import pytest

from libeq.data_structure import PotentiometryTitrationsParameters


def test_pxrange():
    ptp = PotentiometryTitrationsParameters(
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
    for upx in ptp.pX:
        assert 2.0 <= upx <= 3.0 or 7.0 <= upx <= 9.0, f"{upx=}"
