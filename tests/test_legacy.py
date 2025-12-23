"Test collection for legacy file import"

from libeq.data_structure import SolverData

def test_import_superquad():
    parsed_data = SolverData.load_from_superquad("tests/data/hedtac.sup")
