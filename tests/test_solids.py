from libeq import SolverData, EqSolver

def test_solid_aluminium():
    data = SolverData.load_from_pyes('tests/data/aluminium_alkaline.json')

    result, log_beta, log_ks, saturation_index, total_concentration = EqSolver(data, mode="distribution")
    breakpoint()
    print(result)
