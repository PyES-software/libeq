from collections.abc import Sequence, Iterator

from ..data_structure import SolverData
from .optimizers import Flags


def label_refined_parameters(solver_data: SolverData) -> list[str]:
    stoich = solver_data.stoichiometry
    beta_flags = solver_data.potentiometry_opts.beta_flags

    # beta flags
    for st, flag in zip(stoich, beta_flags):
        if flag == Flags.REFINE:
            yield "logß(" + ",".join(str(_) for _ in st) + ")"

    # titration flags


class Titrations(Sequence):
    def __init__(self):
        self.__data = []

    def append(self, titration: Titration):
        self.__data.append(titration)

    def __getitem__(self, n: int):
        return self.__data[n]

    def __iter__(self) -> Iterator:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)


class Titration:
    @property
    def v_add(self) -> FArray:
        ...
