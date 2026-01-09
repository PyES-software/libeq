from .utils import species_concentration
from .errors import uncertanties
from .data_structure import SolverData
from .optimizers import PotentiometryOptimizer, Flags
from .solver import EqSolver

"""
libeq - A Python library for thermodynamic equilibrium calculations.

The library exposes functions both to calculate the equilibrium of a chemical system and to
optimize parameters, such as formation constants and experimental parameters, from experimental data:
- EqSolver: function to solve the equilibrium of a chemical system.
- PotentiometryOptimizer:  function to optimize parameters of a chemical system using potentiometry data.

Data is paassed to the library using the SolverData class, which is a Pydantic model that act as a container for

A tiny function to compute the concentration of species from the concentrations of
free components is also exposed: `species_concentration`.
"""

# Define __all__ to control what gets imported when using `from libeq import *`
__all__ = [
    "EqSolver",
    "PotentiometryOptimizer",
    "SolverData",
    "species_concentration",
    "uncertanties",
    "Flags"
]
