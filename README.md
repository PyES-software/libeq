<p align="center">
<img src="docs/assets/logo.png" style="width: 5vw;">
</p>

# libeq

`libeq` is a Python library for the solution of thermodynamic equilibrium. It is the core routine of [PyES](https://www.github.com/Kastakin/PyES), a frontend for the calculation of species distribution and simulation of titration curves.

## Installation

To install `libeq`, you can use pip:

```sh
pip install libeq
```

## Documentation

For more detailed information about `libeq` and its usage, please refer to the project documentation.

## Available tools in the API

### Data handling

The module provided **SolverData** which is a pydantic dataclass where all the data is
loaded. It can import and export data to and from other formats. This is the data interface
for all the other tools.

### Equilibrium solving tools

* EqSolver: solves equilibria in solution and returns the equilibrium concentrations of
 all the species involved.

* PotentiometryOptimizer: optimizes equilibrium constants and other parameters based on
 potentiometry data.

### Utilities

* species_concentration: completes the rest of the species array

## Acknowledgements

This library is based on the work of many research groups on the topic, in particular the works of Prof. Sammartano's research group from the University of Messina and the Prof. Carrayrou from the University of Strasbourg.

The code has been heavily inspired by the works of [Prof. Blasco](https://github.com/salvadorblasco)
from the [University of Valencia](https://www.uv.es/).
