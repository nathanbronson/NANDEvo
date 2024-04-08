<p align="center"><img src="https://github.com/nathanbronson/NANDEvo/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# NANDEvo
a binary function approximator

## About
NANDEvo is an evolutionary algorithm meant to approximate an arbitrary black box binary function from inputs and associated outputs by composing NAND gates.

This codebase includes the code for a `Scheme` (a complete string protocol for binary functions including utilities to convert from string instructions to Python functions), an evolutionary algorithm to `evolve` these functions, a few simple `Datasets`, and utilities to evaluate a function's `fitness`.

In tests, NANDEvo approximates binary functions of two variables with some success. Further exploration of parallelization and mutation protocols could provide insight to improve NANDEvo's performance.

Maintainence of this codebase will occur only sporadically.

## Usage
To replicate the addition experiment, run:
```
$ python3 tests.py
```

All elements of the code are written to facilitate easy import in other code. They should be imported from their source files.

## License
See `LICENSE`.
