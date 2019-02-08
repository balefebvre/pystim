# PyStim

A Python 3.6 module to generate visual stimuli.


## Installation

1. Create and activate a virtual environment (Python 3.6).
2. Run `pip install --editable .`.

## Windows

To install `pycairo`:
1. Download the wheel [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo) (i.e. Windows binary).
2. Install it with: `python -m pip install <path to wheel>`.

## Usage

The module provides a console command: `pystim`. Run `pystim --help` to
get the proper command syntax (i.e. a list of the correct command-line
arguments or options acceptable to `pystim`).


## Notes

Datasets of natural images:
- [DOVES: A Database Of Visual Eye MovementS](https://live.ece.utexas.edu/research/doves/)
- van Hateren Dataset:
  - [Paul Ivanov's mirror](http://pirsquared.org/research/#van-hateren-database)
  - [Philipp Lies' mirror](http://bethgelab.org/datasets/vanhateren/)


## TODO

### ...
- [ ] Add default path.
- [ ] Handle parameters (global, local (file), local (command line)).
- [ ] Add controls (perturbed grey image).
- [ ] Generate files used for the analysis.
- [ ] Save configurations used during stimulus generations.

### 2019 02 08
- [x] Create a `euler_luminance_profile.csv` file.
    - [x] Use uint8 instead of float for luminance value.
- [x] Check that `k_max` is always inclusive in CSV files.
    - [x] Especially for `euler.csv`.
- [x] Check if repetitions in `euler.csv` are always of the same length (+10?).
