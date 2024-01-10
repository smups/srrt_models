# Slow-roll, (rapid) turn models in two-field inflation
This repository contains the models tested in (ArXiV:2402.????).

To run the code, the install the latest version of [`inflatox`](https://pypi.org/project/inflatox/) (for example using `pip install inflatox`). Basic dependencies, like `matplotlib`, `numpy` and `sympy` are also required to run the code.

## Overview
The code presented here is organised into three python scripts corresponding to the three tested models. These contain all numerically intensive code. The output of these scripts is saved to the `out` folder in the root directory. The jupyter notebook contains all the python plotting code used to make the figures in (paper).

The trajectories in the trajectories folder have been obtained using `pytransport` (angular and EGNO models) and `mtransport` (d5 model). This code is not provided in this repository.

## Minimum `inflatox` version
The minimum inflatox version required to run the code here is `v0.7.0`