# Quantum Parton Shower

Particles produced in high energy collisions that are charged under one of the fundamental forces will radiate proportionally to their charge, such as photon radiation from electrons in quantum electrodynamics. At sufficiently high energies, this radiation pattern is enhanced collinear to the initiating particle, resulting in a complex, many-body quantum system. Classical Markov Chain Monte Carlo simulation approaches work well to capture many of the salient features of the shower of radiation, but cannot capture all quantum effects. We show how quantum algorithms are well-suited for describing the quantum properties of final state radiation. In particular, we develop a polynomial time quantum final state shower that accurately models the effects of intermediate spin states similar to those present in high energy electroweak showers. The algorithm is explicitly demonstrated for a simplified quantum field theory on a quantum computer.

Details can be found in [1904.03196 [quant-ph]](https://arxiv.org/abs/1904.03196).


# File Directories:
The files are organized into several different directories:
* cirq_code
    * contains all files that import `cirq`
* qiskit_code
    * contains all files that import `qiskit` and not `cirq`
* plots
    * contains all plots used in `arxiv:2203.10018`
* data
    * contains qiskit simulation and classical MCMC data
* qiskit_legacy
    * contains QuantumPartonShower.py, before Plato made some adjustments
* qiskit_legacy/PaperPlots
    * contains Ben's original code used for `arxiv:1904.03196`


## Table of Contents:
* `plotting.py`
    * Code for generating plots of the remeasurement paper
* `classical.py`
    * classical MCMC simulations for computing number of emissions, and function for computing θmax analytically

### cirq_code/
* `TestHelpers.py`
    * helper functions for testing cirq circuits
* `SubcircuitTests.py`
    * unittest file for testing cirq circuit elements (Ucount, Ue, Uh, Up, etc.)
* `circuit_drawing.py`
* `create_circuit.py`
    * cirq QPS circuits
* `testing_cirq_code.ipynb`
* `PartonShower_1.ipynb`
* `PartonShower_2.ipynb`

### qiskit_legacy/
* `QuantumPartonShower_legacy.py`
    * the version of `QuantumPartonShower.py` before Plato made adjustments
* `qiskit_create_circ.py`
    * contains the contents of `QuantumPartonShower_legacy.py`, but not wrapped nicely in a class

### qiskit_legacy/PaperPlots/
Contains Ben's original code used for `arxiv:1904.03196`.
* `MakeObservables.py`
* `Fig1.py`
* `2stepSim.py`
* `AppendixPlot.py`

Data files that can be generated from `Fig1.py`:
* `vals_classical.npy` &larr; `Fig1.py`
* `hold_*.txt` &larr; `2stepSim.py`
* `fullsim2step_states.pdf` &larr; `Fig1.py`
* `quantum_0_*.npy` &larr; `Fig1.py`
* `quantum_100_*.npy` &larr; `Fig1.py`
* `test_alpha*.pdf` &larr; `Fig1.py`

###  qiskit_code/
* `QuantumPartonShower_ReM.py`
    * General N-step QPS circuit construction and simulation
* `QuantumPartonShower_ReM_2step_hardcode.py`
    * Hardcoded 2-step QPS circuit construction and simulation
* `QuantumPartonShower_ReM_forGateCounting.py`
    * Classically-controlled operations reduced to once per true operation, as opposed to the hack used for Qiskit. This file should only be used to count gates.

* `QPS_Paper_Plots.ipynb`  
    * Main notebook for generating the paper plots
* `running_hard_coded.ipynb`
    * How to run the hard-coded 2-step QPS simulation in `QuantumPartonShower_ReM_2step_hardcode.py`

### plots/
* `fig4_qubit_count.pdf`
* `fig5_gate_count.pdf`
* `simNstep_emissions_shots=1e+05_paperStyle.pdf`
    * Probability distribution (with errors) of number of emissions for an N-step QPS simulation
* `simNstep_thetamax_shots=1e+05_paperStyle.pdf`
    * Probability distribution (with errors) of θmax for an N-step QPS simulation

### data/
* `counts_Nstep_*.npy`           &larr;  QPS_Paper_Plots.ipynb
    * QPS with mid-circuit meas. simulation counts
* `counts_OLD_Nstep_*.npy`       &larr;  QPS_Paper_Plots.ipynb
    * original QPS simulation counts
* `thetamax_analytical_N=*.npz`  &larr;  QPS_Paper_Plots.ipynb
    * analytic θmax curve                               
    * arr0 is the θmax array, arr1 is the solid angle dσ / dθmax (proportional to probability)
* `mcmc_Nstep*.npy`              &larr;  QPS_Paper_Plots.ipynb           
    * MCMC counts
* `cx_hack.npy`                  &larr;  QPS_Paper_Plots.ipynb
    * CNOT counts for hacked QPS with mid-circuit meas.
* `cx_naive.npy`                 &larr;  QPS_Paper_Plots.ipynb
    * CNOT counts for original QPS