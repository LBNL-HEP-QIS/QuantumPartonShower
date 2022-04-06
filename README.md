# Quantum Parton Shower

Particles produced in high energy collisions that are charged under one of the fundamental forces will radiate proportionally to their charge, such as photon radiation from electrons in quantum electrodynamics. At sufficiently high energies, this radiation pattern is enhanced collinear to the initiating particle, resulting in a complex, many-body quantum system. Classical Markov Chain Monte Carlo simulation approaches work well to capture many of the salient features of the shower of radiation, but cannot capture all quantum effects. We show how quantum algorithms are well-suited for describing the quantum properties of final state radiation. In particular, we develop a polynomial time quantum final state shower that accurately models the effects of intermediate spin states similar to those present in high energy electroweak showers. The algorithm is explicitly demonstrated for a simplified quantum field theory on a quantum computer.

Details can be found in [1904.03196 [quant-ph]](https://arxiv.org/abs/1904.03196).


# File Directory:

## Old files:
* `PartonShower (1).ipynb`
* `PartonShower_2.ipynb`
* `SubcircuitTests.py`
* `TestHelpers.py`
* `circuit_drawing.py`
* `create_circuit.py`
* `qiskit_create_circ.py`
* `testing_cirq_code.ipynb`

## Updated files:
* `PaperPlots/2stepSim.py` &rarr; `PaperPlots/two_stepSim.py`
* `PaperPlots/MakeObservables.py`
* `QuantumPartonShower.py`
* `README.md`

## New files:

### Python:
* `plotting.py`
    * Code for generating plots of the remeasurement paper
* `classical.py`
    * classical MCMC simulations for computing number of emissions, and function for computing θmax analytically
* `QuantumPartonShower_ReM.py`
    * General N-step QPS circuit construction and simulation
* `QuantumPartonShower_ReM_forGateCounting.py`
    * Classically-controlled operations reduced to the actual number, as opposed to the hack used for Qiskit. This file should only be used to count gates.
* `QuantumPartonShower_ReM_2step_hardcode.py`
    * Hardcoded 2-step QPS circuit construction and simulation

### Notebooks:
* `QPS_Paper_Plots.ipynb`  
    * Main notebook for generating the paper plots
* `running_hard_coded.ipynb`
    * How to run the hard-coded 2-step QPS simulation in `QuantumPartonShower_ReM_2step_hardcode.py`

### PaperPlots/
* `Fig1_duplicate.py`
* `vals_classical.npy` &larr; `Fig1.py`

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