# Quantum Parton Shower

Particles produced in high energy collisions that are charged under one of the fundamental forces will radiate proportionally to their charge, such as photon radiation from electrons in quantum electrodynamics. At sufficiently high energies, this radiation pattern is enhanced collinear to the initiating particle, resulting in a complex, many-body quantum system. Classical Markov Chain Monte Carlo simulation approaches work well to capture many of the salient features of the shower of radiation, but cannot capture all quantum effects. We show how quantum algorithms are well-suited for describing the quantum properties of final state radiation. In particular, we develop a polynomial time quantum final state shower that accurately models the effects of intermediate spin states similar to those present in high energy electroweak showers. The algorithm is explicitly demonstrated for a simplified quantum field theory on a quantum computer.

Details can be found in [1904.03196 [quant-ph]](https://arxiv.org/abs/1904.03196).


File Directory:

    Old files:

        PartonShower (1).ipynb
        PartonShower_2.ipynb
        SubcircuitTests.py
        TestHelpers.py
        circuit_drawing.py
        create_circuit.py
        qiskit_create_circ.py
        testing_cirq_code.ipynb

    Updated files:

        PaperPlots/2stepSim.py --> PaperPlots/two_stepSim.py
        PaperPlots/MakeObservables.py
        QuantumPartonShower.py
        README.md

    New files:

        PaperPlots/
            Fig1_duplicate.py
            vals_classical.npy <- Fig1.py

        Plots/*
        plotting.py
        
        classical.py                                     = MCMC for number of emissions, analytical θmax
        QuantumPartonShower_ReM.py                       = Final general N-step simulation
        QuantumPartonShower_ReM_forGateCounting.py       = an intermediate version
        QuantumPartonShower_ReM_2step_hardcode.py        = Final hardcoded 2-step simulation
        QuantumPartonShower_ReM_2step_hardcode_old.py    = an intermediate version

        Data:
            counts_Nstep_*.npy           <-- QPS_Paper_Plots.ipynb    =  QPS with mid-circuit meas. simulation counts
            counts_OLD_Nstep_*.npy       <-- QPS_Paper_Plots.ipynb    =  original QPS simulation counts
            thetamax_analytical_N=*.npz  <-- QPS_Paper_Plots.ipynb    =  analytic θmax curve
            mcmc_Nstep*.npy              <-- QPS_Paper_Plots.ipynb    =  MCMC counts
            cx_hack.npy                  <-- QPS_Paper_Plots.ipynb    =  CNOT counts for hacked QPS with mid-circuit meas.
            cx_naive.npy                 <-- QPS_Paper_Plots.ipynb    =  CNOT counts for original QPS
        
        Notebooks:
            QPS_Paper_Plots.ipynb                    = Main notebook for generating the paper plots
    
            Noteboooooooooook_manual.ipynb           = investigating strange qiskit simulator behaviors
	        Notebooooooooooook.ipynb                 = old notebook, nothing much here
            gate_counting.ipynb                      = Messy workbook with just about everything in it
            hard_coded.ipynb                         = old notebook, used to test simplifications to QPS (more resets and fewer qubits)
