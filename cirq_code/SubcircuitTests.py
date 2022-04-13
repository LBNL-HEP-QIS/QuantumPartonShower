import unittest
import cirq
import main
import math
import numpy as np
import TestHelpers


class InitializeQubitTest(unittest.TestCase):

    def test_oneQubit(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        initialParticles = [[0, 0, 1]]
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        simulator = cirq.Simulator()
        circuit.append(cirq.measure(*pReg[0], key='x'))
        result = simulator.simulate(circuit)
        firstParticle = result.measurements['x']
        np.testing.assert_array_equal(firstParticle,np.array([0,0,1]))

    def test_threeQubits(self):
        n_i, N = 2, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        initialParticles = [[0, 0, 1], [0, 1, 0]]
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        simulator = cirq.Simulator()
        circuit.append(cirq.measure(*TestHelpers.flatten(pReg), key='x'))
        result = simulator.simulate(circuit)
        particles = result.measurements['x']
        np.testing.assert_array_equal(particles,np.array([0,0,1,0,1,0,0,0,0]))


class RotateTest(unittest.TestCase):

    def test_rotateOneBoson(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        u = .5
        initialParticles = [[1, 0, 0]]
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        for p_k in pReg:
            circuit.append(cirq.ry(2*math.asin(-u)).controlled().on(p_k[2], p_k[0]))
        simulator = cirq.Simulator()
        circuit.append(cirq.measure(*pReg[0], key='x'))
        result = simulator.simulate(circuit)
        firstParticle = result.measurements['x']
        np.testing.assert_array_equal(firstParticle,np.array([1, 0, 0]))

    def test_rotateBosonAndFermionA(self):
        n_i, N = 2, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        u = .5
        initialParticles = [[1, 0, 0], [0, 0, 1]] #boson and f_a fermion described in [LSB, .. , MSB] format
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        for p_k in pReg:
            circuit.append(cirq.ry(2*math.asin(-u)).controlled().on(p_k[2], p_k[0]))
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        waveVectorString = str(result)
        vectorString = waveVectorString.split('output vector: ', 1)[1] #outputs wavevector in format pReg[1], pReg[0] in format [MSB..LSB]]
        self.assertEqual(vectorString, "0.866|100100⟩ - 0.5|101100⟩")

    def test_rotateBosonAndFermionB(self):
        n_i, N = 2, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        u = .5
        initialParticles = [[1, 0, 0], [1, 0, 1]]  # boson and f_a fermion described in [LSB, .. , MSB] format
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        for p_k in pReg:
            circuit.append(cirq.ry(2*math.asin(-u)).controlled().on(p_k[2], p_k[0]))
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        waveVectorString = str(result) # outputs wavevector in order pReg[1], pReg[0] in format [MSB..LSB]
        vectorString = waveVectorString.split('output vector: ', 1)[1]
        self.assertEqual(vectorString, "0.5|100100⟩ + 0.866|101100⟩")


class UCountTest(unittest.TestCase):

    def test_countTwoBosonsOneBFermion(self):
        n_i, N = 3, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        initialParticles = [[1, 0, 0], [1, 0, 1], [1, 0, 0]] #boson, f_b fermion, boson
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        # Currently altered to use only w_phiReg to save qubits (done for rest of tests as well)
        main.uCount(circuit, 0, n_i, L, pReg, wReg, n_aReg, w_phiReg, n_bReg, w_phiReg, n_phiReg, w_phiReg)
        # measure in order n_phi, n_a, n_b in format [LSB, MSB]
        circuit.append(cirq.measure(*TestHelpers.flatten([n_phiReg, n_aReg, n_bReg]), key='x'))
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        particles = result.measurements['x']
        np.testing.assert_array_equal(particles,np.array([0, 1, 0, 0, 0, 0, 1, 0, 0]))

    # WARNING: THIS TEST PASSED BUT TOOK ME 306 SECONDS TO RUN ON MY PC
    def test_countTwoBosonsOneAFermionOneAAntifermion(self):
        n_i, N = 4, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        initialParticles = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]] #2 bosons, f_a fermion, f_a antifermion
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        main.uCount(circuit, 0, n_i, L, pReg, wReg, n_aReg, w_phiReg, n_bReg, w_phiReg, n_phiReg, w_phiReg)
        # measure in order n_phi, n_a, n_b in format [LSB, MSB]
        circuit.append(cirq.measure(*TestHelpers.flatten([n_phiReg, n_aReg, n_bReg]), key='x'))
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        particles = result.measurements['x']
        np.testing.assert_array_equal(particles,np.array([0, 1, 0, 0, 1, 0, 0, 0, 0]))


class GenerateParticleCountsTest(unittest.TestCase):

    def test_genCounts(self):
        target1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        gen1 = np.array(main.generateParticleCounts(1,0,0))
        np.testing.assert_array_equal(target1, gen1)
        target2 = np.array([[0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]])
        gen2 = np.array(main.generateParticleCounts(2,0,0))
        np.testing.assert_array_equal(target2, gen2)
        target3 = np.array([[0,0,1], [0,1,0], [1,0,0], [0,0,2], [0,1,1], [0,2,0], [1,0,1], [1,1,0], [2,0,0]])
        gen3 = np.array(main.generateParticleCounts(1,1,0))
        np.testing.assert_array_equal(target3, gen3)
        target4 = np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0]])
        gen4 = np.array(main.generateParticleCounts(1,1,1))
        np.testing.assert_array_equal(target4, gen4)
        target5 = np.array([[0,0,0]])
        gen5 = np.array(main.generateParticleCounts(1,1,2))
        np.testing.assert_array_equal(target5, gen5)


class UETest(unittest.TestCase):

    def test_OneBoson(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        Delta_phi, Delta_a, Delta_b,  = .25, .4, .7
        initialParticles = [[1, 0, 0]] # 1 boson
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        main.uCount(circuit, 0, n_i, L, pReg, wReg, n_aReg, w_phiReg, n_bReg, w_phiReg, n_phiReg, w_phiReg)
        main.uE(circuit, L, n_i, 0, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, Delta_phi, Delta_a, Delta_b)
        simulator = cirq.Simulator()
        qubitOrder = pReg[0][::-1] + eReg + n_phiReg[::-1] + n_aReg[::-1] + n_bReg[::-1] + w_phiReg + w_aReg + w_bReg + wReg
        result = simulator.simulate(circuit, qubit_order = qubitOrder)
        waveVectorString = str(result) # outputs wavevector in order pReg[1], pReg[0] in format [MSB..LSB]]
        vectorString = waveVectorString.split('output vector: ', 1)[1]
        self.assertEqual("0.5|001001000000000000⟩ + 0.866|001101000000000000⟩", vectorString)

    # THIS TOOK 200 s on my PC
    def test_OneBosonOneAFermion(self):
        n_i, N = 2, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        Delta_phi, Delta_a, Delta_b,  = .25, .4, .7
        initialParticles = [[1, 0, 0], [0, 0, 1]] # boson, f_a fermion
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        main.uCount(circuit, 0, n_i, L, pReg, wReg, n_aReg, w_phiReg, n_bReg, w_phiReg, n_phiReg, w_phiReg)
        main.uE(circuit, L, n_i, 0, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, Delta_phi, Delta_a, Delta_b)
        simulator = cirq.Simulator()
        qubitOrder = pReg[0][::-1] + pReg[1][::-1] + eReg + n_phiReg[::-1] + n_aReg[::-1] + n_bReg[::-1] + w_phiReg + w_aReg + w_bReg + wReg
        result = simulator.simulate(circuit, qubit_order = qubitOrder)
        waveVectorString = str(result) # outputs wavevector in order pReg[1], pReg[0] in format [MSB..LSB]]
        vectorString = waveVectorString.split('output vector: ', 1)[1]
        self.assertEqual("0.316|001100001010000000000⟩ + 0.949|001100101010000000000⟩", vectorString)


class TwoLevelControlledRyTest(unittest.TestCase):
    def test_oneQubit(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        control = eReg[0]
        circuit.append(cirq.X(control), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        main.twoLevelControlledRy(circuit, 2, np.pi, 1, control, hReg[0], w_hReg)

        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg[0])), key='h'))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        np.testing.assert_array_equal(np.array([0,1]), result.measurements['h'])

    def test_controlOn180Degrees(self):
        n_i, N = 1, 6
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        control = eReg[0]
        circuit.append(cirq.X(control), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        for i in range(6):
            main.twoLevelControlledRy(circuit, L, np.pi, i, control, hReg[i], w_hReg)

        for i in range(6):
            circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg[i])), key=f"h{i}"))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        np.testing.assert_array_equal(np.array([0, 0, 0]), result.measurements['h0'])
        np.testing.assert_array_equal(np.array([0, 0, 1]), result.measurements['h1'])
        np.testing.assert_array_equal(np.array([0, 1, 0]), result.measurements['h2'])
        np.testing.assert_array_equal(np.array([0, 1, 1]), result.measurements['h3'])
        np.testing.assert_array_equal(np.array([1, 0, 0]), result.measurements['h4'])
        np.testing.assert_array_equal(np.array([1, 0, 1]), result.measurements['h5'])

    def test_controlOn90Degrees(self):
        n_i, N = 2, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        control = eReg[0]
        circuit.append(cirq.X(control), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        main.twoLevelControlledRy(circuit, L, np.pi / 2, 0, control, hReg[0], w_hReg)
        main.twoLevelControlledRy(circuit, L, np.pi / 2, 1, control, hReg[0], w_hReg)

        simulator = cirq.Simulator()
        qubitOrder = hReg[0][::-1] + eReg + wReg + w_hReg
        result = simulator.simulate(circuit, qubit_order=qubitOrder)
        waveVectorString = str(result)  # outputs wave-vector in order qubitOrder
        vectorString = waveVectorString.split('output vector: ', 1)[1]
        self.assertEqual("0.707|001000000⟩ + 0.707|011000000⟩", vectorString)

    def test_controlOff(self):
        n_i, N = 1, 3
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        control = eReg[0]

        main.twoLevelControlledRy(circuit, L, np.pi, 0, control, hReg[0], w_hReg)
        main.twoLevelControlledRy(circuit, L, np.pi, 1, control, hReg[1], w_hReg)

        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg[0])), key='h0'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg[1])), key='h1'))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        np.testing.assert_array_equal(np.array([0, 0, 0]), result.measurements['h0'])
        np.testing.assert_array_equal(np.array([0, 0, 0]), result.measurements['h1'])


class UHTest(unittest.TestCase):

    def test_FermionEmissionHistory(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        # we want the f_a fermion to emit, so make P_b and P_phi very small but avoid div0 errors
        P_a, P_phi, P_b = 1, .000001, .000001
        initialParticles = [[0, 0, 1]]  # 1 f_a fermion
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        main.uCount(circuit, 0, n_i, 2, pReg, wReg, n_aReg, w_phiReg, n_bReg, w_phiReg, n_phiReg, w_phiReg)
        circuit.append(cirq.X(eReg[0]), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)  # turn emission bit on
        main.U_h(circuit, 2, n_i, 0, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, pReg, hReg, w_hReg,
                 P_phi, P_a, P_b)

        circuit.append(cirq.measure(*eReg, key='e'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg)), key='h'))
        circuit.append(cirq.measure(*TestHelpers.flatten([n_phiReg, n_aReg, n_bReg]), key='n'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten([pReg])), key='p'))
        circuit.append(cirq.measure(*TestHelpers.flatten([wReg, w_phiReg, w_aReg, w_bReg]), key='w'))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        np.testing.assert_array_equal(np.array([0]), result.measurements['e'])  # emission bit should be turned off
        # history reg should be 01 (only particle 1, the f_a fermion should have emitted)
        np.testing.assert_array_equal(np.array([0, 1]), result.measurements['h'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0]), result.measurements['n'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 1, 0, 0]), result.measurements['p'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0, 0, 0]), result.measurements['w'])

    # Takes around 12 minutes on my PC!
    def test_TwoQubitBosonEmissionHistory(self):
        n_i, N = 2, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)
        # we want the f_a fermion and the boson to emit, so make P_b and P_phi very small but avoid div0 errors
        P_phi = 1
        P_a = P_b = .0001
        initialParticles = [[0, 0, 1], [1, 0, 0]]  # 1 f_a fermion, 1 boson
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)
        main.uCount(circuit, 0, n_i, L, pReg, wReg, n_aReg, w_phiReg, n_bReg, w_phiReg, n_phiReg, w_phiReg)
        circuit.append(cirq.X(eReg[0]), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)  # turn emission bit on
        main.U_h(circuit, 2, n_i, 0, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, pReg, hReg, w_hReg,
                 P_phi, P_a, P_b)

        circuit.append(cirq.measure(*eReg, key='e'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg)), key='h'))
        circuit.append(cirq.measure(*TestHelpers.flatten([n_phiReg, n_aReg, n_bReg]), key='n'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten([pReg])), key='p'))
        circuit.append(cirq.measure(*TestHelpers.flatten([wReg, w_phiReg, w_aReg, w_bReg]), key='w'))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        np.testing.assert_array_equal(np.array([0]), result.measurements['e'])  # emission bit should be turned off
        # history reg should be 10 (only particle 2 (0 indexed), the boson should have emitted)
        np.testing.assert_array_equal(np.array([1, 0]), result.measurements['h'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0]), result.measurements['n'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 1, 0, 0]), result.measurements['p'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0, 0, 0]), result.measurements['w'])


class UPTest(unittest.TestCase):

    def test_FermionEmission(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)

        initialParticles = [[0, 0, 1]]  # 1 f_a fermion
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)

        circuit.append(cirq.X(hReg[0][0]), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)  # first particle emits

        main.U_p(circuit, 2, n_i, 0, pReg, hReg, w_hReg, wReg, .5, .5)

        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg)), key='h'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten([pReg])), key='p'))
        circuit.append(cirq.measure(*TestHelpers.flatten([wReg, w_hReg]), key='w'))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # history reg should be 01 (only particle 1, the f_a fermion should have emitted)
        np.testing.assert_array_equal(np.array([0, 1]), result.measurements['h'])
        np.testing.assert_array_equal(np.array([0, 0, 1, 1, 0, 0]), result.measurements['p'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0]), result.measurements['w'])

    def test_AntiFermionEmission(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)

        initialParticles = [[1, 1, 1]]  # 1 f_b anti-fermion
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)

        circuit.append(cirq.X(hReg[0][0]), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)  # first particle emits

        main.U_p(circuit, 2, n_i, 0, pReg, hReg, w_hReg, wReg, .5, .5)

        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten(hReg)), key='h'))
        circuit.append(cirq.measure(*TestHelpers.reverse(TestHelpers.flatten([pReg])), key='p'))
        circuit.append(cirq.measure(*TestHelpers.flatten([wReg, w_hReg]), key='w'))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # history reg should be 01 (only particle 1, the f_a fermion should have emitted)
        np.testing.assert_array_equal(np.array([0, 1]), result.measurements['h'])
        np.testing.assert_array_equal(np.array([0, 0, 1, 1, 1, 1]), result.measurements['p'])
        np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0]), result.measurements['w'])

    def test_BosonSplitting(self):
        n_i, N = 1, 1
        L = int(math.floor(math.log(N + n_i, 2)) + 1)

        initialParticles = [[1, 0, 0]]  # 1 boson
        pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], \
                                                                                             [], [], [], [], []
        main.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        circuit = cirq.Circuit()
        main.intializeParticles(circuit, pReg, initialParticles)

        circuit.append(cirq.X(hReg[0][0]), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)  # first particle emits

        g_a = g_b = 1  # equal probability of splitting into fermion a or b
        main.U_p(circuit, 2, n_i, 0, pReg, hReg, w_hReg, wReg, g_a, g_b)

        simulator = cirq.Simulator()
        # Output looks like |h[1]h[0], p1[2]p1[1]p0[0], p0[2]p0[1]p0[0], work qubits⟩
        qubitOrder = hReg[0][::-1] + pReg[1][::-1] + pReg[0][::-1] + wReg + w_hReg
        result = simulator.simulate(circuit, qubit_order=qubitOrder)
        waveVectorString = str(result)  # outputs wave-vector in order qubitOrder
        vectorString = waveVectorString.split('output vector: ', 1)[1]
        # Found it outputs in order |a anti_a⟩ + |b anti_b⟩ + |anti_a a⟩ + |anti_b b⟩
        self.assertEqual("0.5|01100110000000⟩ + 0.5|01101111000000⟩ + 0.5|01110100000000⟩ + 0.5|01111101000000⟩",
                         vectorString)



if __name__ == '__main__':
    unittest.main()
