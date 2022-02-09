import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import qiskit.providers.aer as qpa

class QuantumPartonShower:
    """
    Args:
        N (int): number of steps
        ni (int): number of initial particles
        m (int): the mth step ranging from 0 to N-1 (not needed)
    """
    def __init__(self, N, ni):
        self._N = N
        self._ni = ni
        #self._L = int(math.floor(math.log(N + ni, 2))+1)
        self._L = int(math.ceil(math.log(N + ni, 2)))

        # Define these variables for indexing - to convert from cirq's grid qubits (see explaination in notebook)
        self._p_len = 3
        self._h_len = self._L
        self._w_h_len = self._L - 1
        self._e_len = 1
        self._w_len = 5
        self._np_len = self._L
        self._wp_len = self._L - 1
        self._na_len = self._L
        self._wa_len = self._L - 1
        self._nb_len = self._L
        self._wb_len = self._L - 1

        #defining the registers
        self.pReg, self.hReg, self.w_hReg, self.eReg, self.wReg, self.n_aReg, self.w_aReg,self.n_bReg, self.w_bReg, \
        self.n_phiReg, self.w_phiReg = self.allocateQubits(N, ni, self._L)

        #print(self.pReg, self.hReg, self.w_hReg, self.eReg, self.wReg, self.n_aReg,
        #                               self.w_aReg, self.n_bReg, self.w_bReg, self.n_phiReg, self.w_phiReg)
        #self._circuit = QuantumCircuit(self.pReg, self.hReg, self.w_hReg, self.eReg, self.wReg, self.n_aReg,
        #                               self.w_aReg, self.n_bReg, self.w_bReg, self.n_phiReg, self.w_phiReg)
        self._circuit = QuantumCircuit(self.wReg, self.pReg, self.hReg, self.eReg, self.n_phiReg, self.n_aReg,
                                       self.n_bReg, self.w_hReg, self.w_phiReg, self.w_aReg, self.w_bReg)



    def flatten(self, l):
        """
        :param l: nested list of qubits in order given by fullReg
        :return: return list of qubits in order of registers as given in qubit dictionary and from MSB to LSB.
        This used to determine the order of qubits to display in the simulations results
        For a qubit order [a,b], cirq will output in the form (sum |ab>)
        """
        flatList = []
        for i in l:
            if isinstance(i, list):
                flatList.extend(self.flatten(i))
            else:
                flatList.append(i)
        return flatList

    def ptype(self, x):
        #parses particle type
        if x=='000':
            return '0'    
        if x=='001':
            return 'phi'   
        if x=='100':
            return 'f1'   
        if x=='101':
            return 'f2'   
        if x=='110':
            return 'af1'   
        if x=='111':
            return 'af2'   
        else:
            return "NAN"

    def P_f(self, t, g):
        alpha = g ** 2 * self.Phat_f(t) / (4 * math.pi)
        return alpha

    def Phat_f(self, t):
        return math.log(t)

    def Phat_bos(self, t):
        return math.log(t)

    def Delta_f(self, t, g):
        return math.exp(self.P_f(t, g))

    def P_bos(self, t, g_a, g_b):
        alpha = g_a ** 2 *self.Phat_bos(t) / (4 * math.pi) + g_b ** 2 * self.Phat_bos(t) / (4 * math.pi)
        return alpha

    def Delta_bos(self, t, g_a, g_b):
        return math.exp(self.P_bos(t, g_a, g_b))

    def populateParameterLists(self, N, timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList,
                               g_a,
                               g_b, eps):
        """Populates the 6 lists with correct values for each time step theta"""
        for i in range(N):
            # Compute time steps
            t_up = eps ** ((i) / N)
            t_mid = eps ** ((i + 0.5) / N)
            t_low = eps ** ((i + 1) / N)
            timeStepList.append(t_mid)
            # Compute values for emission matrices
            Delta_a = self.Delta_f(t_low, g_a) / self.Delta_f(t_up, g_a)
            Delta_b = self.Delta_f(t_low, g_b) / self.Delta_f(t_up, g_b)
            Delta_phi = self.Delta_bos(t_low, g_a, g_b) / self.Delta_bos(t_up, g_a, g_b)
            P_a, P_b, P_phi = self.P_f(t_mid, g_a), self.P_f(t_mid, g_b), self.P_bos(t_mid, g_a, g_b)

            # Add them to the list
            P_aList.append(P_a)
            P_bList.append(P_b)
            P_phiList.append(P_phi)
            Delta_aList.append(Delta_a)
            Delta_bList.append(Delta_b)
            Delta_phiList.append(Delta_phi)

    def allocateQubits(self, N, n_i, L):
        nqubits_p = 3 * (N + n_i)
        nqubits_h = N * math.ceil(math.log2((N + n_i)))
        #nqubits_h = N * int(math.floor(math.log2((N + n_i)) + 1))
        nqubits_e = 1
        nqubits_a_b_phi = L

        pReg = QuantumRegister(nqubits_p, 'p')
        # pReg = QuantumRegister(6)

        hReg = QuantumRegister(nqubits_h, 'h')
        #w_hReg = QuantumRegister(nqubits_h, 'w_h')
        w_hReg = QuantumRegister(L, 'w_h')

        eReg = QuantumRegister(nqubits_e, 'e')
        wReg = QuantumRegister(5, 'w')  # we use all 5 of these work register qubits, but not sure why it is 5

        n_aReg = QuantumRegister(nqubits_a_b_phi, 'n_a')
        w_aReg = QuantumRegister(nqubits_a_b_phi, 'w_a')

        n_bReg = QuantumRegister(nqubits_a_b_phi, 'n_b')
        w_bReg = QuantumRegister(nqubits_a_b_phi, 'w_b')

        n_phiReg = QuantumRegister(nqubits_a_b_phi, 'n_phi')
        w_phiReg = QuantumRegister(nqubits_a_b_phi, 'w_phi')
        #print(pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
        return (pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)

    def initializeParticles(self, circuit, pReg, initialParticles):
        """ Apply appropriate X gates to ensure that the p register contains all of the initial particles.
            The p registers contains particles in the form of a list [LSB, middle bit, MSB]"""
        for currentParticleIndex in range(len(initialParticles)):
            for particleBit in range(3):
                if int(initialParticles[currentParticleIndex][particleBit]) == 1:
                    circuit.x(pReg[currentParticleIndex * self._p_len + particleBit])

    def flavorControl(self, circuit, flavor, control, target, ancilla, control_index, target_index, ancilla_index):
        """Controlled x onto targetQubit if "control" particle is of the correct flavor"""
        if flavor == "phi":
            circuit.x(control[control_index + 1])
            circuit.x(control[control_index + 2])
            circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index])
            circuit.ccx(control[control_index + 2], ancilla[ancilla_index], target[target_index + 0])
            # undo work
            circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index])
            #circuit.x(1)
            #circuit.x(2)
            circuit.x(control[control_index + 1])
            circuit.x(control[control_index + 2])
        if flavor == "a":
            #circuit.x(0)
            circuit.x(control[control_index + 0])
            circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])
            # undo work
            #circuit.x(0)
            circuit.x(control[control_index + 0])
        if flavor == "b":
            circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])


    def plus1(self, circuit, l, countReg, workReg, control, ancilla, level):
        """
        Recursively add 1 to the LSB of a register and carries to all bits, if control == 1
        l: number of qubits in count register
        countReg, workReg: count register and associated work register
        control: control qubit to determine if plus1 should be executed
        ancilla: extra work qubit
        level: current qubit we are operating on, recursively travels from qubit 0 to l-1
        """
        # apply X to LSB
        if level == 0:
            circuit.cx(control, countReg[0])
        if level < l - 1:
            # first level uses CNOT instead of TOFFOLI gate
            if level == 0:
                # move all X gates to first step to avoid unnecesarry gates
                [circuit.x(qubit) for qubit in countReg]
                circuit.ccx(countReg[0], control, workReg[0])
            else:
                circuit.ccx(countReg[level], workReg[level - 1], ancilla)
                circuit.ccx(ancilla, control, workReg[level])
                circuit.ccx(countReg[level], workReg[level - 1], ancilla)

            circuit.ccx(workReg[level], control, countReg[level + 1])
            # recursively call next layer
            self.plus1(circuit, l, countReg, workReg, control, ancilla, level + 1)
            # undo work qubits (exact opposite of first 7 lines - undoes calculation)
            if level == 0:
                circuit.ccx(countReg[0], control, workReg[0])
                [circuit.x(qubit) for qubit in countReg]
            else:
                circuit.ccx(countReg[level], workReg[level - 1], ancilla)
                circuit.ccx(ancilla, control, workReg[level])
                circuit.ccx(countReg[level], workReg[level - 1], ancilla)

    def uCount(self, circuit, m, n_i, l, pReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg):
        """
        Populate the count registers using current particle states.
        Uses wReg[0] as the control and wReg[1] as ancilla qubit for flavorControl and plus1, respectively
        """
        for k in range(n_i + m):
            # bosons
            self.flavorControl(circuit, "phi", pReg, wReg, wReg, (k * self._p_len), 0, 1)
            self.plus1(circuit, l, n_phiReg, w_phiReg, wReg[0], wReg[1], 0)
            self.flavorControl(circuit, "phi", pReg, wReg, wReg, (k * self._p_len), 0, 1)
            # a fermions
            self.flavorControl(circuit, "a", pReg, wReg, wReg, (k * self._p_len), 0, 1)
            self.plus1(circuit, l, n_aReg, w_aReg, wReg[0], wReg[1], 0)
            self.flavorControl(circuit, "a", pReg, wReg, wReg, (k * self._p_len), 0, 1)
            # b fermions
            self.flavorControl(circuit, "b", pReg, wReg, wReg, (k * self._p_len), 0, 1)
            self.plus1(circuit, l, n_bReg, w_bReg, wReg[0], wReg[1], 0)
            self.flavorControl(circuit, "b", pReg, wReg, wReg, (k * self._p_len), 0, 1)

    def generateParticleCounts(self, n_i, m, k):
        """Fill countsList with all combinations of n_phi, n_a, and n_b where each n lies in range [0, n_i+m-k],
        and the sum of all n's lies in range [n_i-k, m+n_i-k], all inclusive
        """
        countsList = []
        for numParticles in range(n_i - k, m + n_i - k + 1):
            for numPhi in range(0, n_i + m - k + 1):
                for numA in range(0, numParticles - numPhi + 1):
                    numB = numParticles - numPhi - numA
                    countsList.append([numPhi, numA, numB])
        return countsList

    def reverse(self, lst):
        """reverse a list in place"""
        lst.reverse()
        return lst

    def intToBinary(self, l, number):
        """Converts integer to binary list of size l with LSB first and MSB last"""
        numberBinary = [int(x) for x in list('{0:0b}'.format(number))]
        numberBinary = (l - len(numberBinary)) * [0] + numberBinary
        return self.reverse(numberBinary)

    def numberControl(self, circuit, l, number, countReg, workReg):
        """
        Applies an X to the l-2 (0 indexed) qubit of the work register if count register encodes the inputted number in binary
        returns this l-2 qubit, unless l=1, in which case return the only count register qubit
        DOES NOT CLEAN AFTER ITSELF - USE numberControlT to clean after this operation
        """
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number
        #print('length of numberBinary: ' + str(len(numberBinary)))
        #print('length of countReg: ' + str(len(countReg)))
        [circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]

        # first level does not use work qubits as control
        if l > 1:
            circuit.ccx(countReg[0], countReg[1], workReg[0])
            # subfunction to recursively handle toffoli gates

        def binaryToffolis(level):
            circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
            if level < l - 1:
                binaryToffolis(level + 1)

        if l > 2:
            binaryToffolis(2)
        # return qubit containing outcome of the operation
        if l == 1:
            return countReg[0]
        else:
            return workReg[l - 2]

    def numberControlT(self, circuit, l, number, countReg, workReg):
        """CLEANS AFTER numberControl operation"""
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number

        # subfunction to recursively handle toffoli gates
        def binaryToffolisT(level):
            # circuit.append(TOFFOLI(countReg[level], workReg[level-2], workReg[level-1]), strategy=new)
            if level < l:
                binaryToffolisT(level + 1)
                # undo
                circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])

        if l > 2:
            binaryToffolisT(2)
            # undo
        if l > 1:
            circuit.ccx(countReg[0], countReg[1], workReg[0])
            # undo
        [circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]

    def uE(self, circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, Delta_phi, Delta_a,
           Delta_b):
        """Determine if emission occured in current step m"""
        countsList = self.generateParticleCounts(n_i, m, 0)
        #print('\nm=%d, Length of countsList= ' %(m) + str(len(countsList)))
        for counts in countsList:
            n_phi, n_a, n_b = counts[0], counts[1], counts[2]
            Delta = Delta_phi ** n_phi * Delta_a ** n_a * Delta_b ** n_b
            phiControlQub = self.numberControl(circuit, l, n_phi, n_phiReg, w_phiReg)
            aControlQub = self.numberControl(circuit, l, n_a, n_aReg, w_aReg)
            bControlQub = self.numberControl(circuit, l, n_b, n_bReg, w_bReg)
            circuit.ccx(phiControlQub, aControlQub, wReg[0])
            circuit.ccx(bControlQub, wReg[0], wReg[1])
            circuit.cry((2 * math.acos(np.sqrt(Delta))), wReg[1], eReg[0])
            # undo
            circuit.ccx(bControlQub, wReg[0], wReg[1])
            circuit.ccx(phiControlQub, aControlQub, wReg[0])
            self.numberControlT(circuit, l, n_b, n_bReg, w_bReg)
            self.numberControlT(circuit, l, n_a, n_aReg, w_aReg)
            self.numberControlT(circuit, l, n_phi, n_phiReg, w_phiReg)

    def generateGrayList(self, l, number):
        """
        l is the size of the current count register
        Return list of elements in gray code from |0> to |number> where each entry is of type[int, binary list].
        int: which bit is the target in the current iteration, binary list: the state of the rest of the qubits (controls)
        """
        grayList = [[0, l * [0]]]
        targetBinary = self.intToBinary(l, number)
        for index in range(len(targetBinary)):
            if targetBinary[index] == 1:
                grayList.append([index, (list(grayList[-1][1]))])
                grayList[-1][1][index] = 1
        return grayList[1:]

    def twoLevelControlledRy(self, circuit, l, angle, k, externalControl, reg, workReg):
        """
        Implements two level Ry rotation from state |0> to |k>, if externalControl qubit is on
        for reference: http://www.physics.udel.edu/~msafrono/650/Lecture%206.pdf
        """
        #print("Generating gray list with     l= %d, k= %d" %(l, k))
        grayList = self.generateGrayList(l, k)
        #print('Graylist: ' + str(grayList))
        #print('reg: ' + str(reg))
        # handle the case where l=0 or 1
        if k == 0:
            return
        if l == 1 and k == 1:
            circuit.cry(angle, externalControl, reg[0])
            return

        # swap states according to Gray Code until one step before the end
        for element in grayList:
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
            if element == grayList[-1]:  # reached end
                circuit.ccx(controlQub, externalControl, workReg[l - 2])
                circuit.cry(angle, workReg[l - 2], reg[targetQub])
                circuit.ccx(controlQub, externalControl, workReg[l - 2])
            else:  # swap states
                circuit.cx(controlQub, reg[targetQub])
            self.numberControlT(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
        # undo
        for element in self.reverse(grayList[:-1]):
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
            circuit.cx(controlQub, reg[targetQub])
            self.numberControlT(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
        return

    def U_hAngle(self, flavor, n_phi, n_a, n_b, P_phi, P_a, P_b):
        """Determine angle of rotation used in U_h"""
        denominator = n_phi * P_phi + n_a * P_a + n_b * P_b
        # print("denom: ", denominator)
        # print("n vals: ", n_a, n_b, n_phi)
        if denominator == 0:  # occurs if we are trying the case of no particles remaining (n_a = n_b = n_phi = 0)
            return 0
        flavorStringToP = {'phi': P_phi, 'a': P_a, 'b': P_b}
        # print("numerator: ", flavorStringToP)
        # print("flavor: ", flavor)
        emissionAmplitude = np.sqrt(flavorStringToP[flavor] / denominator)
        # correct for arcsin input greater than 1 errors for various input combinations that are irrelevant anyway
        emissionAmplitude = min(1, emissionAmplitude)
        # print("angle: ", 2 * np.arcsin(emissionAmplitude))
        return 2 * np.arcsin(emissionAmplitude)

    def minus1(self, circuit, l, countReg, workReg, control, ancilla, level):
        """
        Recursively carries an subtraction of 1 to the LSB of a register to all bits if control == 1
        Equivalent to plus1 but with an X applied to all count qubits before and after gate
        """
        [circuit.x(qubit) for qubit in countReg]
        self.plus1(circuit, l, countReg, workReg, control, ancilla, level)
        [circuit.x(qubit) for qubit in countReg]

    def U_h(self, circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, pReg, hReg, w_hReg,
            P_phi,
            P_a, P_b):
        """Implement U_h from paper"""
        #print('\nm= ' + str(m))
        #if m == 2:
            #print(hReg)
            #print('h_len= ' + str(self._h_len))
            #print(hReg[4:6])
            #print(hReg[m*self._h_len : (m+1)*self._h_len])
        for k in range(n_i + m):
            # for k in range(1):
            #print("k: ", k)
            countsList = self.generateParticleCounts(n_i, m, k)  # reduce the available number of particles

            for counts in countsList:
                #print('counts: ' + str(counts))
                n_phi, n_a, n_b = counts[0], counts[1], counts[2]
                # controlled R-y from |0> to |k> on all qubits with all possible angles depending on n_phi, n_a, n_b, and flavor
                # for flavor in ['phi']:
                # if n_phi == 3 and n_a == 0 and n_b ==0:
                # print("counts: ", counts)
                # print("after if : ", n_phi, n_a, n_b)

                for flavor in ['phi', 'a', 'b']:
                    angle = self.U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b)

                    #phiControl, aControl, and bControl are the corresponding work registers, and since we call
                    # numberControl we also add x gates on the respective number registers (aka n_phi, n_a, n_b)
                    phiControl = self.numberControl(circuit, l, n_phi, n_phiReg, w_phiReg)
                    # print("qiskit phiControl: ", phiControl)
                    aControl = self.numberControl(circuit, l, n_a, n_aReg, w_aReg)
                    # print("qiskit aControl: ", aControl)
                    bControl = self.numberControl(circuit, l, n_b, n_bReg, w_bReg)
                    # print("qiskit bControl: ", bControl)

                    circuit.ccx(phiControl, aControl, wReg[0])
                    circuit.ccx(bControl, wReg[0], wReg[1])

                    self.flavorControl(circuit, flavor, pReg, wReg, wReg, (k * self._p_len), 2,
                                  4)  # wReg[4] is work qubit but is reset to 0
                    circuit.ccx(wReg[1], wReg[2], wReg[3])
                    circuit.ccx(eReg[0], wReg[3], wReg[4])

                    #print('flavor: ' + flavor)
                    #self.twoLevelControlledRy(circuit, l, angle, k + 1, wReg[4], hReg, w_hReg)
                    self.twoLevelControlledRy(circuit, l, angle, k + 1, wReg[4], hReg[m*self._h_len : (m+1)*self._h_len], w_hReg)

                    circuit.ccx(eReg[0], wReg[3], wReg[4])  # next steps undo work qubits
                    circuit.ccx(wReg[1], wReg[2], wReg[3])
                    self.flavorControl(circuit, flavor, pReg, wReg, wReg, (k * self._p_len), 2,
                                  4)  # wReg[4] is work qubit but is reset to 0
                    circuit.ccx(bControl, wReg[0], wReg[1])
                    circuit.ccx(phiControl, aControl, wReg[0])
                    self.numberControlT(circuit, l, n_b, n_bReg, w_bReg)
                    self.numberControlT(circuit, l, n_a, n_aReg, w_aReg)
                    self.numberControlT(circuit, l, n_phi, n_phiReg, w_phiReg)

            # subtract from the counts register depending on which flavor particle emitted
            for flavor, countReg, workReg in zip(['phi', 'a', 'b'], [n_phiReg, n_aReg, n_bReg],
                                                 [w_phiReg, w_aReg, w_bReg]):
                self.flavorControl(circuit, flavor, pReg, wReg, wReg, (k * self._p_len), 0,
                              1)  # wReg[4] is work qubit but is reset to 0
                self.minus1(circuit, l, countReg, workReg, wReg[0], wReg[1], 0)
                self.flavorControl(circuit, flavor, pReg, wReg, wReg, (k * self._p_len), 0,
                              1)  # wReg[4] is work qubit but is reset to 0

        # apply x on eReg if hReg[m] = 0, apply another x so we essentially control on not 0 instead of 0
        isZeroControl = self.numberControl(circuit, l, 0, hReg[m*self._L : (m+1)*self._L], w_hReg)
        circuit.cx(isZeroControl, eReg[0])
        circuit.x(eReg[0])
        self.numberControlT(circuit, l, 0, hReg[m*self._L : (m+1)*self._L], w_hReg)

        # Alternate: use reset
        #circuit.reset(eReg)


    def updateParticles(self, circuit, l, n_i, m, k, pReg, wReg, controlQub, g_a, g_b):
        """Updates particle if controlQub is on"""
        oldParticleReg = pReg
        newParticleReg = pReg
        # first gate in paper U_p
        # print("k: ", k)
        # print("m: ", m)
        # print("ni: ", n_i)
        # print("pReg: ", pReg)
        # print(oldParticleReg)
        # print("new particle register: ", newParticleReg)
        #
        # print(" oldParticleReg[k*self._p_len + 2]: ", oldParticleReg[k * self._p_len + 2])
        # print("newParticleReg[(n_i+m)*self._p_len+0]: ", newParticleReg[(n_i + m) * self._p_len + 0])
        circuit.ccx(controlQub, oldParticleReg[k * self._p_len + 2], newParticleReg[(n_i + m) * self._p_len + 0])
        # second gate in paper (undoes work register immediately)
        circuit.x(oldParticleReg[k * self._p_len + 1])
        circuit.x(oldParticleReg[k * self._p_len + 2])
        circuit.ccx(controlQub, oldParticleReg[k * self._p_len + 2], wReg[0])
        circuit.ccx(wReg[0], oldParticleReg[k * self._p_len + 1], wReg[1])
        circuit.ccx(wReg[1], oldParticleReg[k * self._p_len + 0], newParticleReg[(n_i + m) * self._p_len + 2])
        circuit.ccx(wReg[0], oldParticleReg[k * self._p_len + 1], wReg[1])
        circuit.ccx(controlQub, oldParticleReg[k * self._p_len + 2], wReg[0])
        circuit.x(oldParticleReg[k * self._p_len + 1])
        circuit.x(oldParticleReg[k * self._p_len + 2])
        # third gate in paper
        circuit.ccx(controlQub, newParticleReg[(n_i + m) * self._p_len + 2], oldParticleReg[k * self._p_len + 2]) ############################
        # fourth and fifth gate in paper (then undoes work register)
        circuit.ccx(controlQub, newParticleReg[(n_i + m) * self._p_len + 2], wReg[0])
        # check the format for the control state here
        circuit.ch(wReg[0], newParticleReg[(n_i + m) * self._p_len + 1])
        angle = (2 * np.arccos(g_a / np.sqrt(g_a ** 2 + g_b ** 2)))
        circuit.cry(angle, wReg[0], newParticleReg[(n_i + m) * self._p_len + 0])
        circuit.ccx(controlQub, newParticleReg[(n_i + m) * self._p_len + 2], wReg[0])
        # sixth and seventh gate in paper (then undoes work register)
        circuit.x(newParticleReg[(n_i + m) * self._p_len + 0])
        circuit.x(newParticleReg[(n_i + m) * self._p_len + 1])
        circuit.ccx(newParticleReg[(n_i + m) * self._p_len + 1], newParticleReg[(n_i + m) * self._p_len + 2], wReg[0])
        circuit.ccx(controlQub, wReg[0], oldParticleReg[k * self._p_len + 1])
        circuit.ccx(newParticleReg[(n_i + m) * self._p_len + 1], newParticleReg[(n_i + m) * self._p_len + 2], wReg[0])
        circuit.ccx(newParticleReg[(n_i + m) * self._p_len + 0], newParticleReg[(n_i + m) * self._p_len + 2], wReg[0])
        circuit.ccx(controlQub, wReg[0], oldParticleReg[k * self._p_len + 0])
        circuit.ccx(newParticleReg[(n_i + m) * self._p_len + 0], newParticleReg[(n_i + m) * self._p_len + 2], wReg[0])
        circuit.x(newParticleReg[(n_i + m) * self._p_len + 0])
        circuit.x(newParticleReg[(n_i + m) * self._p_len + 1])

    def U_p(self, circuit, l, n_i, m, pReg, hReg, w_hReg, wReg, g_a, g_b):
        """Applies U_p from paper"""
        for k in range(0, n_i + m):
            #controlQub = self.numberControl(circuit, l, k + 1, hReg, w_hReg)
            controlQub = self.numberControl(circuit, l, k + 1, hReg[m*self._h_len : (m+1)*self._h_len], w_hReg)
            self.updateParticles(circuit, l, n_i, m, k, pReg, wReg, controlQub, g_a, g_b)
            #self.numberControlT(circuit, l, k + 1, hReg, w_hReg)
            #self.numberControlT(circuit, l, k + 1, hReg[m:(m + self._h_len)], w_hReg)
            self.numberControlT(circuit, l, k + 1, hReg[m*self._h_len : (m+1)*self._h_len], w_hReg)

    def createCircuit(self, eps, g_1, g_2, g_12, initialParticles):
        """
        Create full circuit with n_i initial particles and N steps
        Inputs:
        n_i: number of initial particles
        N: number of steps
        eps, g_1, g_2, g_12: pre-chosen qft parameters
        initialParticles: list of initial particles, each particle in a list of qubits [MSB, middle bit, LSB]
        (opposite order of the paper pg 6 - e.g a f_a fermion is [0,0,1])
        in order [particle 1, particle 2, ..... particle n_i]
        """
        # calculate constants
        gp = math.sqrt(abs((g_1 - g_2) ** 2 + 4 * g_12 ** 2))
        if g_1 > g_2:
            gp = -gp
        g_a, g_b = (g_1 + g_2 - gp) / 2, (g_1 + g_2 + gp) / 2
        u = math.sqrt(abs((gp + g_1 - g_2) / (2 * gp)))
        #print('ga= %.4f, gb= %.4f, u= %.4f' %(g_a, g_b, u))
        # evaluate P(Theta) and Delta(Theta) at every time step
        timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList = [], [], [], [], [], [], []
        self.populateParameterLists(self._N, timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList,
                               g_a,
                               g_b, eps)

        qubits = {'pReg': self.pReg, 'hReg': self.hReg, 'w_hReg': self.w_hReg, 'eReg': self.eReg, 'wReg': self.wReg,
                  'n_aReg': self.n_aReg, 'w_aReg': self.w_aReg, 'n_bReg': self.n_bReg, 'w_bReg': self.w_bReg,
                  'n_phiReg': self.n_phiReg, 'w_phiReg': self.w_phiReg}

        self.initializeParticles(self._circuit, self.pReg, initialParticles)

        # begin stepping through subcircuits
        for m in range(self._N):
            l = int(math.floor(math.log(m + self._ni, 2)) + 1)

            # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
            index = 0
            while index < self.pReg.size:
                self._circuit.cry((2 * math.asin(-u)), self.pReg[index + 2], self.pReg[index + 0])
                index += self._p_len

            # populate count register (step 2)
            self.uCount(self._circuit, m, self._ni, l, self.pReg, self.wReg, self.n_aReg, self.w_aReg, self.n_bReg,
                        self.w_bReg, self.n_phiReg, self.w_phiReg)

            # assess if emmision occured (step 3)
            self.uE(self._circuit, l, self._ni, m, self.n_phiReg, self.w_phiReg, self.n_aReg, self.w_aReg, self.n_bReg,
                    self.w_bReg, self.wReg, self.eReg,
               Delta_phiList[m], Delta_aList[m], Delta_bList[m])

            # choose a particle to split (step 4)
            #if m == 2:
                #print('l, ni, m, pReg, hReg, P_phiList[m], P_aList[m], P_bList[m] ...')
                #print(l, self._ni, m, self.pReg, self.hReg, P_phiList[2], P_aList[2], P_bList[2])
            self.U_h(self._circuit, l, self._ni, m, self.n_phiReg, self.w_phiReg, self.n_aReg, self.w_aReg, self.n_bReg,
                     self.w_bReg, self.wReg, self.eReg, self.pReg, self.hReg, self.w_hReg,
                P_phiList[m], P_aList[m], P_bList[m])

            # update particle based on which particle split/emmitted (step 5)
            self.U_p(self._circuit, l, self._ni, m, self.pReg, self.hReg, self.w_hReg, self.wReg, g_a, g_b)

            # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
            index2 = 0
            while index2 < self.pReg.size:
                # circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
                self._circuit.cry((2 * math.asin(u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
                index2 += self._p_len

        #print('generated circuit on', len(self.flatten(list(qubits.values()))), 'qubits')

        
        return self._circuit, qubits


    def allocateClbits(self, N, n_i, L):
        nbits_p = 3 * (N + n_i)
        nbits_h = N * math.ceil(math.log2((N + n_i)))
        #nbits_h = N * int(math.floor(math.log2(N + n_i)) + 1)
        nbits_e = 1
        nbits_a_b_phi = L

        wReg_cl = ClassicalRegister(5, 'w_cl')  # we use all 5 of these work register qubits, but not sure why it is 5
        pReg_cl = []
        for j in range(N + n_i):
            pReg_cl.append(ClassicalRegister(3, 'p%d_cl' %(j)))
        hReg_cl = []
        for j in range(N):
            #hReg_cl.append(ClassicalRegister(int(math.floor(math.log2(N + n_i)) + 1), 'h%d_cl' %(j)))
            hReg_cl.append(ClassicalRegister(int(math.ceil(math.log2(N + n_i))), 'h%d_cl' %(j)))
        eReg_cl = ClassicalRegister(nbits_e, 'e_cl')
        n_phiReg_cl = ClassicalRegister(nbits_a_b_phi, 'nphi_cl')
        n_aReg_cl = ClassicalRegister(nbits_a_b_phi, 'na_cl')
        n_bReg_cl = ClassicalRegister(nbits_a_b_phi, 'nb_cl')

        #w_hReg_cl = ClassicalRegister(nbits_h, 'wh_cl')
        w_hReg_cl = ClassicalRegister(L, 'wh_cl')
        w_phiReg_cl = ClassicalRegister(nbits_a_b_phi, 'wphi_cl')
        w_aReg_cl = ClassicalRegister(nbits_a_b_phi, 'wa_cl')
        w_bReg_cl = ClassicalRegister(nbits_a_b_phi, 'wb_cl')

        return (wReg_cl, pReg_cl, hReg_cl, eReg_cl, n_phiReg_cl, n_aReg_cl, n_bReg_cl, 
                w_hReg_cl, w_phiReg_cl, w_aReg_cl, w_bReg_cl)


    def simulate(self, type, shots=None, position=False):
        """
        :param type: either the qasm simulaot or the statevector simulator
        :param shots: if using the qasm simulator the number of shots needs to be specified
        :param position: the statevector is very long, so if position=True the function will print the value and
        position of tbe non-zero elements
        :return: either counts (qasm) or the statevector
        """
        if type == 'qasm' or type == 'mps':
            if type == 'qasm':
                simulator = Aer.get_backend('qasm_simulator')
            else:
                simulator = qpa.QasmSimulator(method= 'matrix_product_state')

            (wReg_cl, pReg_cl, hReg_cl, # Note: pReg_cl, hReg_cl aer lists of ClassicalRegisters
             eReg_cl, n_phiReg_cl, n_aReg_cl, 
             n_bReg_cl, w_hReg_cl, w_phiReg_cl, w_aReg_cl, w_bReg_cl) = self.allocateClbits(self._N, self._ni, self._L)
            
            self._circuit.add_register(wReg_cl)
            for j in range(self._N + self._ni):
                self._circuit.add_register(pReg_cl[j])
            for j in range(self._N):
                self._circuit.add_register(hReg_cl[j])
            self._circuit.add_register(eReg_cl)
            self._circuit.add_register(n_phiReg_cl)
            self._circuit.add_register(n_aReg_cl)
            self._circuit.add_register(n_bReg_cl)
            self._circuit.add_register(w_hReg_cl)
            self._circuit.add_register(w_phiReg_cl)
            self._circuit.add_register(w_aReg_cl)
            self._circuit.add_register(w_bReg_cl)

            self._circuit.measure(self.wReg, wReg_cl)
            for j in range(self._N + self._ni):
                self._circuit.measure(self.pReg[3*j : 3*(j+1)], pReg_cl[j])
            for j in range(self._N):
                self._circuit.measure(self.hReg[self._L*j : self._L*(j+1)], hReg_cl[j])
            self._circuit.measure(self.eReg, eReg_cl)
            self._circuit.measure(self.n_phiReg, n_phiReg_cl)
            self._circuit.measure(self.n_aReg, n_aReg_cl)
            self._circuit.measure(self.n_bReg, n_bReg_cl)
            self._circuit.measure(self.w_hReg, w_hReg_cl)
            self._circuit.measure(self.w_phiReg, w_phiReg_cl)
            self._circuit.measure(self.w_aReg, w_aReg_cl)
            self._circuit.measure(self.w_bReg, w_bReg_cl)

            job = execute(self._circuit, simulator, shots=shots)
            result = job.result()
            counts = result.get_counts(self._circuit)
            return counts
        elif type == 'statevector':
            simulator = Aer.get_backend('statevector_simulator')
            result = execute(self._circuit, simulator).result()
            statevector = result.get_statevector(self._circuit)
            if position:
                [print("position of non zero element: ", list(statevector).index(i), "\nvalue: ",
                       i, "\nabsolute value: ", abs(i)) for i in statevector if abs(i) > 10 ** (-5)]
            return statevector
        else:
            print("choose 'qasm' or 'statevector'")


    def simulate2(self, type, shots=None, position=False):
        """
        :param type: either the qasm simulaot or the statevector simulator
        :param shots: if using the qasm simulator the number of shots needs to be specified
        :param position: the statevector is very long, so if position=True the function will print the value and
        position of tbe non-zero elements
        :return: either counts (qasm) or the statevector
        """
        if type == 'qasm':
            simulator = Aer.get_backend('qasm_simulator')
            self._circuit.measure_all()
            job = execute(self._circuit, simulator, shots=shots)
            result = job.result()
            counts = result.get_counts(self._circuit)
            return counts
        elif type == 'statevector':
            simulator = Aer.get_backend('statevector_simulator')
            result = execute(self._circuit, simulator).result()
            statevector = result.get_statevector(self._circuit)
            if position:
                [print("position of non zero element: ", list(statevector).index(i), "\nvalue: ",
                       i, "\nabsolute value: ", abs(i)) for i in statevector if abs(i) > 10 ** (-5)]
            return statevector
        else:
            print("choose 'qasm' or 'statevector'")



    def bar_plot(self, counts, events, eps, g1, g2, counts2= None):
        mycounter = 0

        firstisf1_y = []
        firstisf1_ey = []
        firstisf1_x = []

        firstisf2_y = []
        firstisf2_ey = []
        firstisf2_x = []

        if counts2 != None:
            firstisf1b_y = []
            firstisf1b_ey = []
            firstisf1b_x = []

            firstisf2b_x = []
            firstisf2b_y = []
            firstisf2b_ey = []

        mymap = {}
        mymap['0','0']=1
        mymap['phi','0']=2
        mymap['0','phi']=2
        mymap['phi','phi']=3
        mymap['af1','f1']=4
        mymap['f1','af1']=4
        mymap['af2','f2']=5
        mymap['f2','af2']=5
        mymap['af2','f1']=6
        mymap['f2','af1']=6
        mymap['af1','f2']=6
        mymap['f1','af2']=6

        firstemission = [0,0,0,0]
        if counts2 != None:
            firstemission2 = [0,0,0,0]

        for c in counts:
            print(mycounter, c, self.ptype(c.split()[10]), self.ptype(c.split()[11]), self.ptype(c.split()[12]), counts[c])
            mycounter+=1

            if c.split()[9]=='0':
                if c.split()[8]=='01' or c.split()[8]=='10' or c.split()[8]=='11':
                    firstemission[0]+= counts[c]/events
                else:
                    firstemission[1]+= counts[c]/events
            else:
                if c.split()[8]=='01' or c.split()[8]=='10' or c.split()[8]=='11':
                    firstemission[2]+= counts[c]/events
                else:
                    firstemission[3]+= counts[c]/events

            if (self.ptype(c.split()[12])=='f1'):
                firstisf1_y+=[100*counts[c]/events]
                firstisf1_ey+=[100*counts[c]**0.5/events]
                firstisf1_x+=[-0.2+mymap[self.ptype(c.split()[10]), self.ptype(c.split()[11])]]
                pass
            if (self.ptype(c.split()[12])=='f2'):
                firstisf2_y+=[100*counts[c]/events]
                firstisf2_ey+=[100*counts[c]**0.5/events]
                firstisf2_x+=[0.0+mymap[self.ptype(c.split()[10]), self.ptype(c.split()[11])]]
                pass

        if counts2 != None:
            for c in counts2:

                if c.split()[9]=='0':
                    if c.split()[8]=='01' or c.split()[8]=='10' or c.split()[8]=='11':
                        firstemission2[0]+=counts2[c]/events
                    else:
                        firstemission2[1]+=counts2[c]/events
                else:
                    if c.split()[8]=='01' or c.split()[8]=='10' or c.split()[8]=='11':
                        firstemission2[2]+=counts2[c]/events
                    else:
                        firstemission2[3]+=counts2[c]/events

                if (self.ptype(c.split()[12])=='f1'):
                    firstisf1b_y+=[100*counts2[c]/events]
                    firstisf1b_ey+=[100*counts2[c]**0.5/events]
                    firstisf1b_x+=[0.2+mymap[self.ptype(c.split()[10]),self.ptype(c.split()[11])]]
                    pass
                if (self.ptype(c.split()[12])=='f2'):
                    firstisf2b_y+=[100*counts2[c]/events]
                    firstisf2b_ey+=[100*counts2[c]**0.5/events]
                    firstisf2b_x+=[0.2+mymap[self.ptype(c.split()[10]),self.ptype(c.split()[11])]]
                    pass


        emits_classical = []
        Nev_classical = events
        for i in range(2):

            t_up2 = eps**((i)/2)
            t_mid2 =  eps**((i+0.5)/2)  
            t_low2 =  eps**((i+1)/2)  

            deltaL2 = math.sqrt(self.Delta_f(t_low2, g1)) / math.sqrt(self.Delta_f(t_up2, g1))
            pL2 = 1. - deltaL2*deltaL2
            emits_classical+=[np.random.binomial(1, pL2, Nev_classical)]
            pass

        counts_classical = {}
        counts_classical['0 phi f1']=0
        counts_classical['phi 0 f1']=0
        counts_classical['phi phi f1']=0
        counts_classical['0 0 f1']=0
        for j in range(Nev_classical):
            if emits_classical[0][j]==1 and emits_classical[1][j]==1:
                counts_classical['phi phi f1']+=1
            elif emits_classical[0][j]==0 and emits_classical[1][j]==1:
                counts_classical['phi 0 f1']+=1
            elif emits_classical[0][j]==1 and emits_classical[1][j]==0:
                counts_classical['0 phi f1']+=1
            else:
                counts_classical['0 0 f1']+=1
                pass
            pass

        print("Sanity check: g12 = 0, phi->ff = 0, classical")
        for c in counts_classical:
            print(c,counts_classical[c])
        

        #print("Sanity check: g12 = 0, phi->ff = 0, full quantum")
        #for c in counts2b:
        #    print(self.ptype(c.split()[10]), self.ptype(c.split()[11]), self.ptype(c.split()[12]), counts2b[c])

        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(1, 1, 1)
        plt.ylim((100*1e-4, 100*5.))
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]')
        bar1 = plt.bar(firstisf1_x, firstisf1_y, color='#228b22', width=0.2, label=r"$f' = f_{1}$", hatch='\\') #,yerr=firstisf1_ey)
        bar1b = plt.bar(firstisf2_x, firstisf2_y, color='#01B9FF', width=0.2, label=r"$f' = f_{2}$", hatch='//') #,yerr=firstisf2_ey)

        ax.set_xticks([1,2,3,4,5,6])
        ax.set_xticklabels( (r"$f_{1}\rightarrow f'$", r"$f_{1}\rightarrow f'\phi$", r"$f_{1}\rightarrow f'\phi\phi$",r"$f_{1}\rightarrow f' f_{1} \bar{f}_{1}$",r"$f_{1}\rightarrow f' f_{2} \bar{f}_{2}$",r"$f_{1}\rightarrow f' f_{1/2} \bar{f}_{2/1}$") )

        plt.legend(loc='upper right',prop={'size': 9.5})

        if counts2 != None:
            bar2 = plt.bar(firstisf1b_x, firstisf1b_y, color='#FF4949', width=0.2, label=r"$f' = f_{1}$", alpha=1.0) #,hatch="//")
            #plt.bar(firstisf2b_x,firstisf2b_y,color='blue',width=0.4,label=r"$f' = f_{2}$",yerr=firstisf2b_ey,alpha=0.5)
            leg2 = ax.legend([bar1, bar1b, bar2],[r"$f' = f_{1}, g_{12} = 1$",r"$f' = f_{2}, g_{12} = 1$",r"$f' = f_{1}, g_{12} = 0$"], loc='upper right', prop={'size': 12.5}, frameon=False)
        else:
            leg2 = ax.legend([bar1,bar1b], [r'$g_{12} = 1$',r'$g_{12} = 0$'], loc='upper right',frameon=False,prop={'size': 12.5},bbox_to_anchor=(1.,0.8))

        ax.add_artist(leg2);

        plt.text(0.7, 55*3, r"2-step Full Quantum Simulation", fontsize=14)
        plt.text(1.5, 30*2.8, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=10)

        #f.savefig("fullsim2step_states.pdf")
        plt.show()
        #print(sum(firstisf1b_y))
        print(sum(firstisf1_y))
        #print(sum(firstisf2_y))

        print(firstemission)
        #print(firstemission2)






    def bar_plot3(self, counts, events, eps, g1, g2, counts2= None):

        mycounter = 0
        mycounter2 = 0

        y = []
        ey = []
        x = []
        if counts2 != None:
            y2 = []
            ey2 = []
            x2 = []

        # Perhaps sort by uniqueness of number of where phis are
        def mymap(pTup):
            # pTup: sorted tuple/list of particles
            #print(pTup)
            if pTup[-1] == 'phi':
                if pTup[-2] == 'phi':
                    if pTup[-3] == 'phi':
                        return 3
                    return 2
                return 1
            return 0


        for c in counts:
            pList= list((self.ptype(c.split()[11]), self.ptype(c.split()[12]), self.ptype(c.split()[13]), self.ptype(c.split()[14])))
            #print(mycounter, c, pList, counts[c])
            mycounter+=1

            y+=[100*counts[c]/events]
            ey+=[100*counts[c]**0.5/events]
            x+=[-0.1 + mymap(sorted(pList))]
            pass            

        if counts2 != 0:
            for c in counts2:
                pList= list((self.ptype(c.split()[11]), self.ptype(c.split()[12]), self.ptype(c.split()[13]), self.ptype(c.split()[14])))
                #print(mycounter, c, pList, counts[c])
                mycounter2+= 1

                y2+=[100*counts2[c]/events]
                ey2+=[100*counts2[c]**0.5/events]
                x2+=[0.1 + mymap(sorted(pList))]
                pass

        f = plt.figure(figsize=(9, 6))
        ax = f.add_subplot(1, 1, 1)
        plt.ylim((100*1e-4, 100*5.))
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]', size= 20)
        bar1 = plt.bar(x, y, color='#228b22', width=0.2, label=r"$g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
        if counts2!= 0:
            bar2 = plt.bar(x2, y2, color='#01B9FF', width=0.2, label=r"$g_{12}= 0$", hatch='\\') #,yerr=firstisf1_ey)
            leg2 = ax.legend([bar1, bar2],[r'$g_{12} = 1$',r'$g_{12} = 0$'], loc='upper right', frameon=False, prop={'size': 12.5}, bbox_to_anchor=(1.,0.8))

        ax.set_xticks([0, 1 , 2 , 3])
        #ax.set_xticklabels( (r"$f_{1}\rightarrow f'$", r"$f_{1}\rightarrow f'\phi$", r"$f_{1}\rightarrow f'\phi\phi$",r"$f_{1}\rightarrow f' f_{1} \bar{f}_{1}$",r"$f_{1}\rightarrow f' f_{2} \bar{f}_{2}$",r"$f_{1}\rightarrow f' f_{1/2} \bar{f}_{2/1}$") )
        plt.xlabel(r'Number of $\phi$', size=20)

        plt.legend(loc='upper right',prop={'size': 14})



        plt.text(-0.3, 55*4, r"3-step Full Quantum Simulation", fontsize=24)
        #plt.title(r"3-step Full Quantum Simulation", fontsize=24)
        plt.text(-0.3, 30*2.8, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=16)

        f.savefig("fullsim3step_states.pdf")
        plt.show()





















