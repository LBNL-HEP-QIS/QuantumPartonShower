import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import qiskit.providers.aer as qpa

from itertools import chain, combinations

from PaperPlots import MakeObservables as mo


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
        self._L = int(math.floor(math.log(N + ni, 2)) + 1)

        # Define these variables for indexing - to convert from cirq's grid qubits (see explaination in notebook)
        self._p_len = 3
        self._h_len = self._L
        self._w_h_len = self._L - 1
        self._e_len = 1
        self._w_len = 3
        self._na_len = self._L
        self._wa_len = self._L - 1

        #defining the registers
        self.pReg, self.hReg, self.w_hReg, self.eReg, self.wReg, self.n_aReg, self.w_aReg= self.allocateQubits()

        self._circuit = QuantumCircuit(self.wReg, self.pReg, self.hReg, self.eReg, self.n_aReg,
                                       self.w_hReg, self.w_aReg)


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


    def populateParameterLists(self, timeStepList, P_aList, P_bList, P_phiList, 
                               Delta_aList, Delta_bList, Delta_phiList, g_a, g_b, eps):
        """Populates the 6 lists with correct values for each time step theta"""
        for i in range(self._N):
            # Compute time steps
            t_up = eps ** ((i) / self._N)
            t_mid = eps ** ((i + 0.5) / self._N)
            t_low = eps ** ((i + 1) / self._N)
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


    def allocateQubits(self):
        nqubits_p = 3 * (self._N + self._ni)

        pReg = QuantumRegister(nqubits_p, 'p')

        hReg = QuantumRegister(self._L, 'h')
        #w_hReg = QuantumRegister(self._L, 'w_h')
        w_hReg = QuantumRegister(self._L - 1, 'w_h')

        eReg = QuantumRegister(1, 'e')
        wReg = QuantumRegister(3, 'w')  # we use all 5 of these work register qubits, but not sure why it is 5

        n_aReg = QuantumRegister(self._L, 'n_a')
        #w_aReg = QuantumRegister(self._L, 'w_a')
        w_aReg = QuantumRegister(self._L - 1, 'w_a')

        return (pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg)


    def initializeParticles(self, initialParticles):
        """ Apply appropriate X gates to ensure that the p register contains all of the initial particles.
            The p registers contains particles in the form of a string '[MSB, middle bit, LSB]' """
        for currentParticleIndex in range(len(initialParticles)):
            for particleBit in range(3):
                pBit= 2 - particleBit
                if int(initialParticles[currentParticleIndex][particleBit]) == 1:
                    self._circuit.x(self.pReg[currentParticleIndex * self._p_len + pBit])


    def flavorControl(self, flavor, control, target, ancilla, control_index, target_index, ancilla_index, h_bool=None):
        """Controlled x onto targetQubit if "control" particle is of the correct flavor"""
        if h_bool == None:
            if flavor == "phi":
                self._circuit.x(control[control_index + 1])
                self._circuit.x(control[control_index + 2])
                self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index])
                self._circuit.ccx(control[control_index + 2], ancilla[ancilla_index], target[target_index + 0])
                # undo work
                self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index])
                self._circuit.x(control[control_index + 1])
                self._circuit.x(control[control_index + 2])
            if flavor == "a":
                self._circuit.x(control[control_index + 0])
                self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])
                # undo work
                self._circuit.x(control[control_index + 0])
            if flavor == "b":
                self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])
        else:
            for h in h_bool:
                if flavor == "phi":
                    self._circuit.x(control[control_index + 1]).c_if(self.hReg_cl, h)
                    self._circuit.x(control[control_index + 2]).c_if(self.hReg_cl, h)
                    self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index]).c_if(self.hReg_cl, h)
                    self._circuit.ccx(control[control_index + 2], ancilla[ancilla_index], target[target_index + 0]).c_if(self.hReg_cl, h)
                    # undo work
                    self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index]).c_if(self.hReg_cl, h)
                    self._circuit.x(control[control_index + 1]).c_if(self.hReg_cl, h)
                    self._circuit.x(control[control_index + 2]).c_if(self.hReg_cl, h)
                if flavor == "a":
                    self._circuit.x(control[control_index + 0]).c_if(self.hReg_cl, h)
                    self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0]).c_if(self.hReg_cl, h)
                    # undo work
                    self._circuit.x(control[control_index + 0]).c_if(self.hReg_cl, h)
                if flavor == "b":
                    self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0]).c_if(self.hReg_cl, h)


    def plus1(self, l, countReg, workReg, control, ancilla, level):
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
            self._circuit.cx(control, countReg[0])
        if level < l - 1:
            # first level uses CNOT instead of TOFFOLI gate
            if level == 0:
                # move all X gates to first step to avoid unnecesarry gates
                [self._circuit.x(qubit) for qubit in countReg]
                self._circuit.ccx(countReg[0], control, workReg[0])
            else:
                self._circuit.ccx(countReg[level], workReg[level - 1], ancilla)
                self._circuit.ccx(ancilla, control, workReg[level])
                self._circuit.ccx(countReg[level], workReg[level - 1], ancilla)

            self._circuit.ccx(workReg[level], control, countReg[level + 1])
            # recursively call next layer
            self.plus1(l, countReg, workReg, control, ancilla, level + 1)
            # undo work qubits (exact opposite of first 7 lines - undoes calculation)
            if level == 0:
                self._circuit.ccx(countReg[0], control, workReg[0])
                [self._circuit.x(qubit) for qubit in countReg]
            else:
                self._circuit.ccx(countReg[level], workReg[level - 1], ancilla)
                self._circuit.ccx(ancilla, control, workReg[level])
                self._circuit.ccx(countReg[level], workReg[level - 1], ancilla)


    def uCount(self, m, l):
        """
        Populate the count registers using current particle states.
        Uses wReg[0] as the control and wReg[1] as ancilla qubit for flavorControl and plus1, respectively
        """
        for k in range(self._ni + m):
            # a fermions
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            self.plus1(l, self.n_aReg, self.w_aReg, self.wReg[0], self.wReg[1], 0)
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)


    def generateParticleCounts(self, m, k):
        """Fill countsList with all combinations of n_phi, n_a, and n_b where each n lies in range [0, n_i+m-k],
        and the sum of all n's lies in range [n_i-k, m+n_i-k], all inclusive
        """
        countsList = []
        for numParticles in range(self._ni - k, m + self._ni - k + 1):
            for numPhi in range(0, self._ni + m - k + 1):
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


    def numberControl(self, l, number, countReg, workReg, h_bool= None):
        """
        Applies an X to the l-2 (0 indexed) qubit of the work register if count register encodes the inputted number in binary
        returns this l-2 qubit, unless l=1, in which case return the only count register qubit
        DOES NOT CLEAN AFTER ITSELF - USE numberControlT to clean after this operation
        """
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number

        if h_bool == None:
            [self._circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]
        else:
            for h in h_bool:
                [self._circuit.x(countReg[i]).c_if(self.hReg_cl, h) for i in range(len(numberBinary)) if numberBinary[i] == 0]

        # first level does not use work qubits as control
        if l > 1:
            if h_bool == None: self._circuit.ccx(countReg[0], countReg[1], workReg[0])
            else: 
                for h in h_bool: self._circuit.ccx(countReg[0], countReg[1], workReg[0]).c_if(self.hReg_cl, h)

        # subfunction to recursively handle toffoli gates
        def binaryToffolis(level):
            if h_bool == None: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
            else: 
                for h in h_bool: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1]).c_if(self.hReg_cl, h)
            if level < l - 1:
                binaryToffolis(level + 1)

        if l > 2:
            binaryToffolis(2)

        # return qubit containing outcome of the operation
        if l == 1:
            return countReg[0]
        else:
            return workReg[l - 2]


    def numberControlT(self, l, number, countReg, workReg, h_bool= None):
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
                if h_bool == None: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
                else: 
                    for h in h_bool: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1]).c_if(self.hReg_cl, h)

        if l > 2:
            binaryToffolisT(2)

        if l > 1:
            if h_bool == None: self._circuit.ccx(countReg[0], countReg[1], workReg[0])
            else: 
                for h in h_bool: self._circuit.ccx(countReg[0], countReg[1], workReg[0]).c_if(self.hReg_cl, h)

        if h_bool == None:
            [self._circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]
        else:
            for h in h_bool: [self._circuit.x(countReg[i]).c_if(self.hReg_cl, h) for i in range(len(numberBinary)) if numberBinary[i] == 0]


##############################################################################################################################
# Utilities for classical controls                                                                                           #
##############################################################################################################################


    @staticmethod
    def history_to_counts(hList, niList):
        '''
        For a given history, encode as a list hList, return the total number of particles and the number of 
        bosons: n_tot, n_phi.

        List index 0 of hList is the first step, and subsequent steps proceed forwards through the list

        niList is a list of the initial particles:

            5, 7= fermion b
            4, 6= fermion a
            1= boson
            0= no particle
        '''
        fermList= [4, 5, 6, 7]
        ni= len(niList)
        n_tot= ni
        n_phi= 0
        particles= []

        # initialize
        for n in range(ni):
            particles+= [niList[n]]
            if niList[n] == 1:
                n_phi+= 1

        for j in range(len(hList)):
            h= hList[-j-1] # string
            hint= int(h, 2)
            if hint == 0: 
                particles+= [0]
            else:
                if particles[hint-1] in fermList:
                    particles+= [1]
                    n_phi+= 1
                elif particles[hint-1] == 1:
                    particles+= [4]
                    particles[hint-1]= 6
                    n_phi-= 1
                else:
                    print('Warning: emission from None particle. Result: None particle...')
                    particles+= [0]
                n_tot+= 1
        return n_tot, n_phi


    @staticmethod
    def generate_h(N, L, ni, l=[[]]):
        '''
        Generate and return a list of all possible emission histories for an N-step process, starting with 1 fermion.

        :param N:       (int)  number of steps
        :param L:       (int)  size of counting register
        :ni:      (list(str)) number of initial particles
        :param l:      (list) of previous histories, empty by default (start at step 0)
        '''
        if N == 0:
            return [[]]
        nh = len(l[0])
        l_new= []
        for h in l:
            for j in range(nh + ni + 1):
                binj= bin(j)[2:]

                if len(binj) < L:
                    binj= '0'*(L - len(binj)) + binj
                if j < ni + 1: l_new+= [[binj] + h]
                
                # index -1 corresponds to particle ni + 1, index -2 to particle ni + 2
                # For emission on particle j to be possible, particle j can't be zero. And particle j = ni + i
                # corresponds to index -i. ==> index= ni - j

                else:
                    if int(h[ni - j], 2) != 0: l_new+= [[binj] + h]

        if nh >= N - 1:
            return l_new
        else:
            return QuantumPartonShower.generate_h(N, L, ni, l_new)


    @staticmethod
    def h_map(N, L, niList):
        '''
        Return a map from tuples (n_tot, n_phi) to a list of all possible emission histories (list)
        that could produce that particlar number configuration.

        Note: reverses the order of emissions in the resulting values (lists)
        '''

        # Note: It's an exponentially long sequence.
        histories= QuantumPartonShower.generate_h(N, L, len(niList))

        counts= {}
        for history in histories:
            c= QuantumPartonShower.history_to_counts(history, niList)
            if c in counts:
                counts[c]+= [history]
            else:
                counts[c]= [history]

        return counts


##############################################################################################################################
#                                                                                                                            #
##############################################################################################################################


    def uE(self, l, m, Delta_phi, Delta_a, Delta_b, initialParticles):
        """Determine if emission occured in current step m"""
        hmap= QuantumPartonShower.h_map(m, self._L, [int(x, 2) for x in initialParticles])

        for key in hmap:
            n_tot, n_phi= key       

            if m > 0:
                h_bool= [int(''.join(hmap[key][0]), 2)]
                for j in range(1, len(hmap[key])):
                    h_bool+= [int(''.join(hmap[key][j]), 2)]
            else:
                h_bool= None

            for n_a in range(0, n_tot - n_phi + 1):
                n_b= n_tot - n_phi - n_a
                Delta = Delta_phi ** n_phi * Delta_a ** n_a * Delta_b ** n_b
                #print('na= %d, nb= %d, nphi= %d, emit angle: ' %(n_a, n_b, n_phi) + str((2 * math.acos(np.sqrt(Delta)))))
                aControlQub = self.numberControl(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)
                if m > 0: 
                    for h in h_bool: self._circuit.cry((2 * math.acos(np.sqrt(Delta))), aControlQub, self.eReg[0]).c_if(self.hReg_cl, h)
                else: self._circuit.cry((2 * math.acos(np.sqrt(Delta))), aControlQub, self.eReg[0])
                self.numberControlT(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)


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


    def twoLevelControlledRy(self, l, angle, k, externalControl, reg, workReg, h_bool= None):
        """
        Implements two level Ry rotation from state |0> to |k>, if externalControl qubit is on
        for reference: http://www.physics.udel.edu/~msafrono/650/Lecture%206.pdf
        """
        grayList = self.generateGrayList(l, k)
        #print('angle: ' + str(angle))
        if k == 0:
            return
        if l == 1 and k == 1:
            if h_bool == None: self._circuit.cry(angle, externalControl, reg[0])
            else: 
                for h in h_bool: self._circuit.cry(angle, externalControl, reg[0]).c_if(self.hReg_cl, h)
            return
        # swap states according to Gray Code until one step before the end
        for element in grayList:
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
            if element == grayList[-1]:  # reached end
                if h_bool == None:
                    self._circuit.ccx(controlQub, externalControl, workReg[l - 2])
                    self._circuit.cry(angle, workReg[l - 2], reg[targetQub])
                    self._circuit.ccx(controlQub, externalControl, workReg[l - 2])
                else:
                    for h in h_bool:
                        self._circuit.ccx(controlQub, externalControl, workReg[l - 2]).c_if(self.hReg_cl, h)
                        self._circuit.cry(angle, workReg[l - 2], reg[targetQub]).c_if(self.hReg_cl, h)
                        self._circuit.ccx(controlQub, externalControl, workReg[l - 2]).c_if(self.hReg_cl, h)
            else:  # swap states
                if h_bool == None: self._circuit.cx(controlQub, reg[targetQub])
                else: 
                    for h in h_bool: self._circuit.cx(controlQub, reg[targetQub]).c_if(self.hReg_cl, h)
            self.numberControlT(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
        # undo
        for element in self.reverse(grayList[:-1]):
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
            if h_bool == None: self._circuit.cx(controlQub, reg[targetQub])
            else: 
                for h in h_bool: self._circuit.cx(controlQub, reg[targetQub]).c_if(self.hReg_cl, h)
            self.numberControlT(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
        return


    def U_hAngle(self, flavor, n_phi, n_a, n_b, P_phi, P_a, P_b):
        """Determine angle of rotation used in U_h"""
        denominator = n_phi * P_phi + n_a * P_a + n_b * P_b
        if denominator == 0:  # occurs if we are trying the case of no particles remaining (n_a = n_b = n_phi = 0)
            return 0
        flavorStringToP = {'phi': P_phi, 'a': P_a, 'b': P_b}
        emissionAmplitude = np.sqrt(flavorStringToP[flavor] / denominator)
        #print(emissionAmplitude)
        # correct for arcsin input greater than 1 errors for various input combinations that are irrelevant anyway
        emissionAmplitude = min(1, emissionAmplitude)

        return 2 * np.arcsin(emissionAmplitude)


    def minus1(self, l, countReg, workReg, control, ancilla, level):
        """
        Recursively carries an subtraction of 1 to the LSB of a register to all bits if control == 1
        Equivalent to plus1 but with an X applied to all count qubits before and after gate
        """
        [self._circuit.x(qubit) for qubit in countReg]
        self.plus1(l, countReg, workReg, control, ancilla, level)
        [self._circuit.x(qubit) for qubit in countReg]


    def U_h(self, l, m, P_phi, P_a, P_b, initialParticles):
        """Implement U_h from paper"""
        #print(P_phi, P_a, P_b)
        for k in reversed(range(self._ni + m)):
            if k >= self._ni:
                hmap= QuantumPartonShower.h_map(k - self._ni + 1, self._L, [int(x, 2) for x in initialParticles])
            else:
                hmap= QuantumPartonShower.h_map(0, self._L, [int(x, 2) for x in initialParticles[:k+1]])

            for key in hmap:
                n_tot, n_phi= key
                if m == 0 or k < self._ni:
                    h_bool= None
                else:
                    h_bool= [int(''.join(hmap[key][0]), 2)]
                    for j in range(1, len(hmap[key])):
                        h_bool+= [int(''.join(hmap[key][j]), 2)]

                for n_a in range(0, n_tot - n_phi + 1):
                    #print('\n')
                    n_b= n_tot - n_phi - n_a
                    aControl = self.numberControl(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)
                    # controlled R-y from |0> to |k> on all qubits with all possible angles depending on n_phi, n_a, n_b, and flavor
                    for flavor in ['phi', 'a', 'b']:
                        angle = self.U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b)
                        #print('na= %d, nb= %d, nphi= %d, flavor: ' %(n_a, n_b, n_phi) + flavor + ', angle= ' + str(angle))
                        self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 2, h_bool= h_bool)  # wReg[2] is work qubit but is reset to 0
                        if m == 0 or k < self._ni:
                            self._circuit.ccx(aControl, self.wReg[0], self.wReg[1])
                            self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2])
                        else:
                            for h in h_bool:
                                self._circuit.ccx(aControl, self.wReg[0], self.wReg[1]).c_if(self.hReg_cl, h)
                                self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2]).c_if(self.hReg_cl, h)
                        self.twoLevelControlledRy(l, angle, k + 1, self.wReg[2], self.hReg, self.w_hReg, h_bool= h_bool)

                        if m == 0 or k < self._ni:
                            self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2])  # next steps undo work qubits
                            self._circuit.ccx(aControl, self.wReg[0], self.wReg[1])
                        else:
                            for h in h_bool:
                                self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2]).c_if(self.hReg_cl, h)  # next steps undo work qubits
                                self._circuit.ccx(aControl, self.wReg[0], self.wReg[1]).c_if(self.hReg_cl, h)

                        self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 2, h_bool= h_bool)  # wReg[2] is work qubit but is reset to 0
                    self.numberControlT(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)

            # subtract from the counts register depending on which flavor particle emitted
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)  # wReg[2] is work qubit but is reset to 0
            self.minus1(l, self.n_aReg, self.w_aReg, self.wReg[0], self.wReg[1], 0)
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)  # wReg[2] is work qubit but is reset to 0

        # apply x on eReg if hReg[m] = 0, apply another x so we essentially control on not 0 instead of 0
        isZeroControl = self.numberControl(l, 0, self.hReg, self.w_hReg)
        self._circuit.cx(isZeroControl, self.eReg[0])
        self._circuit.x(self.eReg[0])
        self.numberControlT(l, 0, self.hReg, self.w_hReg)


    def updateParticles(self, l, m, k, g_a, g_b):
        """Updates particle if controlQub is on

            hReg_cl --> Classical register upon which hReg was measured.
        """
        oldParticleReg = self.pReg
        newParticleReg = self.pReg

        pk0= k * self._p_len # particle k first (zero) index
        pNew= (self._ni + m) * self._p_len # new/current particle first(zero) index

        # Have to get all histories where the current emission is from particle k.
        gen_h= QuantumPartonShower.generate_h(m+1, self._L, self._ni) # current particle to update is actually step m+1
        gen_hk= []
        for h in gen_h:
            if int(h[0], 2) == k + 1: # k + 1 is the correct particle
                gen_hk+= [h]

        for hList in gen_hk:
            h= int(''.join(hList), 2)
            #print(h)
            self._circuit.cx(oldParticleReg[pk0 + 2], newParticleReg[pNew + 0]).c_if(self.hReg_cl, h)

            # second gate in paper (undoes work register immediately)
            self._circuit.x(oldParticleReg[pk0 + 1]).c_if(self.hReg_cl, h)
            self._circuit.x(oldParticleReg[pk0 + 2]).c_if(self.hReg_cl, h)
            self._circuit.cx(oldParticleReg[pk0 + 2], self.wReg[0]).c_if(self.hReg_cl, h)
            self._circuit.ccx(self.wReg[0], oldParticleReg[pk0 + 1], self.wReg[1])
            self._circuit.ccx(self.wReg[1], oldParticleReg[pk0 + 0], newParticleReg[pNew + 2])
            self._circuit.ccx(self.wReg[0], oldParticleReg[pk0 + 1], self.wReg[1])
            self._circuit.cx(oldParticleReg[pk0 + 2], self.wReg[0]).c_if(self.hReg_cl, h)
            self._circuit.x(oldParticleReg[pk0 + 1]).c_if(self.hReg_cl, h)
            self._circuit.x(oldParticleReg[pk0 + 2]).c_if(self.hReg_cl, h)

            # third gate in paper
            self._circuit.cx(newParticleReg[pNew + 2], oldParticleReg[pk0 + 2]).c_if(self.hReg_cl, h)

            # fourth and fifth gate in paper (then undoes work register)
            self._circuit.ch(newParticleReg[pNew + 2], newParticleReg[pNew + 1]).c_if(self.hReg_cl, h)
            angle = (2 * np.arccos(g_a / np.sqrt(g_a ** 2 + g_b ** 2)))
            self._circuit.cry(angle, newParticleReg[pNew + 2], newParticleReg[pNew + 0]).c_if(self.hReg_cl, h)

            # sixth and seventh gate in paper (then undoes work register)
            self._circuit.x(newParticleReg[pNew + 0]).c_if(self.hReg_cl, h)
            self._circuit.x(newParticleReg[pNew + 1]).c_if(self.hReg_cl, h)
            self._circuit.cx(newParticleReg[pNew + 1], self.wReg[0]).c_if(self.hReg_cl, h)
            self._circuit.ccx(self.wReg[0], newParticleReg[pNew + 2], oldParticleReg[pk0 + 1])
            self._circuit.cx(newParticleReg[pNew + 1], self.wReg[0]).c_if(self.hReg_cl, h)
            self._circuit.cx(newParticleReg[pNew + 0], self.wReg[0]).c_if(self.hReg_cl, h)
            self._circuit.ccx(self.wReg[0], newParticleReg[pNew + 2], oldParticleReg[pk0 + 0])
            self._circuit.cx(newParticleReg[pNew + 0], self.wReg[0]).c_if(self.hReg_cl, h)
            self._circuit.x(newParticleReg[pNew + 0]).c_if(self.hReg_cl, h)
            self._circuit.x(newParticleReg[pNew + 1]).c_if(self.hReg_cl, h)


    def U_p(self, l, m, g_a, g_b):
        """Applies U_p from paper"""
        for k in range(0, self._ni + m):
            self.updateParticles(l, m, k, g_a, g_b)


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

        # evaluate P(Theta) and Delta(Theta) at every time step
        timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList = [], [], [], [], [], [], []
        self.populateParameterLists(timeStepList, P_aList, P_bList, P_phiList, 
                                    Delta_aList, Delta_bList, Delta_phiList, g_a, g_b, eps)

        qubits = {'pReg': self.pReg, 'hReg': self.hReg, 'w_hReg': self.w_hReg, 'eReg': self.eReg, 'wReg': self.wReg,
                  'n_aReg': self.n_aReg, 'w_aReg': self.w_aReg}

        self.initializeParticles(initialParticles)

        (self.wReg_cl, self.pReg_cl, self.hReg_cl, self.eReg_cl, self.n_aReg_cl, self.w_hReg_cl, self.w_aReg_cl) = self.allocateClbits()
        
        self.add_Clbits()

        # begin stepping through subcircuits
        for m in range(self._N):
            print('\n\nm= %d\n\n' %(m))
            l = int(math.floor(math.log(m + self._ni, 2)) + 1)

            # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
            index = 0
            while index < self.pReg.size:
                self._circuit.cry((2 * math.asin(-u)), self.pReg[index + 2], self.pReg[index + 0])
                index += self._p_len

            # populate count register (step 2)
            print('Apply uCount()...')
            self.uCount(m, l)

            # assess if emmision occured (step 3)
            print('Apply uE()...')
            self.uE(l, m, Delta_phiList[m], Delta_aList[m], Delta_bList[m], initialParticles)

            print('Apply U_h()...')
            self.U_h(l, m, P_phiList[m], P_aList[m], P_bList[m], initialParticles)

            print('Measure and reset |h>...')
            self._circuit.measure(self.hReg, self.hReg_cl[m*self._L : (m+1)*self._L])
            self._circuit.reset(self.hReg)

            print('Apply U_p()...')
            # update particle based on which particle split/emmitted (step 5)
            self.U_p(l, m, g_a, g_b)

            # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
            index2 = 0
            while index2 < self.pReg.size:
                # circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
                self._circuit.cry((2 * math.asin(u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
                index2 += self._p_len

        print('generated circuit on', self.flatten(list(qubits.values())), 'qubits')

        
        return self._circuit, qubits


    def allocateClbits(self):
        nbits_h = self._N * self._L

        wReg_cl = ClassicalRegister(3, 'w_cl')
        pReg_cl = []
        for j in range(self._N + self._ni):
            pReg_cl.append(ClassicalRegister(3, 'p%d_cl' %(j)))
        hReg_cl = ClassicalRegister(nbits_h, 'h_cl')
        eReg_cl = ClassicalRegister(self._e_len, 'e_cl')
        n_aReg_cl = ClassicalRegister(self._L, 'na_cl')

        #w_hReg_cl = ClassicalRegister(self._L, 'wh_cl')
        #w_aReg_cl = ClassicalRegister(self._L, 'wa_cl')
        w_hReg_cl = ClassicalRegister(self._L - 1, 'wh_cl')
        w_aReg_cl = ClassicalRegister(self._L - 1, 'wa_cl')

        return (wReg_cl, pReg_cl, hReg_cl, eReg_cl, n_aReg_cl, w_hReg_cl, w_aReg_cl)


    def add_Clbits(self):
        # Add all classical registers stored in self to self._circuit
        self._circuit.add_register(self.wReg_cl)
        for j in range(self._N + self._ni):
            self._circuit.add_register(self.pReg_cl[j])
        self._circuit.add_register(self.hReg_cl)
        self._circuit.add_register(self.eReg_cl)
        self._circuit.add_register(self.n_aReg_cl)
        self._circuit.add_register(self.w_hReg_cl)
        self._circuit.add_register(self.w_aReg_cl)


    def measure_Clbits(self):
        # Measures all bits other than the history register (already measured)
        self._circuit.measure(self.wReg, self.wReg_cl)
        for j in range(self._N + self._ni):
            self._circuit.measure(self.pReg[3*j : 3*(j+1)], self.pReg_cl[j])

        self._circuit.measure(self.eReg, self.eReg_cl)
        self._circuit.measure(self.n_aReg, self.n_aReg_cl)
        self._circuit.measure(self.w_hReg, self.w_hReg_cl)
        self._circuit.measure(self.w_aReg, self.w_aReg_cl)


    def simulate(self, type, shots=None, position=False):
        """
        :param type: either the qasm simulaot or the statevector simulator
        :param shots: if using the qasm simulator the number of shots needs to be specified
        :param position: the statevector is very long, so if position=True the function will print the value and
        position of tbe non-zero elements
        :return: either counts (qasm) or the statevector
        """
        if type == 'qasm':
            #simulator = Aer.get_backend('qasm_simulator')
            #simulator = Aer.get_backend('aer_simulator_matrix_product_state')
            simulator = qpa.QasmSimulator(method= 'matrix_product_state')

            self.measure_Clbits()
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


    def bar_plot2(self, counts, events, eps, g1, g2, counts2= None):
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
            print(mycounter, c, self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7]), counts[c])
            mycounter+=1

            if c.split()[4]=='0':
                if c.split()[3]=='01' or c.split()[3]=='10' or c.split()[3]=='11':
                    firstemission[0]+= counts[c]/events
                else:
                    firstemission[1]+= counts[c]/events
            else:
                if c.split()[3]=='01' or c.split()[3]=='10' or c.split()[3]=='11':
                    firstemission[2]+= counts[c]/events
                else:
                    firstemission[3]+= counts[c]/events

            if (self.ptype(c.split()[7])=='f1'):
                firstisf1_y+=[100*counts[c]/events]
                firstisf1_ey+=[100*counts[c]**0.5/events]
                firstisf1_x+=[-0.2+mymap[self.ptype(c.split()[5]), self.ptype(c.split()[6])]]
                pass
            if (self.ptype(c.split()[7])=='f2'):
                firstisf2_y+=[100*counts[c]/events]
                firstisf2_ey+=[100*counts[c]**0.5/events]
                firstisf2_x+=[0.0+mymap[self.ptype(c.split()[5]), self.ptype(c.split()[6])]]
                pass

        if counts2 != None:
            for c in counts2:

                if c.split()[4]=='0':
                    if c.split()[3]=='01' or c.split()[3]=='10' or c.split()[3]=='11':
                        firstemission2[0]+=counts2[c]/events
                    else:
                        firstemission2[1]+=counts2[c]/events
                else:
                    if c.split()[3]=='01' or c.split()[3]=='10' or c.split()[3]=='11':
                        firstemission2[2]+=counts2[c]/events
                    else:
                        firstemission2[3]+=counts2[c]/events

                if (self.ptype(c.split()[7])=='f1'):
                    firstisf1b_y+=[100*counts2[c]/events]
                    firstisf1b_ey+=[100*counts2[c]**0.5/events]
                    firstisf1b_x+=[0.2+mymap[self.ptype(c.split()[5]),self.ptype(c.split()[6])]]
                    pass
                if (self.ptype(c.split()[7])=='f2'):
                    firstisf2b_y+=[100*counts2[c]/events]
                    firstisf2b_ey+=[100*counts2[c]**0.5/events]
                    firstisf2b_x+=[0.2+mymap[self.ptype(c.split()[5]),self.ptype(c.split()[6])]]
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


        f = plt.figure(figsize=(10, 8))
        ax = f.add_subplot(1, 1, 1)
        plt.ylim((100*1e-4, 100*5.))
        plt.xlim((0, 6.2))
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]', size=20)
        #bar1 = plt.bar(firstisf1_x, firstisf1_y, color='#228b22', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
        #bar1b = plt.bar(firstisf2_x, firstisf2_y, color='#01B9FF', width=0.2, label=r"$f' = f_{2}, g_{12}= 1$", hatch='//') #,yerr=firstisf2_ey)
        #n, bins, patches= plt.hist(firstisf1_x, weights=firstisf1_y, bins=np.arange(0.3, 6.31, 1), align= 'mid', color='#228b22', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
        n, bins, patches= plt.hist(firstisf1_x, weights=firstisf1_y, bins=6, range=[0.3, 6.3], align= 'mid', color='#228b22', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
        plt.hist(firstisf2_x, weights=firstisf2_y, bins=6, range=[0.5, 6.5], align= 'mid', color='#01B9FF', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='//')
        print(n, bins, patches)
        ax.set_xticks(np.arange(0.6, 6.6, 1))
        ax.set_xticklabels( (r"$f_{1}\rightarrow f'$", r"$f_{1}\rightarrow f'\phi$", r"$f_{1}\rightarrow f'\phi\phi$",r"$f_{1}\rightarrow f' f_{1} \bar{f}_{1}$",r"$f_{1}\rightarrow f' f_{2} \bar{f}_{2}$",r"$f_{1}\rightarrow f' f_{1/2} \bar{f}_{2/1}$") )


        if counts2 != None:
            #bar2 = plt.bar(firstisf1b_x, firstisf1b_y, color='#FF4949', width=0.2, label=r"$f' = f_{1}, g_{12}= 0$", alpha=1.0) #,hatch="//")
            plt.hist(firstisf1b_x, weights=firstisf1b_y, bins=6, range=[0.7, 6.7], align= 'mid', color='#FF4949', width=0.2, label=r"$f' = f_{1}, g_{12}= 0$", alpha=1.0) #,hatch="//")
            #plt.bar(firstisf2b_x,firstisf2b_y,color='blue',width=0.4,label=r"$f' = f_{2}$",yerr=firstisf2b_ey,alpha=0.5)
            #leg2 = ax.legend([bar1, bar1b, bar2],[r"$f' = f_{1}, g_{12} = 1$",r"$f' = f_{2}, g_{12} = 1$",r"$f' = f_{1}, g_{12} = 0$"], loc='upper right', prop={'size': 12.5}, frameon=False)
            pass
        else:
            #leg2 = ax.legend([bar1,bar1b], [r'$f = f_{a}$',r'$f = f_{b}$'], loc='upper right',frameon=False,prop={'size': 12.5},bbox_to_anchor=(1.,0.8))
            pass
        #ax.add_artist(leg2);

        plt.legend(loc='upper right', prop={'size': 14})
        #plt.text(0.7, 55*3, r"2-step Full Quantum Simulation", fontsize=14)
        plt.title(r"2-step Full Quantum Simulation", fontsize=24)
        plt.text(0.2, 220, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=16)

        #f.savefig("sim2step_states_shots=%d.pdf" %(events))
        plt.show()
        print(firstisf1_y)
        print(firstisf1_x)

        print(firstisf2_y)
        print(firstisf2_x)

        print(firstisf1b_y)
        print(firstisf1b_x)

        print(sum(firstisf1_y))
        print(sum(firstisf2_y))
        print(sum(firstisf1b_y))
        print('First emission: ' + str(firstemission))


    def bar_plot_history(self, counts, events, eps, g1, g2, N, counts2= None):
        # Plots counts vs. emission history for an N-step process. Maybe not a good idea because there are factorially many histories
        pass


    def bar_plot_numbers(self, counts, events, eps, g1, g2, N, ni, counts2= None):
        # Plots counts vs. number configuration

        # Can handle two different couplings, counts and counts2

        # For now assume initially start with only fermions

        mycounter = 0

        plot1_y = []
        plot1_ey = []
        plot1_x = []

        if counts2 != None:
            plot2_y = []
            plot2_ey = []
            plot2_x = []

        # Generate possible configurations
        mymap = {}

        i= 1
        for nf1 in range(0, N+ni+1):
            for nf2 in range(0, N+ni+1-nf1):
                for nphi in range(0, N+ni+1-nf1-nf2):
                    mymap[(nf1, nf2, nphi)]= i
                    i+= 1

        print(mymap)
        for c in counts:
            printList= [mycounter, c]
            for n in range(N+ni):
                printList.append(self.ptype(c.split()[5+n]))
            print(tuple(printList))
            mycounter+=1

            nf1= 0
            nf2= 0
            nphi= 0

            for n in range(N+ni):
                p= self.ptype(c.split()[5+n])
                if p == 'f1' or p == 'af1':
                    nf1+= 1
                if p == 'f2' or p == 'af2':
                    nf2+= 1
                if p == 'phi':
                    nphi+= 1


            plot1_y+=[100*counts[c]/events]
            plot1_ey+=[100*counts[c]**0.5/events]
            plot1_x+=[-0.15+mymap[(nf1, nf2, nphi)]]
            

        if counts2 != None:    
            for c in counts2:
                printList= [mycounter, c]
                for n in range(N+ni):
                    printList.append(self.ptype(c.split()[5+n]))
                print(tuple(printList))
                mycounter+=1

                nf1= 0
                nf2= 0
                nphi= 0

                for n in range(N+ni):
                    p= self.ptype(c.split()[5+n])
                    if p == 'f1' or p == 'af1':
                        nf1+= 1
                    if p == 'f2' or p == 'af2':
                        nf2+= 1
                    if p == 'phi':
                        nphi+= 1


                plot2_y+=[100*counts2[c]/events]
                plot2_ey+=[100*counts2[c]**0.5/events]
                plot2_x+=[0.15+mymap[(nf1, nf2, nphi)]]


        f = plt.figure(figsize=(10, 8))
        ax = f.add_subplot(1, 1, 1)
        plt.ylim((100*1e-4, 100*5.))
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]', fontsize=24)
        bar1 = plt.bar(plot1_x, plot1_y, color='#228b22', width=0.3, label=r"$f' = f_{1}$", hatch='\\') #,yerr=firstisf1_ey)

        plt.xticks(np.arange(1, len(mymap.keys())+1, 1), rotation=45)
        ticklabels= []
        for key in mymap.keys():
            ticklabels+= [str(key)]

        ax.set_xticklabels( tuple(ticklabels) )

        plt.legend(loc='upper right',prop={'size': 9.5})

        if counts2 != None:
            bar2 = plt.bar(plot2_x, plot2_y, color='#FF4949', width=0.3, label=r"$f' = f_{1}$", alpha=1.0) #,hatch="//")
            #plt.bar(firstisf2b_x,firstisf2b_y,color='blue',width=0.4,label=r"$f' = f_{2}$",yerr=firstisf2b_ey,alpha=0.5)
            leg2 = ax.legend([bar1, bar2],[r"$f' = f_{1}, g_{12} = 1$",r"$f' = f_{2}, g_{12} = 1$",r"$f' = f_{1}, g_{12} = 0$"], loc='upper right', prop={'size': 12.5}, frameon=False)
        else:
            leg2 = ax.legend([bar1], [r'$f = f_{a}$',r'$f = f_{b}$'], loc='upper right',frameon=False,prop={'size': 12.5},bbox_to_anchor=(1.,0.8))

        ax.add_artist(leg2);

        plt.title(r"%d-step Full Quantum Simulation" %(N), fontsize=24)
        plt.text(0.1, 0.9, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=18, transform=ax.transAxes)

        plt.xlabel(r'$(n_{f_1}, n_{f_2}, n_{\phi})$', fontsize=24)
        #f.savefig("fullsim2step_states.pdf")
        plt.show()



    def bar_plot_emissions(self, counts, events, eps, g1, g2, N, counts2= None):
        # Plots counts vs. number of emissions

        # Can handle two different counts: counts and counts2
        
        emissions_x= []
        emissions_ey= []
        emissions_y= []

        offset= 0.2

        for c in counts:
            emit_list= []
            for n in range(self._N + self._ni):
                emit_list= [self.ptype(c.split()[-1-self._ni-n])] + emit_list

            #print(c, self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7]), counts[c])

            emit_N= mo.Nemissions(emit_list, n_I= self._ni)

            emissions_y+= [100*counts[c]/events]
            emissions_ey+= [100*counts[c]**0.5/events]
            emissions_x+= [-offset + emit_N]

        
        if counts2 != None:
        
            emissions2_x= []
            emissions2_ey= []
            emissions2_y= []
        
            for c in counts2:
                emit_list= []
                for n in range(self._N + self._ni):
                    emit_list= [self.ptype(c.split()[-1-self._ni-n])] + emit_list

                #print(c, self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7]), counts2[c])
                #print(emit_list)

                emit_N= mo.Nemissions(emit_list, n_I= self._ni)
                
                emissions2_y+= [100*counts2[c]/events]
                emissions2_ey+= [100*counts2[c]**0.5/events]
                emissions2_x+= [offset + emit_N]
    
            print(sum(emissions2_y))
        print(sum(emissions_y))
        print(emissions_y)
        print(emissions_x)

        f = plt.figure(figsize=(10, 8))
        ax = f.add_subplot(1, 1, 1)
        #plt.ylim((0, 50))
        plt.xlim((-3*offset, N+1- 2*offset))
        #ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]', fontsize=24)
        bar1 = plt.hist(emissions_x, weights=emissions_y, bins= np.arange(0 - 2*offset, N+2 - 2*offset, 1), align= 'mid', color='#228b22', width=2*offset, label=r"$g_{12} = 1$") #,yerr=firstisf1_ey)

        plt.xticks(np.arange(0, self._N + 1), size= 18)

        if counts2 != None:
            bar2 = plt.hist(emissions2_x, weights=emissions2_y, bins= np.arange(0, N+2 - 2*offset, 1), align= 'mid', color='#FF4949', width=2*offset, label=r"$g_{12} = 0$", alpha=1.0) #,hatch="//")
            #plt.bar(firstisf2b_x,firstisf2b_y,color='blue',width=0.4,label=r"$f' = f_{2}$",yerr=firstisf2b_ey,alpha=0.5)
            #leg2 = ax.legend([bar1, bar2],[r"$g_{12} = 1$",r"$g_{12} = 0$"], loc='upper right', prop={'size': 16}, frameon=False)
        else:
            #leg2 = ax.legend([bar1], [r"$g_{12} = 1$"], loc='upper right',frameon=False,prop={'size': 16},bbox_to_anchor=(1.,0.8))
            pass
        #ax.add_artist(leg2);

        plt.legend(loc='upper left', prop={'size': 16})
        plt.title(r"%d-step Full Quantum Simulation" %(self._N), fontsize=24)
        plt.text(0.1, 0.7, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=18, transform=ax.transAxes)

        plt.xlabel('Number of emissions', fontsize=24)
        f.savefig("sim%dstep_emissions_shots=%d.pdf" %(N, events))
        plt.show()









    def bar_plot_thetamax(self, counts, events, eps, g1, g2, N, counts2= None):
        # Plots counts vs. number of emissions

        # Can handle two different counts: counts and counts2
        
        tm_x= [] # tm= theta max
        tm_ey= []
        tm_y= []


        for c in counts:
            emit_list= []
            for n in range(0, self._N + self._ni):
                emit_list= [self.ptype(c.split()[-1-self._ni-n])] + emit_list

            #print(c, self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7]), counts[c])

            theta_max, centers, hist_bins= mo.LogThetaMax(emit_list, n_I= self._ni, eps= eps)

            offset= (hist_bins[1] - hist_bins[0]) / 8

            tm_y+= [100*counts[c]/events]
            tm_ey+= [100*counts[c]**0.5/events]
            tm_x+= [-offset + theta_max]

        
        if counts2 != None:
        
            tm2_x= []
            tm2_ey= []
            tm2_y= []
        
            for c in counts2:
                emit_list= []
                for n in range(0, self._N + self._ni):
                    emit_list= [self.ptype(c.split()[-1-self._ni-n])] + emit_list

                theta_max, centers, hist_bins= mo.LogThetaMax(emit_list, n_I= self._ni, eps= eps)
                
                offset= (hist_bins[1] - hist_bins[0]) / 8

                tm2_y+= [100*counts2[c]/events]
                tm2_ey+= [100*counts2[c]**0.5/events]
                tm2_x+= [offset + theta_max]
    
        #    print(sum(tm2_y))
        #print(sum(tm_y))
        #print(tm_y)
        print(tm_x)
        print(offset)
    
        f = plt.figure(figsize=(10, 8))
        ax = f.add_subplot(1, 1, 1)
        #plt.ylim((0, 50))
        plt.xlim((hist_bins[0]-offset, hist_bins[-1]+offset))
        #ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]', fontsize=24)
        #bar1 = plt.hist(tm_x, weights=tm_y, bins= [centers[0] - offset] + list(centers - offset) + [centers[-1] + offset], align= 'mid', color='#228b22', width=2*offset, label=r"$g_{12} = 1$") #,yerr=firstisf1_ey)
        bar1 = plt.hist(tm_x, weights=tm_y, bins= hist_bins, align= 'mid', width= 2*offset, color='#228b22', label=r"$g_{12} = 1$") #,yerr=firstisf1_ey)

        #plt.xticks(np.arange(0, self._N + 1), size= 18)

        if counts2 != None:
            #bar2 = plt.hist(tm2_x, weights=tm2_y, bins= [centers[0] + offset] + list(centers + offset) + [centers[-1] + offset], align= 'mid', color='#FF4949', width=2*offset, label=r"$g_{12} = 0$", alpha=1.0) #,hatch="//")
            bar2 = plt.hist(tm2_x, weights=tm2_y, bins= hist_bins, align= 'mid', width= 2*offset, color='#FF4949', label=r"$g_{12} = 0$", alpha=1.0) #,hatch="//")
            #plt.bar(firstisf2b_x,firstisf2b_y,color='blue',width=0.4,label=r"$f' = f_{2}$",yerr=firstisf2b_ey,alpha=0.5)
            #leg2 = ax.legend([bar1, bar2],[r"$g_{12} = 1$",r"$g_{12} = 0$"], loc='upper right', prop={'size': 16}, frameon=False)
        else:
            #leg2 = ax.legend([bar1], [r"$g_{12} = 1$"], loc='upper right',frameon=False,prop={'size': 16},bbox_to_anchor=(1.,0.8))
            pass
        #ax.add_artist(leg2);

        plt.legend(loc='upper left', prop={'size': 16})
        plt.title(r"%d-step Full Quantum Simulation" %(self._N), fontsize=24)
        plt.text(0.1, 0.7, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=18, transform=ax.transAxes)

        plt.xlabel(r'$\log(\theta_{\max})$', fontsize=24)
        f.savefig("sim%dstep_θmax_shots=%d.pdf" %(N, events))
        plt.show()











    def bar_plot3(self, counts, events, eps, g1, g2, initialParticles, counts2= None):

        mycounter = 0
        mycounter2 = 0

        y_f1 = []
        ey_f1 = []
        x_f1 = []

        y_f2 = []
        ey_f2 = []
        x_f2 = []

        if counts2 != None:
            y2_f1 = []
            ey2_f1 = []
            x2_f1 = []

            y2_f2 = []
            ey2_f2 = []
            x2_f2 = []

            offset= 1.2
        else: offset= 2.4

        print('offset= ' + str(offset))
        mymap = {}
        # 0
        mymap['0', '0', '0']= 1

        # phi
        mymap['phi', '0', '0']= 2
        mymap['0', 'phi', '0']= 2
        mymap['0', '0', 'phi']= 2

        # phi phi
        mymap['phi', 'phi', '0']= 3
        mymap['phi', '0', 'phi']= 3
        mymap['0', 'phi', 'phi']= 3
        
        # phi phi phi
        mymap['phi', 'phi', 'phi']= 4

        # f1 af1
        mymap['af1', 'f1', '0']= 5
        mymap['f1', 'af1', '0']= 5
        mymap['af1', '0', 'f1']= 5
        mymap['f1', '0', 'af1']= 5
        mymap['0', 'af1', 'f1']= 5
        mymap['0', 'f1', 'af1']= 5

        # f2 af2
        mymap['af2', 'f2', '0']= 6
        mymap['f2', 'af2', '0']= 6
        mymap['af2', '0', 'f2']= 6
        mymap['f2', '0', 'af2']= 6
        mymap['0', 'af2', 'f2']= 6
        mymap['0', 'f2', 'af2']= 6

        # f1 af2 / f2 af1
        mymap['af2', 'f1', '0']= 7
        mymap['f2', 'af1', '0']= 7
        mymap['af1', 'f2', '0']= 7
        mymap['f1', 'af2', '0']= 7

        mymap['af2', '0', 'f1']= 7
        mymap['f2', '0', 'af1']= 7
        mymap['af1', '0', 'f2']= 7
        mymap['f1', '0', 'af2']= 7

        mymap['0', 'af2', 'f1']= 7
        mymap['0', 'f2', 'af1']= 7
        mymap['0', 'af1', 'f2']= 7
        mymap['0', 'f1', 'af2']= 7

        # f1 af1 phi
        mymap['af1', 'f1', 'phi']= 8
        mymap['f1', 'af1', 'phi']= 8
        mymap['af1', 'phi', 'f1']= 8
        mymap['f1', 'phi', 'af1']= 8
        mymap['phi', 'af1', 'f1']= 8
        mymap['phi', 'f1', 'af1']= 8

        # f2 af2 phi
        mymap['af2', 'f2', 'phi']= 9
        mymap['f2', 'af2', 'phi']= 9
        mymap['af2', 'phi', 'f2']= 9
        mymap['f2', 'phi', 'af2']= 9
        mymap['phi', 'af2', 'f2']= 9
        mymap['phi', 'f2', 'af2']= 9

        # (f1 af2 / f2 af1) phi
        mymap['af2', 'f1', 'phi']= 10
        mymap['f2', 'af1', 'phi']= 10
        mymap['af1', 'f2', 'phi']= 10
        mymap['f1', 'af2', 'phi']= 10

        mymap['af2', 'phi', 'f1']= 10
        mymap['f2', 'phi', 'af1']= 10
        mymap['af1', 'phi', 'f2']= 10
        mymap['f1', 'phi', 'af2']= 10

        mymap['phi', 'af2', 'f1']= 10
        mymap['phi', 'f2', 'af1']= 10
        mymap['phi', 'af1', 'f2']= 10
        mymap['phi', 'f1', 'af2']= 10


        for c in counts:
            pList= list((self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7]), self.ptype(c.split()[8])))
            #print(mycounter, c, pList, counts[c])
            if (self.ptype(c.split()[-2])=='f1'):
                y_f1+= [100*counts[c]/events]
                ey_f1+= [100*counts[c]**0.5/events]
                x_f1+= [-1.5*offset + 6*mymap[self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7])]]
                pass
            if (self.ptype(c.split()[-2])=='f2'):
                y_f2+= [100*counts[c]/events]
                ey_f2+= [100*counts[c]**0.5/events]
                x_f2+= [-0.5*offset + 6*mymap[self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7])]]
                pass
            pass            

        if counts2 != 0:
            for c in counts2:
                pList= list((self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7]), self.ptype(c.split()[8])))
                #print(mycounter, c, pList, counts[c])
                if (self.ptype(c.split()[-2])=='f1'):
                    y2_f1+= [100*counts2[c]/events]
                    ey2_f1+= [100*counts2[c]**0.5/events]
                    x2_f1+= [0.5*offset + 6*mymap[self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7])]]
                    pass
                if (self.ptype(c.split()[-2])=='f2'):
                    y2_f2+= [100*counts2[c]/events]
                    ey2_f2+= [100*counts2[c]**0.5/events]
                    x2_f2+= [1.5*offset + 6*mymap[self.ptype(c.split()[5]), self.ptype(c.split()[6]), self.ptype(c.split()[7])]]
                    pass
                pass


        f = plt.figure(figsize=(14, 10))
        ax = f.add_subplot(1, 1, 1)
        plt.ylim((100*1e-4, 100*5.))
        plt.xlim((0, 66))
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylabel('Probability [%]', size= 24)
        #bar1_f1 = plt.bar(x_f1, y_f1, color='#228b22', width=offset, label=r"$g_{12}= 1, f'= f_{1}$") #,yerr=firstisf1_ey)
        #bar1_f2 = plt.bar(x_f2, y_f2, color='#9AEE9A', width=offset, label=r"$g_{12}= 1, f'= f_{2}$") #,yerr=firstisf1_ey)

        plt.hist(x_f1, weights=y_f1, bins= np.arange(6-1.5*offset, 72-1.5*offset, 6), align= 'mid', color='#228b22', width=offset, label=r"$f' = f_{1}, g_{12}= 1$", hatch='//')
        plt.hist(x_f2, weights=y_f2, bins= np.arange(6-0.5*offset, 72-0.5*offset, 6), align= 'mid', color='#9AEE9A', width=offset, label=r"$f' = f_{2}, g_{12}= 1$", hatch='\\')

        if counts2!= 0:
            #bar2_f1 = plt.bar(x2_f1, y2_f1, color='#01B9FF', width=offset, label=r"$g_{12}= 0, f'= f_{1}$")
            #bar2_f2 = plt.bar(x2_f2, y2_f2, color='#C0EDFE', width=offset, label=r"$g_{12}= 0, f'= f_{2}$")
            plt.hist(x2_f1, weights=y2_f1, bins= np.arange(6+0.5*offset, 72+0.5*offset, 6), align= 'mid', color='#01B9FF', width=offset, label=r"$f' = f_{1}, g_{12}= 0$")
            #leg2 = ax.legend([bar1_f1, bar1_f2],[r'$f = f_{a}$',r'$f = f_{b}$'], loc='upper right', frameon=False, prop={'size': 12.5}, bbox_to_anchor=(1.,0.8))

        pmap= {'f1': r'$f_{1}$', 'af1': r'$f_{1}$', 'f2': r'$f_{2}$', 'af2': r'$f_{2}$', 'phi': r'$\phi$'}
        iP_str= ''
        for j in range(len(initialParticles)):
            iP_str+= pmap[self.ptype(initialParticles[j])]
            if j > 0: iP_str+= ', '

        ax.set_xticks(6*np.arange(1, 11))
        ax.set_xticklabels((iP_str + r"$\rightarrow f'$", iP_str + r"$\rightarrow f'\phi$", iP_str + r"$\rightarrow f'\phi\phi$", iP_str + r"$\rightarrow f'\phi\phi\phi$",
                            iP_str + r"$\rightarrow f' f_{1} \bar{f}_{1}$", iP_str + r"$\rightarrow f' f_{2} \bar{f}_{2}$", iP_str + r"$\rightarrow f' f_{1/2} \bar{f}_{2/1}$",
                            iP_str + r"$\rightarrow f' f_{1} \bar{f}_{1} \phi$", iP_str + r"$\rightarrow f' f_{2} \bar{f}_{2} \phi$", iP_str + r"$\rightarrow f' f_{1/2} \bar{f}_{2/1} \phi$"), size= 10)
        plt.xlabel('Final State', size=24)

        plt.legend(loc='upper right',prop={'size': 20})



        #plt.text(-0.3, 55*4, r"3-step Full Quantum Simulation", fontsize=24)
        plt.title(r"3-step Full Quantum Simulation", fontsize=28)
        plt.text(2.8, 200, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=20)

        plt.text(2.8, 100, r"Initial: " + iP_str, fontsize=20)

        f.savefig("sim3step_states_shots=%d.pdf" %(events))
        plt.show()