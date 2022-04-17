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
        self._L = int(math.floor(math.log(N + ni - 1, 2)) + 1)

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
        w_hReg = QuantumRegister(max(self._L - 1, 1), 'w_h')

        eReg = QuantumRegister(1, 'e')
        wReg = QuantumRegister(3, 'w')  # we use all 5 of these work register qubits, but not sure why it is 5

        n_aReg = QuantumRegister(self._L, 'n_a')
        #w_aReg = QuantumRegister(self._L, 'w_a')
        w_aReg = QuantumRegister(max(self._L - 1, 1), 'w_a')


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


    def incrementer(self, l, b, a, control):
        '''
            b= countReg
            a= workReg
            len(workReg) = len(countReg) - 1 = l - 1
        '''
        print('Incrementer l= %d' %(l))
        if l > 1:
            self._circuit.ccx(control, b[0], a[0])
        for j in range(1, l-1):
            self._circuit.ccx(a[j-1], b[j], a[j])

        for j in reversed(range(1, l)):
            self._circuit.cx(a[j-1], b[j])
            if j > 1:
                self._circuit.ccx(a[j-2], b[j-1], a[j-1])
            else:
                self._circuit.ccx(control, b[j-1], a[j-1])

        self._circuit.cx(control, b[0])



    def uCount(self, m, l):
        """
        Populate the count registers using current particle states.
        Uses wReg[0] as the control and wReg[1] as ancilla qubit for flavorControl and plus1, respectively
        """
        for k in range(self._ni + m):
            # a fermions
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            self.incrementer(l, self.n_aReg, self.w_aReg, self.wReg[0])
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
        print(hmap)
        ################
        # OVERRIDE --> Don't want to loop over every key in hmap, just na=0 through max(n_tot - n_phi)
        ################
        max_na= 0
        for key in hmap:
            n_tot, n_phi= key
            if n_tot - n_phi > max_na:
                max_na= n_tot - n_phi
        print('max_na= %d' %(max_na))

        if m > 0:
            h_bool= [int(''.join(hmap[key][0]), 2)]
            for j in range(1, len(hmap[key])):
                h_bool+= [int(''.join(hmap[key][j]), 2)]
        else:
            h_bool= None

        ################
        # OVERRIDE    
        h_bool= None
        ################

        ################
        # OVERRIDE, n_tot - n_phi --> max_na
        for n_a in range(0, max_na + 1):
            n_b= max_na - n_a
            Delta = Delta_phi ** n_phi * Delta_a ** n_a * Delta_b ** n_b
            #print('na= %d, nb= %d, nphi= %d, emit angle: ' %(n_a, n_b, n_phi) + str((2 * math.acos(np.sqrt(Delta)))))
            aControlQub = self.numberControl(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)
            if m > 0 and h_bool != None: 
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

        ################
        # OVERRIDE    
        h_bool= None
        ################

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


    def minus1(self, l, countReg, workReg, control):
        """
        Recursively carries an subtraction of 1 to the LSB of a register to all bits if control == 1
        Equivalent to plus1 but with an X applied to all count qubits before and after gate
        """
        [self._circuit.x(qubit) for qubit in countReg]
        self.incrementer(l, countReg, workReg, control)
        [self._circuit.x(qubit) for qubit in countReg]


    def U_h(self, l, m, P_phi, P_a, P_b, initialParticles):
        """Implement U_h from paper"""
        #print(P_phi, P_a, P_b)
        for k in reversed(range(self._ni + m)):
            if k >= self._ni:
                hmap= QuantumPartonShower.h_map(k - self._ni + 1, self._L, [int(x, 2) for x in initialParticles])
            else:
                hmap= QuantumPartonShower.h_map(0, self._L, [int(x, 2) for x in initialParticles[:k+1]])
            
            ################
            # OVERRIDE --> see Ue
            ################
            max_na= 0
            for key in hmap:
                n_tot, n_phi= key
                if n_tot - n_phi > max_na:
                    max_na= n_tot - n_phi
            print('max_na= %d' %(max_na))
            
            #for key in hmap:
            #    n_tot, n_phi= key
            #    if m == 0 or k < self._ni:
            #        h_bool= None
            #    else:
            #        h_bool= [int(''.join(hmap[key][0]), 2)]
            #        for j in range(1, len(hmap[key])):
            #            h_bool+= [int(''.join(hmap[key][j]), 2)]

            ################
            # OVERRIDE    
            h_bool= None
            ################

            ################
            # OVERRIDE, n_tot - n_phi --> max_na
            ################
            for n_a in range(0, max_na + 1):
                #print('\n')
                n_b= max_na - n_a
                aControl = self.numberControl(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)
                # controlled R-y from |0> to |k> on all qubits with all possible angles depending on n_phi, n_a, n_b, and flavor
                for flavor in ['phi', 'a', 'b']:
                    angle = self.U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b)
                    #print('na= %d, nb= %d, nphi= %d, flavor: ' %(n_a, n_b, n_phi) + flavor + ', angle= ' + str(angle))
                    self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 2, h_bool= h_bool)  # wReg[2] is work qubit but is reset to 0
                    if m == 0 or k < self._ni or h_bool == None:
                        self._circuit.ccx(aControl, self.wReg[0], self.wReg[1])
                        self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2])
                    else:
                        for h in h_bool:
                            self._circuit.ccx(aControl, self.wReg[0], self.wReg[1]).c_if(self.hReg_cl, h)
                            self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2]).c_if(self.hReg_cl, h)
                    self.twoLevelControlledRy(l, angle, k + 1, self.wReg[2], self.hReg, self.w_hReg, h_bool= h_bool)

                    if m == 0 or k < self._ni or h_bool == None:
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
            self.minus1(l, self.n_aReg, self.w_aReg, self.wReg[0])
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
            #index = 0
            #while index < self.pReg.size:
            #    self._circuit.cry((2 * math.asin(-u)), self.pReg[index + 2], self.pReg[index + 0])
            #    index += self._p_len

            # populate count register (step 2)
            print('Apply uCount()...')
            #self.uCount(m, l)

            # assess if emmision occured (step 3)
            print('Apply uE()...')
            #self.uE(l, m, Delta_phiList[m], Delta_aList[m], Delta_bList[m], initialParticles)

            print('Apply U_h()...')
            self.U_h(l, m, P_phiList[m], P_aList[m], P_bList[m], initialParticles)

            print('Measure and reset |h>...')
            #self._circuit.measure(self.hReg, self.hReg_cl[m*self._L : (m+1)*self._L])
            #self._circuit.reset(self.hReg)

            print('Apply U_p()...')
            # update particle based on which particle split/emmitted (step 5)
            #self.U_p(l, m, g_a, g_b)

            # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
            #index2 = 0
            #while index2 < self.pReg.size:
                # circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
            #    self._circuit.cry((2 * math.asin(u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
            #    index2 += self._p_len

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
        w_hReg_cl = ClassicalRegister(max(self._L - 1, 1), 'wh_cl')
        w_aReg_cl = ClassicalRegister(max(self._L - 1, 1), 'wa_cl')
        
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
