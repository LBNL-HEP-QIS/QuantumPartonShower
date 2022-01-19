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
        self.pReg, self.hReg, self.eReg= self.allocateQubits()

        self._circuit = QuantumCircuit(self.pReg, self.hReg, self.eReg)

    def __str__(self):
        tot_qubits= 3*(self._N + self._ni) + self._L + 1
        return "N= %d \n ni= %d \n L= %d \n Total qubits: %d" %(self._N, self._ni, self._L, tot_qubits)

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

    def P_bos_g(self, t, g):
        return g ** 2 *self.Phat_bos(t) / (4 * math.pi)

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
        eReg = QuantumRegister(1, 'e')

        return (pReg, hReg, eReg)


    def initializeParticles(self, initialParticles):
        """ Apply appropriate X gates to ensure that the p register contains all of the initial particles.
            The p registers contains particles in the form of a string '[MSB, middle bit, LSB]' """
        for currentParticleIndex in range(len(initialParticles)):
            for particleBit in range(3):
                pBit= 2 - particleBit
                if int(initialParticles[currentParticleIndex][particleBit]) == 1:
                    self._circuit.x(self.pReg[currentParticleIndex * self._p_len + pBit])






    def createCircuit(self, eps, g_1, g_2, g_12, initialParticles, verbose=False):
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
        if verbose:
            print('\n\ng_a= %.4f, g_b= %.4f, u= %.4f' %(g_a, g_b, u))
            print('Delta_aList: ' + str(Delta_aList))
            print('Delta_bList: ' + str(Delta_bList))
            print('Delta_phiList: ' + str(Delta_phiList))
            print('P_aList: ' + str(P_aList))
            print('P_bList: ' + str(P_bList))
            print('P_phiList: ' + str(P_phiList))
            print('timeStepList: ' + str(timeStepList))

        qubits = {'pReg': self.pReg, 'hReg': self.hReg, 'eReg': self.eReg}

        self.initializeParticles(initialParticles)

        (self.pReg_cl, self.hReg_cl) = self.allocateClbits()

        self.add_Clbits()

        #########################################################################################################
        # Step 1                                                                                                #
        #########################################################################################################
        print('Applying step 1.')

        # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
        self._circuit.ry((2 * math.asin(-u)), self.pReg[0])

        # assess if emmision occured (step 3)
        print('Apply uE()...')
        if verbose:
            print('\t DeltaA: ' + str(Delta_aList[0]))
            print('\t DeltaB: ' + str(Delta_bList[0]))
        self._circuit.x(self.pReg[0])
        self._circuit.cu3(2*np.arccos(math.sqrt(Delta_aList[0])), 0, 0, self.pReg[0], self.eReg[0]) #a emission rotation
        self._circuit.x(self.pReg[0])
        self._circuit.cu3(2*np.arccos(math.sqrt(Delta_bList[0])), 0, 0, self.pReg[0] , self.eReg[0]) #b emission rotation

        print('Measure and Reset |e>...')
        self._circuit.measure(self.eReg, self.hReg_cl[0][-1])
        self._circuit.reset(self.eReg)
        
        print('Apply U_h()...')
        #########################################################################################################
        #self._circuit.u3(2*np.arccos(0), 0, 0, self.hReg[0]).c_if(self.hReg_cl[0], 2**self._L)
        self._circuit.x(self.hReg[0]).c_if(self.hReg_cl[0], 2**self._L)
        #########################################################################################################

        print('Measure and reset |h>...')
        # NOTE: ONLY NEED TO MEASURE AND RESET hReg[0]
        #self._circuit.measure(self.hReg, self.hReg_cl[0][:self._L])
        #self._circuit.reset(self.hReg)
        self._circuit.measure(self.hReg[0], self.hReg_cl[0][0])
        self._circuit.reset(self.hReg[0])

        print('Apply U_p()...') # update particle based on which particle split/emmitted (step 5)
        self._circuit.x(self.pReg[3]).c_if(self.hReg_cl[0], 5)

        
        #########################################################################################################
        # Step 2                                                                                                #
        #########################################################################################################
        if True:
            print('\nApplying step 2.')

            print('Apply uE()...')
            if verbose:
                print('\t DeltaAphi: ' + str(Delta_phiList[1]*Delta_aList[1]))
                print('\t DeltaBphi: ' + str(Delta_phiList[1]*Delta_bList[1]))
                print('\t DeltaA: ' + str(Delta_aList[1]))
                print('\t DeltaB: ' + str(Delta_bList[1]))
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(Delta_phiList[1]*Delta_aList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 5)
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(Delta_phiList[1]*Delta_bList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 5)

            #print('a-phi emit angle: ' + str(2*np.arccos(math.sqrt(Delta_phiList[1]*Delta_aList[1]))))
            #print('b-phi emit angle: ' + str(2*np.arccos(math.sqrt(Delta_phiList[1]*Delta_bList[1]))))
            #print('a emit angle: ' + str(2*np.arccos(math.sqrt(Delta_aList[1]))))
            #print('b emit angle: ' + str(2*np.arccos(math.sqrt(Delta_bList[1]))))
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(Delta_aList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 0)
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(Delta_bList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 0)

            print('Measure and reset |e>...')
            self._circuit.measure(self.eReg, self.hReg_cl[0][-1])

            print('Apply U_h()...')
            # Over p0
            t_mid= timeStepList[1]
            entry_h_a = 0
            entry_h_aphi = math.sqrt(1-(self.P_f(t_mid, g_a)/(self.P_f(t_mid, g_a) + self.P_bos(t_mid, g_a, g_b)))) #off diagonals in A23
            entry_h_b = 0
            entry_h_bphi = math.sqrt(1-(self.P_f(t_mid, g_b)/(self.P_f(t_mid, g_b) + self.P_bos(t_mid, g_a, g_b))))

            self._circuit.x(self.pReg[0])
            #########################################################################################################
            #v1:self._circuit.cu3(2*np.arccos(entry_h_a), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            #v2: self._circuit.cx(self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            self._circuit.x(self.hReg[0]).c_if(self.hReg_cl[0], 4)
            #########################################################################################################
            
            self._circuit.cu3(2*np.arccos(entry_h_aphi), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 5)
            self._circuit.x(self.pReg[0])
            
            #########################################################################################################
            #v1: self._circuit.cu3(2*np.arccos(entry_h_b), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            #v2: self._circuit.cx(self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            # current: combined with the cx a few lines up
            #########################################################################################################
            
            self._circuit.cu3(2*np.arccos(entry_h_bphi), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 5)

            self._circuit.measure(self.hReg[0], self.hReg_cl[1][0])
            self._circuit.measure(self.eReg, self.hReg_cl[1][-1])
            self._circuit.reset(self.eReg)

            # Now over p1
            entry_h_phi = 0

            #########################################################################################################
            #v1: self._circuit.cu3(2*np.arccos(entry_h_phi), 0, 0, self.pReg[3], self.hReg[1]).c_if(self.hReg_cl[1], 4)
            #v2: self._circuit.cx(self.pReg[3], self.hReg[1]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.hReg[1]).c_if(self.hReg_cl[1], 4)
            #########################################################################################################
            
            # NOTE: WE DON'T NEED TO MEASURE hReg[1] TO GET WHAT WE WANT. IN FACT WE REALLY ONLY NEED 2 
            #       CLASSICAL BITS TO CONDITION ON
            #self._circuit.measure(self.hReg[1], self.hReg_cl[1][1])


            print('Apply U_p()...')
            self._circuit.x(self.pReg[6]).c_if(self.hReg_cl[1], 5)
            self._circuit.x(self.pReg[8]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[5]).c_if(self.hReg_cl[1], 4)

            self._circuit.h(self.pReg[7]).c_if(self.hReg_cl[1], 4)
            entry_r = g_a / (math.sqrt(g_a*g_a + g_b*g_b))

            self._circuit.u3(2*np.arccos(entry_r), 0, 0, self.pReg[6]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[7])
            self._circuit.cx(self.pReg[7], self.pReg[4]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[7])
            self._circuit.x(self.pReg[6])
            self._circuit.cx(self.pReg[6], self.pReg[3]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[6])

        # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
        index2 = 0
        while index2 < self.pReg.size:
            # circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
            self._circuit.cry((2 * math.asin(u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
            index2 += self._p_len

        print('Done.')
        return self._circuit, qubits



    def allocateClbits(self):
        '''
        We have to measure |h> and |e> on the same register to have to proper classical controls.

        '''
        nbits_h = self._N * self._L

        pReg_cl = []
        for j in range(self._N + self._ni):
            pReg_cl.append(ClassicalRegister(3, 'p%d_cl' %(j)))

        # We extend each hReg_cl[i] by one qubit. This extra qubit is a place to measure |e>, if needed there.
        hReg_cl = []
        for j in range(self._N):
            hReg_cl.append(ClassicalRegister(self._L + 1, 'h%d_cl' %(j)))

        return (pReg_cl, hReg_cl)

    def add_Clbits(self):
        # Add all classical registers stored in self to self._circuit
        for j in range(self._N + self._ni):
            self._circuit.add_register(self.pReg_cl[j])
        for j in range(self._N):
            self._circuit.add_register(self.hReg_cl[j])


    def measure_Clbits(self):
        # Measures all bits other than the history register (already measured)
        for j in range(self._N + self._ni):
            self._circuit.measure(self.pReg[3*j : 3*(j+1)], self.pReg_cl[j])


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










    def MCMC(self, eps, g_a, g_b, na_i, nb_i, verbose=False):
        '''
        na_i, nb_i are the initial number of a and b fermions.

        '''

        n_a= na_i
        n_b= nb_i
        n_phi= 0

        n_emits= 0

        for i in range(self._N):
            # Compute time steps
            t_up = eps ** ((i) / self._N)
            t_mid = eps ** ((i + 0.5) / self._N)
            t_low = eps ** ((i + 1) / self._N)
            # Compute values for emission matrices
            Delta_a = self.Delta_f(t_low, g_a) / self.Delta_f(t_up, g_a)
            Delta_b = self.Delta_f(t_low, g_b) / self.Delta_f(t_up, g_b)
            Delta_phi = self.Delta_bos(t_low, g_a, g_b) / self.Delta_bos(t_up, g_a, g_b)
            P_a, P_b, P_phi = self.P_f(t_mid, g_a), self.P_f(t_mid, g_b), self.P_bos(t_mid, g_a, g_b)

            P_phi_a= self.P_bos_g(t_mid, g_a)
            P_phi_b= self.P_bos_g(t_mid, g_b)

            Pemit= 1 - (Delta_a ** n_a) * (Delta_b ** n_b) * (Delta_phi ** n_phi)

            denom= (P_a * n_a) + (P_b * n_b) + (P_phi * n_phi)
            emit_a= (P_a * n_a) / denom
            emit_b= (P_b * n_b) / denom
            emit_phi= (P_phi * n_phi) / denom # = emit_phi_a + emit_phi_b
            emit_phi_a= (P_phi_a * n_phi) / denom
            emit_phi_b= (P_phi_b * n_phi) / denom 

            emit_a *= Pemit
            emit_b *= Pemit
            emit_phi*= Pemit
            emit_phi_a *= Pemit
            emit_phi_b *= Pemit

            cut_a= emit_a
            cut_b= cut_a + emit_b
            cut_phi_a= cut_b + emit_phi_a
            cut_phi_b= cut_phi_a + emit_phi_b

            r= np.random.uniform(0, 1)

            if r < cut_a:
                n_phi+= 1
            elif r < cut_b:
                n_phi+= 1
            elif r < cut_phi_a:
                n_phi-= 1
                n_a+= 2
            elif r < cut_phi_b:
                n_phi-= 1
                n_b+= 2
            else: 
                n_emits-= 1
            n_emits+= 1

            if verbose:
                print('\n\nDelta_a: ' + str(Delta_a))
                print('Delta_b: ' + str(Delta_b))
                print('Delta_phi: ' + str(Delta_phi))
                print('P_a: ' + str(P_a))
                print('P_b: ' + str(P_b))
                print('P_phi_a: ' + str(P_phi_a))
                print('P_phi_b: ' + str(P_phi_b))
                print('P_phi: ' + str(P_phi))
                print('t_mid: ' + str(t_mid))

                print('\nStep %d' %(i+1))
                print('P(emit a)= ' + str(emit_a))
                print('P(emit b)= ' + str(emit_b))
                print('P(emit phi -> aa)= ' + str(emit_phi_a))
                print('P(emit phi -> bb)= ' + str(emit_phi_b))
                print('P(emit phi)= ' + str(emit_phi))
                print('P(no emit)= ' + str(1 - Pemit))
        
        #print('\nNumber of emissions: %d' %(n_emits))
        return n_emits, n_a, n_b, n_phi

    def P(self, g):
        alpha = g**2 / (4 * math.pi)
        return alpha

    def Delta(self, lnt, g):
        alpha = g**2 / (4 * math.pi)
        return math.exp(alpha * lnt)

    # The analytical distribution of the hardest emission
    def dsigma_d_t_max(self, lnt, lneps, g):
        return self.P(g) * self.Delta(lnt, g) / (1 - self.Delta(lneps, g))