import qiskit
import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import QuantumRegister

N = 1
m = 1
n_i = 1
L = int(math.ceil(math.log(N + n_i, 2)))

#Define these global variables for indexing - to convert from cirq's grid qubits 
p_len = 3
h_len = L
w_h_len = L-1
e_len = 1
w_len = 5
np_len = L
wp_len = L - 1
na_len = L
wa_len = L - 1
nb_len = L
wb_len = L - 1 

def P_f(t, g):
    alpha = g**2 * Phat_f(t)/ (4 * math.pi)
    return alpha

def Phat_f(t):
    return math.log(t)

def Phat_bos(t):
    return math.log(t)

def Delta_f(t, g):
    return math.exp(P_f(t,g))

def P_bos(t, g_a, g_b):
    alpha = g_a**2 * Phat_bos(t)/ (4 * math.pi) + g_b**2 * Phat_bos(t)/ (4 * math.pi)
    return alpha

def Delta_bos(t, g_a, g_b):
    return math.exp(P_bos(t, g_a, g_b))

def populateParameterLists(N, timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList, g_a,
                           g_b, eps):
    """Populates the 6 lists with correct values for each time step theta"""
    for i in range(N):
        # Compute time steps
        t_up = eps ** ((i) / N)
        t_mid = eps ** ((i + 0.5) / N)
        t_low = eps ** ((i + 1) / N)
        timeStepList.append(t_mid)
        # Compute values for emission matrices   
        Delta_a = Delta_f(t_low, g_a)/ Delta_f(t_up, g_a)
        Delta_b = Delta_f(t_low, g_b) / Delta_f(t_up, g_b)
        Delta_phi = Delta_bos(t_low, g_a, g_b) / Delta_bos(t_up, g_a, g_b)
        P_a, P_b, P_phi = P_f(t_mid, g_a), P_f(t_mid, g_b), P_bos(t_mid, g_a, g_b)

        # Add them to the list
        P_aList.append(P_a)
        P_bList.append(P_b)
        P_phiList.append(P_phi)
        Delta_aList.append(Delta_a)
        Delta_bList.append(Delta_b)
        Delta_phiList.append(Delta_phi)
        
        
def allocateQubits(N, n_i, L):
    nqubits_p = 6
    nqubits_h = N*math.ceil(math.log2((N + n_i)))
    nqubits_e = 1
    nqubits_a_b_phi = L
    print('nqubits_a_b_phi: ', nqubits_a_b_phi)
    
    #to convert from cirq's grid qubits
    p_len = 3
    h_len = L
    w_h_len = L-1
    
    pReg = QuantumRegister(nqubits_p)
    
    hReg = QuantumRegister(nqubits_h)
    w_hReg = QuantumRegister(nqubits_h)
    
    eReg = QuantumRegister(nqubits_e)
    wReg = QuantumRegister(5)
    
    n_aReg = QuantumRegister(nqubits_a_b_phi)
    w_aReg = QuantumRegister(nqubits_a_b_phi)
    
    n_bReg = QuantumRegister(nqubits_a_b_phi)
    w_bReg = QuantumRegister(nqubits_a_b_phi)
    
    n_phiReg = QuantumRegister(nqubits_a_b_phi)
    w_phiReg = QuantumRegister(nqubits_a_b_phi)
    
    return (pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)


def intializeParticles(circuit, pReg, initialParticles):
    """ Apply appropriate X gates to ensure that the p register contains all of the initial particles.
        The p registers contains particles in the form of a list [LSB, middle bit, MSB]"""
    for currentParticleIndex in range(len(initialParticles)):
        for particleBit in range(3):
            if initialParticles[currentParticleIndex][particleBit] == 1:
                circuit.x(pReg[currentParticleIndex][particleBit])


def flavorControl(circuit, flavor, control, target, ancilla, control_index, target_index, ancilla_index):
    """Controlled x onto targetQubit if "control" particle is of the correct flavor"""
    if flavor == "phi":
        circuit.x(control[control_index + 1])
        circuit.x(control[control_index + 2])
        print("ancilla: ",ancilla)
        print("ancilla index: ",ancilla_index)
        print(ancilla[1])
        circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[1])
        circuit.ccx(control[control_index + 2], ancilla[1], target[target_index + 0])
        # undo work
        circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[1])
        circuit.x(0)
        circuit.x(1)
    if flavor == "a":
        circuit.x(0)
        circuit.ccx(control[control_index + 0], control[control_index + 1], target[target_index + 0])
        # undo work
        circuit.x(0)
    if flavor == "b":
        circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])
#     print("circuit in flavorcontrol: ",circuit)
    
def plus1(circuit, l, countReg, workReg, control, ancilla, level):
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
        print("countReg ", countReg)
        print("countReg[level + 1]: ", countReg[level + 1])              
        circuit.ccx(workReg[level], control, countReg[level + 1])
        # recursively call next layer
        plus1(circuit, l, countReg, workReg, control, ancilla, level + 1)
        # undo work qubits (exact opposite of first 7 lines - undoes calculation)
        if level == 0:
            circuit.ccx(countReg[0], control, workReg[0])
            [circuit.x(qubit) for qubit in countReg]
        else:
            circuit.ccx(countReg[level], workReg[level - 1], ancilla)
            circuit.ccx(ancilla, control, workReg[level])
            circuit.ccx(countReg[level], workReg[level - 1], ancilla)

        
def uCount(circuit, m, n_i, l, pReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg):
    """
    Populate the count registers using current particle states.
    Uses wReg[0] as the control and wReg[1] as ancilla qubit for flavorControl and plus1, respectively
    """
    for k in range(n_i+m):
#         # bosons
#         flavorControl(circuit, "phi", pReg[k], wReg[0], wReg[1])
#         plus1(circuit, l, n_phiReg, w_phiReg, wReg[0], wReg[1], 0)
#         flavorControl(circuit, "phi", pReg[k], wReg[0], wReg[1])
#         # a fermions
#         flavorControl(circuit, "a", pReg[k], wReg[0], wReg[1])
#         plus1(circuit, l, n_aReg, w_aReg, wReg[0], wReg[1], 0)
#         flavorControl(circuit, "a", pReg[k], wReg[0], wReg[1])
#         # b fermions
#         flavorControl(circuit, "b", pReg[k], wReg[0], wReg[1])
#         plus1(circuit, l, n_bReg, w_bReg, wReg[0], wReg[1], 0)
#         flavorControl(circuit, "b", pReg[k], wReg[0], wReg[1])
        # bosons
        flavorControl(circuit, "phi", pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len))
        plus1(circuit, l, n_phiReg, w_phiReg, wReg[0], wReg[1], 0)
        flavorControl(circuit, "phi", pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len))
        # a fermions
        flavorControl(circuit, "a", pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len))
        plus1(circuit, l, n_aReg, w_aReg, wReg[0], wReg[1], 0)
        flavorControl(circuit, "a", pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len))
        # b fermions
        flavorControl(circuit, "b", pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len))
        plus1(circuit, l, n_bReg, w_bReg, wReg[0], wReg[1], 0)
        flavorControl(circuit, "b", pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len))
        
def generateParticleCounts(n_i, m, k):
    """Fill countsList with all combinations of n_phi, n_a, and n_b where each n lies in range [0, n_i+m-k],
    and the sum of all n's lies in range [n_i-k, m+n_i-k], all inclusive
    """
    countsList = []
    for numParticles in range(n_i-k, m+n_i-k+1):
        for numPhi in range(0, n_i+m-k+1):
            for numA in range(0, numParticles-numPhi+1):
                numB = numParticles - numPhi - numA
                countsList.append([numPhi, numA, numB])
    return countsList

def reverse(lst):
    """reverse a list in place"""
    lst.reverse()
    return lst

def intToBinary(l, number):
    """Converts integer to binary list of size l with LSB first and MSB last"""
    numberBinary = [int(x) for x in list('{0:0b}'.format(number))]
    numberBinary = (l - len(numberBinary)) * [0] + numberBinary
    return reverse(numberBinary)


def numberControl(circuit, l, number, countReg, workReg):
    """
    Applies an X to the l-2 (0 indexed) qubit of the work register if count register encodes the inputted number in binary
    returns this l-2 qubit, unless l=1, in which case return the only count register qubit
    DOES NOT CLEAN AFTER ITSELF - USE numberControlT to clean after this operation
    """
    if type(number) == int:
        numberBinary = intToBinary(l, number)
    else:
        numberBinary = number
    [circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]
    # first level does not use work qubits as control
    if l > 1:
        print("l>1")
        circuit.ccx(countReg[0], countReg[1], workReg[0])
        # subfunction to recursively handle toffoli gates

    def binaryToffolis(level):
        circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
        if level < l - 1:
            binaryToffolis(level + 1)

    if l > 2:
        print("l>2")
        binaryToffolis(2)
    # return qubit containing outcome of the operation
    if l == 1:
        return countReg[0]
    else:
        return workReg[l - 2]

def numberControlT(circuit, l, number, countReg, workReg):
    """CLEANS AFTER numberControl operation"""
    if type(number) == int:
        numberBinary = intToBinary(l, number)
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
    
def uE(circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, Delta_phi, Delta_a, Delta_b):
    """Determine if emission occured in current step m"""
    countsList = generateParticleCounts(n_i, m, 0)
    print("counts list: ", countsList)
    print("length counts list: ", len(countsList))

    for counts in countsList:
        n_phi, n_a, n_b = counts[0], counts[1], counts[2]
        Delta = Delta_phi**n_phi * Delta_a**n_a * Delta_b**n_b
        phiControlQub = numberControl(circuit, l, n_phi, n_phiReg, w_phiReg)
        aControlQub = numberControl(circuit, l, n_a, n_aReg, w_aReg)
        bControlQub = numberControl(circuit, l, n_b, n_bReg, w_bReg)
        circuit.ccx(phiControlQub, aControlQub, wReg[0])
        circuit.ccx(bControlQub, wReg[0], wReg[1])
        circuit.cry((2*math.acos(np.sqrt(Delta))),wReg[1], eReg[0])
        #undo
        circuit.ccx(bControlQub, wReg[0], wReg[1])
        circuit.ccx(phiControlQub, aControlQub, wReg[0])
        numberControlT(circuit, l, n_b, n_bReg, w_bReg)
        numberControlT(circuit, l, n_a, n_aReg, w_aReg)
        numberControlT(circuit, l, n_phi, n_phiReg, w_phiReg)
        
def generateGrayList(l, number):
    """
    l is the size of the current count register
    Return list of elements in gray code from |0> to |number> where each entry is of type[int, binary list].
    int: which bit is the target in the current iteration, binary list: the state of the rest of the qubits (controls)
    """
    grayList = [[0, l * [0]]]
    targetBinary = intToBinary(l, number)
    for index in range(len(targetBinary)):
        if targetBinary[index] == 1:
            grayList.append([index, (list(grayList[-1][1]))])
            grayList[-1][1][index] = 1
    return grayList[1:]
        
def twoLevelControlledRy(circuit, l, angle, k, externalControl, reg, workReg):
    """
    Implements two level Ry rotation from state |0> to |k>, if externalControl qubit is on
    for reference: http://www.physics.udel.edu/~msafrono/650/Lecture%206.pdf
    """
    print("l: ", l, "\nk: ", k)
    grayList = generateGrayList(l, k)
    # handle the case where l=0 or 1
    if k==0:
        return
    if l == 1 and k == 1:
        circuit.cry(angle, externalControl, reg[0])
        return

    # swap states according to Gray Code until one step before the end
    for element in grayList:
        targetQub = element[0]
        number = element[1]
        number = number[0:targetQub] + number[targetQub + 1:]
        controlQub = numberControl(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
        if element == grayList[-1]:  # reached end
            circuit.ccx(controlQub, externalControl, workReg[l - 2])
            circuit.cry(angle, workReg[l - 2], reg[targetQub])
            circuit.ccx(controlQub, externalControl, workReg[l - 2])
        else:  # swap states
            circuit.cx(controlQub, reg[targetQub])
        numberControlT(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)

    # undo
    for element in reverse(grayList[:-1]):
        targetQub = element[0]
        number = element[1]
        number = number[0:targetQub] + number[targetQub + 1:]
        controlQub = numberControl(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
        circuit.cx(controlQub, reg[targetQub])
        numberControlT(circuit, l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
    return

def U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b):
    """Determine angle of rotation used in U_h"""
    denominator = n_phi * P_phi + n_a * P_a + n_b * P_b
    if denominator == 0: # occurs if we are trying the case of no particles remaining (n_a = n_b = n_phi = 0)
        return 0
    flavorStringToP = {'phi': P_phi, 'a': P_a, 'b': P_b}
    emissionAmplitude = np.sqrt(flavorStringToP[flavor] / denominator)
    # correct for arcsin input greater than 1 errors for various input combinations that are irrelevant anyway
    emissionAmplitude = min(1, emissionAmplitude)
    return 2 * np.arcsin(emissionAmplitude)


def minus1(circuit, l, countReg, workReg, control, ancilla, level):
    """
    Recursively carries an subtraction of 1 to the LSB of a register to all bits if control == 1
    Equivalent to plus1 but with an X applied to all count qubits before and after gate
    """
    [circuit.x(qubit) for qubit in countReg]
    plus1(circuit, l, countReg, workReg, control, ancilla, level)
    [circuit.x(qubit) for qubit in countReg]


def U_h(circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, pReg, hReg, w_hReg, P_phi,
        P_a, P_b):
    """Implement U_h from paper"""
    for k in range(n_i + m):
        countsList = generateParticleCounts(n_i, m, k)  # reduce the available number of particles
        for counts in countsList:
            n_phi, n_a, n_b = counts[0], counts[1], counts[2]
            # controlled R-y from |0> to |k> on all qubits with all possible angles depending on n_phi, n_a, n_b, and flavor
            for flavor in ['phi', 'a', 'b']:
                angle = U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b)
                phiControl = numberControl(circuit, l, n_phi, n_phiReg, w_phiReg)
                aControl = numberControl(circuit, l, n_a, n_aReg, w_aReg)
                bControl = numberControl(circuit, l, n_b, n_bReg, w_bReg)
                circuit.ccx(phiControl, aControl, wReg[0])
                circuit.ccx(bControl, wReg[0], wReg[1])
#                 flavorControl(circuit, flavor, pReg[k], wReg[2], wReg[4]) # wReg[4] is work qubit but is reset to 0
                flavorControl(circuit, flavor, pReg, wReg, wReg, (k*p_len), (2*w_len), (4*w_len)) # wReg[4] is work qubit but is reset to 0
                circuit.ccx(wReg[1], wReg[2], wReg[3])
                circuit.ccx(eReg[0], wReg[3], wReg[4])

                twoLevelControlledRy(circuit, l, angle, k+1, wReg, hReg, w_hReg)

                circuit.ccx(eReg[0], wReg[3], wReg[4])  # next steps undo work qubits
                circuit.ccx(wReg[1], wReg[2], wReg[3])
                flavorControl(circuit, flavor, pReg, wReg, wReg, (k*p_len), (2*w_len), (4*w_len)) # wReg[4] is work qubit but is reset to 0
                circuit.ccx(bControl, wReg[0], wReg[1])
                circuit.ccx(phiControl, aControl, wReg[0])
                numberControlT(circuit, l, n_b, n_bReg, w_bReg)
                numberControlT(circuit, l, n_a, n_aReg, w_aReg)
                numberControlT(circuit, l, n_phi, n_phiReg, w_phiReg)

        # subtract from the counts register depending on which flavor particle emitted
        for flavor, countReg, workReg in zip(['phi', 'a', 'b'], [n_phiReg, n_aReg, n_bReg], [w_phiReg, w_aReg, w_bReg]):
            flavorControl(circuit, flavor, pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len)) # wReg[4] is work qubit but is reset to 0
            minus1(circuit, l, countReg, workReg, wReg[0], wReg[1], 0)
            flavorControl(circuit, flavor, pReg, wReg, wReg, (k*p_len), (0*w_len), (1*w_len)) # wReg[4] is work qubit but is reset to 0

    # apply x on eReg if hReg[m] = 0, apply another x so we essentially control on not 0 instead of 0
    isZeroControl = numberControl(circuit, l, 0, hReg[m], w_hReg)
    circuit.cx(isZeroControl, eReg[0])
    circuit.x(eReg[0])
    numberControlT(circuit, l, 0, hReg[m], w_hReg)
    
def updateParticles(circuit, l, n_i, m, k, pReg, wReg, controlQub, g_a, g_b):
    """Updates particle if controlQub is on"""
    oldParticleReg = pReg
    newParticleReg = pReg
    #first gate in paper U_p
    circuit.ccx(controlQub, oldParticleReg[k + 2], newParticleReg[n_i+m+0])
    #second gate in paper (undoes work register immediately)
    circuit.x(oldParticleReg[k+1]) 
    circuit.x(oldParticleReg[k+2])
    circuit.ccx(controlQub, oldParticleReg[k+2], wReg[0])
    circuit.ccx(wReg[0], oldParticleReg[k+1], wReg[1])
    circuit.ccx(wReg[1], oldParticleReg[k+0], newParticleReg[n_i+m+2])
    circuit.ccx(wReg[0], oldParticleReg[k+1], wReg[1])
    circuit.ccx(controlQub, oldParticleReg[k+2], wReg[0])
    circuit.x(oldParticleReg[k+1])
    circuit.x(oldParticleReg[k+2])
    #third gate in paper
    circuit.ccx(controlQub, newParticleReg[n_i+m+2], oldParticleReg[k+2])
    #fourth and fifth gate in paper (then undoes work register)
    circuit.ccx(controlQub, newParticleReg[n_i+m+2], wReg[0])
    #check the format for the control state here
    circuit.h.control(num_ctrl_qubits=2, ctrl_state=str(wReg[0], newParticleReg[n_i+m+1])) 
    angle = (2 * np.arccos(g_a/np.sqrt(g_a**2 + g_b**2)))
    circuit.cry(angle, wReg[0], newParticleReg[n_i+m+0])
    circuit.ccx(controlQub, newParticleReg[n_i+m+2], wReg[0])
    #sixth and seventh gate in paper (then undoes work register)
    circuit.x(newParticleReg[n_i+m+0])
    cricuit.x(newParticleReg[n_i+m+1])
    circuit.ccx(newParticleReg[n_i+m+1], newParticleReg[n_i+m+2], wReg[0])
    circuit.ccx(controlQub, wReg[0], oldParticleReg[k+1])
    circuit.ccx(newParticleReg[n_i+m+1], newParticleReg[n_i+m+2], wReg[0])
    circuit.ccx(newParticleReg[n_i+m+0], newParticleReg[n_i+m+2], wReg[0])
    circuit.ccx(controlQub, wReg[0], oldParticleReg[k+0])
    circuit.ccx(newParticleReg[n_i+m+0], newParticleReg[n_i+m+2], wReg[0])
    circuit.x(newParticleReg[n_i+m+0])
    circuit.d(newParticleReg[n_i+m+1])

def U_p(circuit, l, n_i, m, pReg, hReg, w_hReg, wReg, g_a, g_b):
    """Applies U_p from paper"""
    for k in range(0, n_i + m):
#         controlQub = numberControl(circuit, l, k+1, hReg[m], w_hReg)
        controlQub = numberControl(circuit, l, k+1, hReg, w_hReg)
        updateParticles(circuit, l, n_i, m, k, pReg, wReg, controlQub, g_a, g_b)
        numberControlT(circuit, l, k+1, hReg[m], w_hReg)
        
        
def createCircuit(n_i, N, eps, g_1, g_2, g_12, initialParticles):
    """
    Create full circuit with n_i initial particles and N steps
    Inputs:
    n_i: number of initial particles
    N: number of steps
    eps, g_1, g_2, g_12: pre-chosen qft parameters
    initialParticles: list of initial particles, each particle in a list of qubits [MSB middle bit, LSB]
    (opposite order of the paper pg 6 - e.g a f_a fermion is [0,0,1])
    in order [particle 1, particle 2, ..... particle n_i]
    """
    # calculate constants
    gp = math.sqrt(abs((g_1 - g_2) ** 2 + 4 * g_12 ** 2))
    if g_1 > g_2:
        gp = -gp
    g_a, g_b = (g_1 + g_2 - gp) / 2, (g_1 + g_2 + gp) / 2
    u = math.sqrt(abs((gp + g_1 - g_2)/ (2 * gp)))
    
    L = int(math.floor(math.log(N + n_i, 2)) + 1)

    # evaluate P(Theta) and Delta(Theta) at every time step
    timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList = [], [], [], [], [], [], []
    populateParameterLists(N, timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList, g_a,
                           g_b, eps)

    # allocate and populate registers
    pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = allocateQubits(N, n_i, L)

    qubits = {'pReg': pReg, 'hReg': hReg, 'w_hReg': w_hReg, 'eReg': eReg, 'wReg': wReg, 'n_aReg': n_aReg,
               'w_aReg': w_aReg, 'n_bReg': n_bReg, 'w_bReg': w_bReg, 'n_phiReg': n_phiReg, 'w_phiReg': w_phiReg}

    # create circuit object and initialize particles
    circuit = qiskit.QuantumCircuit()
    intializeParticles(circuit, pReg, initialParticles)
        
    print("pReg", pReg)

    # begin stepping through subcircuits
    for m in range(N):
        l = int(math.floor(math.log(m + n_i, 2)) + 1)

        # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
        for p_k in pReg:
            circuit.append(ry(2*math.asin(-u)).controlled().on(p_k[2], p_k[0]))

        # populate count register (step 2)
        uCount(circuit, m, n_i, l, pReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)

        # assess if emmision occured (step 3)
        uE(circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg,
           Delta_phiList[m], Delta_aList[m], Delta_bList[m])

        # choose a particle to split (step 4)
        U_h(circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, pReg, hReg, w_hReg,
            P_phiList[m], P_aList[m], P_bList[m])

        # update particle based on which particle split/emmitted (step 5)
        U_p(circuit, l, n_i, m, pReg, hReg, w_hReg, wReg, g_a, g_b)

        # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
        for p_k in pReg:
            circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
    
    print('generated circuit on', len(flatten(list(qubits.values()))), 'qubits') 

    return circuit, qubits
