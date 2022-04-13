import cirq
import math
import create_circuit as cc
from cirq import Simulator
from cirq import GridQubit, X, CNOT, TOFFOLI, ry

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
from qiskit.providers.aer import QasmSimulator
import qiskit_create_circ as qcc

N = 1
n_i = 1
m = 0
# L = int(math.ceil(math.log(N + n_i, 2)))
L = int(math.floor(math.log(N + n_i, 2)) + 1)
# print(L)
l = int(math.floor(math.log(m + n_i, 2)) + 1)
# l = int(math.ceil(math.log(m + n_i, 2)) )


pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg = [], [], [], [], [], [], [], [], [], [], []
cc.allocateQubs(N, n_i, L, pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
qubits = {'pReg': pReg, 'hReg': hReg, 'w_hReg': w_hReg, 'eReg': eReg, 'wReg': wReg, 'n_aReg': n_aReg,
               'w_aReg': w_aReg, 'n_bReg': n_bReg, 'w_bReg': w_bReg, 'n_phiReg': n_phiReg, 'w_phiReg': w_phiReg}

g_1, g_2, g_12 = 2, 1, 0
gp = math.sqrt(abs((g_1 - g_2) ** 2 + 4 * g_12 ** 2))
if g_1 > g_2:
    gp = -gp
g_a, g_b = (g_1 + g_2 - gp) / 2, (g_1 + g_2 + gp) / 2
u = math.sqrt(abs((gp + g_1 - g_2)/ (2 * gp)))
eps = .001

circuit = cirq.Circuit()
simulator = Simulator()

circuit.append([[X((qubit)) for qubit in pReg[i]] for i in range(N+n_i)])
circuit.append([[X((qubit)) for qubit in hReg[i]] for i in range(N)])
w_hReg.extend([GridQubit(N+n_i,j) for j in range(L-1)])
eReg.extend([GridQubit(N+n_i,L-1)])
wReg.extend([GridQubit(N+n_i,j) for j in range(L,L+5)])
n_phiReg.extend([GridQubit(N+n_i+1,j) for j in range(L)])
w_phiReg.extend([GridQubit(N+n_i+1,j) for j in range(L,2*L-1)])
n_aReg.extend([GridQubit(N+n_i+2,j) for j in range(L)])
w_aReg.extend([GridQubit(N+n_i+2,j) for j in range(L,2*L-1)])
n_bReg.extend([GridQubit(N+n_i+3,j) for j in range(L)])
w_bReg.extend([GridQubit(N+n_i+3,j) for j in range(L,2*L-1)])

timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList = [], [], [], [], [], [], []
cc.populateParameterLists(N, timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList,
                       Delta_phiList, g_a, g_b, eps)

cc.U_h(circuit, l, n_i, m, n_phiReg, w_phiReg, n_aReg, w_aReg, n_bReg, w_bReg, wReg, eReg, pReg, hReg, w_hReg,
            P_phiList[0], P_aList[0], P_bList[0])

result = simulator.simulate(circuit)


print(result.final_state_vector)

print(circuit)


pReg2, hReg2, w_hReg2, eReg2, wReg2, n_aReg2, w_aReg2, n_bReg2, w_bReg2, n_phiReg2, w_phiReg2 = qcc.allocateQubits(N, n_i, L)

# circuit = QuantumCircuit(pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)
circuit2 = QuantumCircuit(pReg2, hReg2, w_hReg2, eReg2, wReg2, n_aReg2, w_aReg2, n_bReg2, w_bReg2, n_phiReg2)

simulator2 = Aer.get_backend('statevector_simulator')

timeStepList2, P_aList2, P_bList2, P_phiList2, Delta_aList2, Delta_bList2, Delta_phiList2 = [], [], [], [], [], [], []
qcc.populateParameterLists(N, timeStepList2, P_aList2, P_bList2, P_phiList2, Delta_aList2, Delta_bList2,
                       Delta_phiList2, g_a, g_b, eps)

qcc.U_h(circuit2, l, n_i, m, n_phiReg2, w_phiReg2, n_aReg2, w_aReg2, n_bReg2, w_bReg2, wReg2, eReg2, pReg2, hReg2, w_hReg2,
            P_phiList2[0], P_aList2[0], P_bList2[0])

result2 = execute(circuit2, simulator2).result()
statevector2 = result2.get_statevector(circuit2)

print("qiskit statevector: ",statevector2)

print(circuit2)