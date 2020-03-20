###########################################################
#Run first time with L47 equal to True, and then again when it is False

#These come from 2stepSim.py
fullsimg121 = [0.104236, 0.27143300000000004, 0.609288, 0.015042999999999999]
fullsimg120 = [0.222035, 0.110856, 0.610722, 0.056387]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

#### Some initial parameters ####
eps = .001
gL = 2
gR = 1
gLR = 0.00001 #1.

qbig = 24
nbig = 100000
Nev_classical = 200000

simulations = [qbig, 4, 4, qbig, 4, 4] #24
sims = ["24 step simulation", "4 step simulation", "4 steps IBMQ Tenerife","24 step simulation", "4 step simulation", "4 steps IBMQ Tenerife"]
label = ["24 step simulation ($g_{12} = 0$)", "simulation ($g_{12} = 0$)", "IBMQ ($g_{12} = 0$)","24 step simulation ($g_{12} = 1$)", "simulation ($g_{12} = 1$)", "IBMQ ($g_{12} = 1$)"]
markers = ["o","v","^","s","o","v","^","s"]
gLRvals = [0.00001,0.00001,0.00001,1.,1.,1.]
hist_type = ["bar", "step", "step","bar", "step", "step"]
colors_hist = ["blue", "red", "green","blue", "red", "green","black"]
opacity = [.3, 1, 1,.3, 1, 1]
line_width = [None, 3, 3,None, 3, 3]


#### Define the splitting function and Sudakov factor ####
def P(lnt, g):
   alpha = g**2 / (4 * math.pi)
   return alpha

def Delta(lnt, g):
   alpha = g**2 / (4 * math.pi)
   return math.exp(alpha * lnt)

# The analytical distribution of the hardest emission
def dsigma_d_t_max(lnt, lneps, g):
   return P(lnt, g) * Delta(lnt, g) / (1 - Delta(lneps, g))

if (True):
  from qiskit import QuantumRegister, ClassicalRegister
  from qiskit import QuantumCircuit, execute
  from qiskit.tools.visualization import circuit_drawer
  from qiskit import Aer #, IBMQ
  from qiskit.providers.aer import noise

  import os
  os.environ['KMP_DUPLICATE_LIB_OK']='True'

  #### Get a noise model for an IBMQ machine ####
  # Choose a real device to simulate
  #IBMQ.load_accounts()
  #device = IBMQ.get_backend('ibmq_5_tenerife')
  #properties_IBM = device.properties()
  #coupling_map_IBM = device.configuration().coupling_map
  #noise_model_IBM = noise.device.basic_device_noise_model(properties_IBM)
  #basis_gates_IBM = noise_model_IBM.basis_gates

  backend_sim = Aer.get_backend('qasm_simulator')

  #### PUT THE CORRECT CODE HERE TO GET THE BACKEND FOR THE REAL QUANTUM COMPUTER ####
  backend_device = Aer.get_backend('qasm_simulator')

  ### Define which simulations we want to do ####
  backend = [backend_sim, backend_sim, backend_device,backend_sim, backend_sim, backend_device]
  #simulations = [qbig, 4, 4, qbig, 4, 4] #24
  hist_bins = [qbig, 4, 4, qbig, 4, 4]
  events = [nbig, nbig, nbig,nbig, nbig, nbig] #dropped a zero
  coupling_map = [None, None, None,None, None, None]
  noise_model = [None, None, None,None, None, None]
  basis_gates = [None, None, None,None, None, None]

  #noise_model = [None, noise_model_IBM, None,None, noise_model_IBM, None]
  #basis_gates = [None, basis_gates_IBM, None,None, basis_gates_IBM, None]

  ##### provide map onto physical qubits
  #map = []
  #map_inv = []
  #fermion = 2
  #for sim in simulations:
  #   print(sim)
  #   gate_map = {}
  #   gate_map[0] = fermion
  #   for i in range(1, sim+1):
  #      if i <= fermion:
  #         gate_map[i] = i-1
  #      else:
  #         gate_map[i] = i
  #   gate_map_inv = {v: k for k, v in gate_map.items()}
  #   map.append(gate_map)
  #   map_inv.append(gate_map_inv)
  #   print(gate_map)
  #   print(gate_map_inv)
  #
  #exit()

  #### Define several observables ####

  # The t value of the hardest emission
  '''
  def create_ln_t_max_list(time_steps, counts):
     NT = len(time_steps)
     t_max_list = []
     for key,val in counts.items():
        max_pos = 0
        for i in reversed(range(NT)):
           if key[i] == '1':
              break
           max_pos+= 1
        t_hard = eps / 10
        if (max_pos < NT):
           t_hard = time_steps[max_pos]
        for i in range(val):
           t_max_list.append(math.log(t_hard))
     return t_max_list
  '''

  def create_ln_t_max_list(time_steps, counts, alpha=0):
     NT = len(time_steps)
     t_max_list = []
     for key,val in counts.items():
        max_pos = 0
        if (alpha==0):
          for i in reversed(range(NT)):
            if key[i] == '1':
              max_pos+= 1
              pass
            pass
          pass
        elif (alpha==100): #"infinity"
          for i in reversed(range(NT)):
            if key[i] == '1':
              break
            max_pos+= 1
          t_hard = eps / 10
          if (max_pos < NT):
            t_hard = time_steps[max_pos]
            pass
          max_pos = math.log(t_hard)
          pass
        else:
          for i in reversed(range(NT)):
            if key[i] == '1':
              max_pos += time_steps[i]**alpha
              pass
            pass
          if (max_pos <= 0):
            max_pos = eps**alpha
            pass
          max_pos = math.log(max_pos)
          pass        
        for i in range(val):
           t_max_list.append(max_pos)
           pass
     return t_max_list

  def create_angularity_list(time_steps, counts, alpha = 0):
     NT = len(time_steps)
     angularity_list = []
     for key,val in counts.items():
        mysum = eps / 10.
        for i in reversed(range(NT)):
           if key[i] == '1':
              mysum+=time_steps[i]**alpha
              pass
        for i in range(val):
           angularity_list.append(mysum)
     return angularity_list

  #### Create the circuit ####
  def create_circ(NT, eps, gL, gR, gLR):

     # number of qubits
     N = NT + 1

     # coupling constants
     gp = math.sqrt( abs((gL - gR)**2 + 4 * gLR * gLR ))
     if (gL > gR):
        gp = -gp
     gA = (gL + gR - gp)/2
     gB = (gL + gR + gp)/2

     # compute the u value which we label qS
     qS = math.sqrt(abs((gp + gL - gR)/ (2 * gp)))

     # define necessary lists of paramenters
     time_steps=[]
     pA_list=[]
     pB_list=[]
     deltaA_list=[]
     deltaB_list=[]

     for i in range(NT):

         # Compute time steps
         t_up = eps**((i)/NT)
         t_mid =  eps**((i+0.5)/NT)
         t_low =  eps**((i+1)/NT)
         time_steps.append(t_mid)

         # Compute values for emission matrices
         deltaA = math.sqrt(Delta(math.log(t_low), gA)) / math.sqrt(Delta(math.log(t_up), gA))
         deltaB = math.sqrt(Delta(math.log(t_low), gB)) / math.sqrt(Delta(math.log(t_up), gB))
         pA = math.sqrt(1 - deltaA*deltaA)
         pB = math.sqrt(1 - deltaB*deltaB)

         # Add them to the list
         pA_list.append(pA)
         pB_list.append(pB)
         deltaA_list.append(deltaA)
         deltaB_list.append(deltaB)

     # Compute respective angles for quantum gates

     thetaA = 2*np.arccos(deltaA_list) #angles for UA rotation
     thetaB = 2*np.arccos(deltaB_list) #angles for UB rotation
     phi = 2*np.arcsin(qS)

     #### Simulation of Quantum Circuit ####

     # Create a Quantum Register and Quantum Circuit
     q = QuantumRegister(N, 'q')
     qc = QuantumCircuit(q)

     f = 0
     if (gLR != 0):
        qc.ry(phi,q[f])
     else:
        print ('Running the simplified circuit for gLR = 0')

     for i in range(NT):

         b = i + 1
         #define rotation angles used in RY matrices
         tA = thetaA[i]
         tB = thetaB[i]
         alpha = tA/4
         beta = - tA/2
         gamma = tB/4
         delta = - tB/2

         # circuit block with standard single-qubit gates and CNOTs only
         if(gLR == 0):
            qc.ry(tA,q[b])
         else:
            qc.x(q[f])
            qc.ry(alpha,q[b])
            qc.cx(q[f],q[b])
            qc.ry(beta,q[b])
            qc.cx(q[f],q[b])
            qc.ry(alpha,q[b])
            qc.x(q[f])

            qc.ry(gamma,q[b])
            qc.cx(q[f],q[b])
            qc.ry(delta,q[b])
            qc.cx(q[f],q[b])
            qc.ry(gamma,q[b])


     if (gLR != 0):
        qc.ry(-phi,q[f])

     # Create a Classical Register with N bits.
     c = ClassicalRegister(N, 'c')
     # Introduce measurements
     meas = QuantumCircuit(q, c)
     meas.barrier(q)
     # Map the quantum measurement to the classical bits
     meas.measure(q,c)
     # Combine classical and quantum registers
     circ = qc+meas
     return [circ, time_steps]

  # Run the quantum circuits
  steps = []
  lnt_max_list = {}
  count_set = {}
  for alpha in [0.,100.]:
    lnt_max_list[alpha]=[]
    pass
  #output_file = open("output_gLR_" + str(gLR) + ".dat", 'w')
  for i, sim  in enumerate(simulations):
     print('Running ' + label[i])
     [circ, time_steps] = create_circ(sim, eps, gL, gR, gLRvals[i])
     if "IBMQ" not in sims[i]:
        job = execute(circ, backend[i], shots=events[i],
                          coupling_map=coupling_map[i],
                          noise_model=noise_model[i],
                          basis_gates=basis_gates[i])
        result = job.result()
        counts = result.get_counts(circ)
     # Here are the counts from IBMQ runs
     else:
        if gLRvals[i] == .00001:
           #counts = {'00100': 21387, '10001': 5272, '01101': 12289, '00010': 43032, '00110': 18836, '11001': 7755, '01011': 14055, '10000': 27083, '01010': 21106, '10110': 15808, '00011': 8019, '10101': 7176, '00101': 11520, '01100': 13517, '01001': 17245, '11110': 11329, '10010': 21455, '01000': 25286, '11000': 15078, '00001': 10455, '00000': 53121, '11010': 12281, '01111': 10074, '10100': 18321, '00111': 8673, '11101': 7142, '10011': 4064, '11011': 6266, '01110': 11633, '11111': 5956, '11100': 12697, '10111': 5397}
           counts = {'11110': 4145, '10010': 13532, '00110': 8071, '01010': 9531, '11001': 6636, '00101': 11242, '01111': 2320, '01011': 5651, '00000': 67753, '10111': 1731, '01101': 5271, '11000': 37012, '01100': 17636, '11111': 977, '00111': 6622, '11101': 2630, '10110': 6180, '10001': 10422, '01110': 4312, '01000': 38377, '11100': 17439, '10011': 4269, '01001': 14649, '10100': 25272, '00001': 34273, '10000': 53692, '10101': 3848, '00010': 18290, '00011': 17228, '00100': 31099, '11010': 9104, '11011': 2306}
        elif gLRvals[i] == 1:
           #counts = {'00100': 14755, '10001': 9205, '01101': 14690, '00010': 36043, '00110': 13228, '11001': 10404, '01011': 30202, '10000': 17935, '01010': 16346, '10110': 13630, '00011': 18196, '10101': 10059, '00101': 18517, '01100': 10417, '01001': 30184, '11110': 9732, '10010': 16446, '01000': 17874, '11000': 11823, '00001': 18755, '00000': 37255, '11010': 10766, '01111': 11736, '10100': 15689, '00111': 14662, '11101': 8565, '10011': 8876, '11011': 9595, '01110': 9613, '11111': 7040, '11100': 11081, '10111': 8201}
           counts = {'11110': 9980, '10010': 8830, '00110': 6739, '01010': 7426, '11001': 11040, '00101': 14191, '01111': 4700, '01011': 6108, '00000': 63178, '10111': 3235, '01101': 9975, '11000': 31437, '01100': 15409, '11111': 3907, '00111': 9612, '11101': 9767, '10110': 7461, '10001': 9390, '01110': 5333, '01000': 24557, '11100': 27423, '10011': 4960, '01001': 9770, '10100': 19476, '00001': 54611, '10000': 28359, '10101': 7373, '00010': 16797, '00011': 28782, '00100': 16375, '11010': 10532, '11011': 4787}
     #output_file.write(label[i] + '\n' + str(counts) + '\n')
     #print(counts)
     print("timesteps",time_steps)
     count_set[i] = counts
     for alpha in [0.,100.]:
        lnt_max_list[alpha].append(create_ln_t_max_list(time_steps, counts,alpha))
        pass
     pass
  #output_file.close()

  for i in count_set:
    np.save("quantum_counts_"+str(i)+".npy",count_set[i])
    pass

  #####
  #####
  # Add in the classical as well.
  #####
  #####

  emits_classical = []
  time_steps = []
  NTclassical = 4
  for i in range(NTclassical):

    if (i%100000==0):
      print("classical, on i=",i)
      pass

    # Compute time steps
    t_up = eps**((i)/NTclassical)
    t_mid =  eps**((i+0.5)/NTclassical)
    t_low =  eps**((i+1)/NTclassical)
    time_steps.append(t_mid)

    #Compute the probability to emmit.
    deltaL = math.sqrt(Delta(math.log(t_low), gL)) / math.sqrt(Delta(math.log(t_up), gL))
    pL = 1. - deltaL*deltaL
    #Each timestep is currently completely independent from every other emission so we can do the generation all at once.
    emits_classical+=[np.random.binomial(1,pL,Nev_classical)]
    pass

  #####
  #####
  # Plotting
  #####
  #####

  #Classical
  vals_classical = []
  for j in range(Nev_classical):
    mysum=0.
    for i in range(NTclassical):
      if (emits_classical[i][j]==1):
        mysum+=1
        pass
      pass
    vals_classical+=[mysum]

  #Let's save the data so that making the plots is super fast.
  vals_classical = np.array(vals_classical)
  np.save("vals_classical.npy", vals_classical)

  print(np.shape(vals_classical))
  for lab in range(len(lnt_max_list[0])):
    np.save("quantum_0_"+str(lab)+".npy",np.array(lnt_max_list[0.][lab]))
    np.save("quantum_100_"+str(lab)+".npy",np.array(lnt_max_list[100][lab]))
    print(np.shape(lnt_max_list[0.][lab]))
  exit(1)

lnt_max_list={}
vals_classical = np.load("vals_classical.npy")
lnt_max_list[0]=[]
lnt_max_list[100]=[]
for k in range(6):
  quantum_0 = np.load("quantum_0_"+str(k)+".npy")
  quantum_100 = np.load("quantum_100_"+str(k)+".npy")
  lnt_max_list[0.].append(quantum_0)
  lnt_max_list[100].append(quantum_100)
  pass

mybins2 = [-0.5]
for i in range(7):
  mybins2+=[i+0.5]
  pass

mybins = [-0.5]
for i in range(11):
  mybins+=[i+0.5]
  pass
ns = []
bs = []
bs2 = []
for i in range(len(simulations)-1):
  n,b = np.histogram(lnt_max_list[0][i],bins=mybins2)
  ns+=[n]
  bs+=[np.array([0.5*(b[i]+b[i+1]) for i in range(0,len(b)-1)])]
  bs2+=[np.array([0.5*(b[i+1]-b[i]) for i in range(0,len(b)-1)])]
  pass

n,b = np.histogram(vals_classical,bins=mybins2)
ns+=[n]
bs+=[np.array([0.5*(b[i+1]+b[i]) for i in range(0,len(b)-1)])]
bs2+=[np.array([0.5*(b[i+1]-b[i]) for i in range(0,len(b)-1)])]

f = plt.figure(figsize=(5, 5))

gs = GridSpec(5, 1, width_ratios=[1], height_ratios=[3.5, 1,0.9,3.5,1])
ax1 = plt.subplot(gs[0])
ax1.set_xticklabels( () )
ax1.tick_params(bottom="True",right="False",top="False",left="True",direction='in')

mysum = sum(ns[len(bs)-1])*2*bs2[len(bs)-1][0]
newn = []
for j in range(len(bs[len(bs)-1])):
  newn += [ns[len(bs)-1][j] / (mysum)]
  pass
newn[len(newn)-1]=-999
newn[len(newn)-2]=-999
plt.plot(bs[len(bs)-1],newn, label='Classical MCMC', color = "black", linewidth = 0,marker="x")
#plt.hist(lnt_max_list[0][0], bins=mybins, alpha=0.3, color = 'blue', density = True, label=label[0], histtype = hist_type[0], linewidth = line_width[0])
xx,yy,zz = plt.hist(lnt_max_list[0][1], bins=mybins2, alpha=0.2, color = 'blue', density = True, label=label[1], histtype = hist_type[0], linewidth = line_width[1])
#plt.hist(lnt_max_list[0][3], bins=mybins, alpha=0.3, color = 'red', density = True, label=label[3], histtype = hist_type[3], linewidth = line_width[3])
xx2,yy2,zz2 = plt.hist(lnt_max_list[0][4], bins=mybins2, alpha=0.2, color = 'red', density = True, label=label[4], histtype = hist_type[3], linewidth = line_width[4])
plt.ylim((0, 0.75))

for i in [2]:
   binvals,binEdges = np.histogram(lnt_max_list[0][i],bins=mybins2)
   centers = (binEdges[:-1] + binEdges[1:]) / 2
   widths = (-binEdges[:-1] + binEdges[1:]) 
   hh = binvals/(sum(binvals)*widths)
   shh = np.sqrt(binvals)/sum(binvals)
   hh[len(hh)-1]=-999
   shh[len(shh)-1]=0
   hh[len(hh)-2]=-999
   shh[len(shh)-2]=0
   plt.errorbar(centers,hh , yerr=shh, markersize = 5.,ls='none',ecolor='blue',color='blue',label=label[2],fmt='v')
   pass
for i in [5]:
   binvals,binEdges = np.histogram(lnt_max_list[0][i],bins=mybins2)
   centers = (binEdges[:-1] + binEdges[1:]) / 2
   widths = (-binEdges[:-1] + binEdges[1:]) 
   hh = binvals/(sum(binvals)*widths)
   shh = np.sqrt(binvals)/sum(binvals)
   hh[len(hh)-1]=-999
   shh[len(shh)-1]=0
   hh[len(hh)-2]=-999
   shh[len(shh)-2]=0
   plt.errorbar(centers, hh, yerr=shh, markersize = 5.,ls='none',ecolor='red',color='red',label=label[5],fmt='^')
   pass

nvalsfull = [0,1,2,3,4,5,6]
nvalsg121 = [fullsimg121[1],fullsimg121[0]+fullsimg121[3],fullsimg121[2],0,0,0,0]
nvalsg120 = [fullsimg120[1],fullsimg120[0]+fullsimg120[3],fullsimg120[2],0,0,0,0]

#plt.bar(nvalsfull,nvalsg120,width=1,label=r'2 step sim. with $\phi\rightarrow f\bar{f}$ ($g_{12}=0$)',fill=False,color='black',linestyle=':')
#plt.bar(nvalsfull,nvalsg121,width=1,label=r'2 step sim. with $\phi\rightarrow f\bar{f}$ ($g_{12}=1$)',fill=False,color='black')

plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $dN$')
plt.legend(loc='upper right',prop={'size': 9},frameon=False)
plt.text(-0.5,0.77,r"$(g_{1},g_{2},\epsilon) = ("+str(gL)+","+str(gR)+",10^{-3})$", fontsize=10)
plt.text(-0.5,0.65,r"4 steps",fontsize=10)
plt.text(-0.5,0.52,r"$\phi\rightarrow f\bar{f}$ excluded",fontsize=10)
plt.plot(bs[len(bs)-1],newn, color = "black", linewidth = 0,marker="x",zorder=10)
plt.text(6.5,0.78,r"(b)",fontsize=10)
plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.15)

ax2 = plt.subplot(gs[1])
#ax2.set_xticklabels( () )
ax2.tick_params(bottom="True",right="False",top="False",left="True",direction='in')
xx_rat = [xx[0]/xx2[0]]
for i in range(len(xx)):
  if (xx2[i]>0):
    xx_rat += [xx[i]/xx2[i]]
  else:
    xx_rat += [1.]
    pass
  xx2[i] = 1.
  pass
plt.step([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5],[1.,1.,1.,1.,1.,1.,1.,1.],color='black',linestyle=':')   
yycenters = (yy[:-1] + yy[1:]) / 2
plt.step(yy[0:len(yy)-2],xx_rat[0:len(yy)-2],color='black')
yycenters2 = [yy[0],yy[len(yy)-1]]
xx2 = [1.,1.]
#plt.plot(yycenters2,xx2,color='black',linestyle=':')
plt.ylim((0, 3.))
plt.ylabel('Classical /\n Quantum',labelpad = 19)

plt.xlabel(r'Number of emissions (N)')
gs.update(wspace=0.025, hspace=0.07)

ax1 = plt.subplot(gs[3])
ax1.set_xticklabels( () )
ax1.tick_params(bottom="True",right="False",top="False",left="True",direction='in')
plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $dN$')
plt.bar(nvalsfull,nvalsg120,width=1,label=r'simulation ($g_{12}=0$)',fill=False,color='black',linestyle=':')
plt.bar(nvalsfull,nvalsg121,width=1,label=r'simulation ($g_{12}=1$)',fill=False,color='black')
plt.legend(loc='upper right',prop={'size': 9.},frameon=False)
plt.text(2.8,0.38-0.04,r"2 steps",fontsize=10)
plt.text(6.5,0.67,r"(d)",fontsize=10)
plt.text(2.8,0.28-0.04,r"$\phi\rightarrow f\bar{f}$ included",fontsize=10)

ax1 = plt.subplot(gs[4])
ax1.tick_params(bottom="True",right="False",top="False",left="True",direction='in')

xx = nvalsg120
xx2 = nvalsg121
yycenters = nvalsfull
xx_rat = [xx[0]/xx2[0]]
for i in range(len(xx)):
  if (xx2[i] > 0):
    xx_rat+=[xx[i]/xx2[i]]
  else:
    xx_rat+=[1.]
  pass
plt.step([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5],[1.,1.,1.,1.,1.,1.,1.,1.],color='black',linestyle=':')  
plt.step([-0.5,0.5,1.5,2.5],xx_rat[0:len(xx_rat)-4],color='black')
yycenters2 = [yy[0],yy[len(yy)-1]]
plt.ylabel('Classical /\n Quantum',labelpad = 19)
xx2 = [1.,1.]
#plt.plot([-0.5,4.5],xx2,color='black',linestyle=':')
plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.1)
plt.ylim((0, 3.))
plt.xlabel(r'Number of emissions (N)')

gs.update(wspace=0.025, hspace=0.1)

f.savefig("test_alpha"+str(0)+str(int(gLR))+".pdf") 
#====

for aa in [100.]:
  f = plt.figure(figsize=(5, 5))

  gs = GridSpec(5, 1, width_ratios=[1], height_ratios=[3.5, 1.,0.9,3.5,1.])
  ax1 = plt.subplot(gs[0])
  #ax1.tick_params(axis='x',length=0.)
  ax1.set_xticklabels( () )
  ax1.tick_params(bottom="True",right="False",top="False",left="True",direction='in')

  # First the analytic plot
  num_points_ana = 10000
  lnxList = [math.log(eps**((i+0.5) / num_points_ana)) for i in range(0, num_points_ana)]
  yLList = [dsigma_d_t_max(lnx, math.log(eps), gL) for lnx in lnxList]
  plt.plot(lnxList, yLList, label=r'Analytical ($g_{12}=0$)', color = "black", linewidth = 1,linestyle='--')

  num_bins_hist = simulations[0]
  bin_size_hist = 1
  hist_bins = [math.log(eps**(j / num_bins_hist)) for j in range(num_bins_hist, -1, -1)]

  num_bins_hist = simulations[1]
  bin_size_hist = 1
  hist_bins2 = [math.log(eps**(j / num_bins_hist)) for j in range(num_bins_hist, -1, -1)]

  #xx,yy,zz = plt.hist(lnt_max_list[aa][0], bins=hist_bins, alpha=0.3, color = 'blue', density = True, label=label[0], histtype = hist_type[0], linewidth = line_width[0])
  #xx,yy,zz = plt.hist(lnt_max_list[aa][1], bins=hist_bins2, alpha=1., color = 'blue', density = True, label=label[1], histtype = hist_type[1], linewidth = line_width[1])
  #xx2,yy2,zz2 = plt.hist(lnt_max_list[aa][3], bins=hist_bins, alpha=0.3, color = 'red', density = True, label=label[3], histtype = hist_type[3], linewidth = line_width[3])
  #xx2,yy2,zz2 = plt.hist(lnt_max_list[aa][4], bins=hist_bins2, alpha=1., color = 'red', density = True, label=label[4], histtype = hist_type[4], linewidth = line_width[4],linestyle='--')

  xx,yy,zz = plt.hist(lnt_max_list[aa][1], bins=hist_bins2, alpha=0.2, color = 'blue', density = True, label=label[1], histtype = hist_type[0], linewidth = line_width[1])
  xx2,yy2,zz2 = plt.hist(lnt_max_list[aa][4], bins=hist_bins2, alpha=0.2, color = 'red', density = True, label=label[4], histtype = hist_type[3], linewidth = line_width[4])
  plt.ylim((0, 0.65))

  for i in [2]:
     num_bins_hist = simulations[i]
     bin_size_hist = 1
     hist_bins = [math.log(eps**(j / num_bins_hist)) for j in range(num_bins_hist, -1, -1)]
     binvals,binEdges = np.histogram(lnt_max_list[aa][i],hist_bins)
     centers = (binEdges[:-1] + binEdges[1:]) / 2
     widths = (-binEdges[:-1] + binEdges[1:]) 
     plt.errorbar(centers, binvals/(sum(binvals)*widths), yerr=np.sqrt(binvals)/sum(binvals), markersize = 5.,ls='none',ecolor='blue',color='blue',label=label[2],fmt='v')
     pass
  for i in [5]:
     num_bins_hist = simulations[i]
     bin_size_hist = 1
     hist_bins = [math.log(eps**(j / num_bins_hist)) for j in range(num_bins_hist, -1, -1)]
     binvals,binEdges = np.histogram(lnt_max_list[aa][i],hist_bins)
     centers = (binEdges[:-1] + binEdges[1:]) / 2
     widths = (-binEdges[:-1] + binEdges[1:]) 
     plt.errorbar(centers, binvals/(sum(binvals)*widths), yerr=np.sqrt(binvals)/sum(binvals), markersize = 5.,ls='none',ecolor='red',color='red',label=label[5],fmt='^')
     pass

  num_bins_hist = 2
  bin_size_hist = 1
  hist_bins3 = [math.log(eps**(j / num_bins_hist)) for j in range(num_bins_hist, -1, -1)]
  hist_bins3 = np.array(hist_bins3)
  centers3 = (hist_bins3[:-1] + hist_bins3[1:]) / 2
  widths3 = (-hist_bins3[:-1] + hist_bins3[1:]) 

  valsg121 = [fullsimg121[0],fullsimg121[2]+fullsimg121[3]]
  valsg120 = [fullsimg120[0],fullsimg120[2]+fullsimg120[3]]

  #plt.bar(centers3,valsg120/(sum(valsg120)*widths3),width=widths3,label=r'2 step sim. with $\phi\rightarrow f\bar{f}$ ($g_{12}=0$)',fill=False,color='black',linestyle=':')
  #plt.bar(centers3,valsg121/(sum(valsg121)*widths3),width=widths3,label=r'2 step sim. with $\phi\rightarrow f\bar{f}$ ($g_{12}=1$)',fill=False,color='black')

  # Add some labels
  plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $d\log(\theta_{max})$')
 
  plt.legend(loc='upper left',prop={'size': 9.},frameon=False)
  plt.text(-7,0.67,r"$(g_{1},g_{2},\epsilon) = ("+str(gL)+","+str(gR)+",10^{-3})$", fontsize=10)
  plt.text(-1,0.55,r"4 steps",fontsize=10)
  plt.text(-2.2,0.45,r"$\phi\rightarrow f\bar{f}$ excluded",fontsize=10)
  plt.text(0.,0.68,r"(a)",fontsize=10)
  plt.subplots_adjust(left=0.15,right=0.9,top=0.9,bottom=0.15)

  ax2 = plt.subplot(gs[1])
  ax2.tick_params(bottom="True",right="False",top="False",left="True",direction='in',labelsize=10)

  yycenters = yy
  xx_rat = [xx[0]/xx2[0]]
  for i in range(len(xx)):
    xx_rat+=[xx[i]/xx2[i]]
    pass
  plt.step(yycenters,xx_rat,color='black')
  yycenters2 = [yy[0],yy[len(yy)-1]]
  xx2 = [1.,1.]
  plt.plot(yycenters2,xx2,color='black',linestyle=':')
  plt.ylim((0, 3.))

  plt.ylabel('Classical /\n Quantum',labelpad = 19)
  plt.xlabel(r'$\log(\theta_{max})$')

  ax1 = plt.subplot(gs[3])
  ax1.set_xticklabels( () )
  ax1.tick_params(bottom="True",right="False",top="False",left="True",direction='in')
  plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $d\log(\theta_{max})$')
  plt.bar(centers3,valsg120/(sum(valsg120)*widths3),width=widths3,label=r'simulation ($g_{12}=0$)',fill=False,color='black',linestyle=':')
  plt.bar(centers3,valsg121/(sum(valsg121)*widths3),width=widths3,label=r'simulation ($g_{12}=1$)',fill=False,color='black')
  plt.legend(loc='upper left',prop={'size': 9.},frameon=False)
  plt.text(-7,0.145,r"2 steps",fontsize=10)
  plt.text(-7,0.105,r"$\phi\rightarrow f\bar{f}$ included",fontsize=10)
  plt.text(0.,0.27,r"(c)",fontsize=10)
  ax1 = plt.subplot(gs[4])
  ax1.tick_params(bottom="True",right="False",top="False",left="True",direction='in')

  xx = valsg120/(sum(valsg120)*widths3)
  xx2 = valsg121/(sum(valsg121)*widths3)
  yycenters = hist_bins3
  xx_rat = [xx[0]/xx2[0]]
  for i in range(len(xx)):
    xx_rat+=[xx[i]/xx2[i]]
    pass
  plt.step(yycenters,xx_rat,color='black')
  yycenters2 = [yy[0],yy[len(yy)-1]]
  plt.ylabel('Classical /\n Quantum',labelpad = 19)
  xx2 = [1.,1.]
  plt.plot(yycenters2,xx2,color='black',linestyle=':')
  plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.1)
  plt.ylim((0, 3.))
  plt.xlabel(r'$\log(\theta_{max})$')

  gs.update(wspace=0.025, hspace=0.1)

  f.savefig("test_alphaINF_"+str(int(gLR))+".pdf") # , bbox_inches=1)
