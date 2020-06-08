# One "event" is represented as a list of N+n_I particles corresponding to the N steps.  
# The possible symbols for the particles are f_1, \bar{f}_1, f_2, \bar{f}_2, \phi, and none.
# For example, an event might look like [f_1, none, none, none] if there is a fermion that does not radiate at all.  
# Another example is [f_1, none, phi, none] in which a fermion radiates a phi in the second step.

import math
import numpy as np

def Nemissions(myevent,n_I=1):
	#This function returns the observable shown in Fig. 1b.
	mycount = 0
	for i in range(n_I,len(myevent)):
		if myevent[i]!='none':
			mycount+=1
			pass
		pass
	return mycount

def LogThetaMax(myevent,n_I=1,eps=0.001):
	#This function returns the observable shown in Fig. 1a.

	N = len(myevent)-n_I
	firstemmit = 0
	for i in range(n_I,len(myevent)):
		if myevent[i]!='none':
			firstemmit = i-n_I
			break

	hist_bins = [math.log(eps**(j / N)) for j in range(N, -1, -1)]
	hist_bins = np.array(hist_bins)
	centers = (hist_bins[:-1] + hist_bins[1:]) / 2.

	return centers[::-1][firstemmit]

#Let's run some tests
myevent = ['f_1','none','phi']
n_I = 1
print("Number of emissions:",Nemissions(myevent,n_I))
print("log(theta_max):",LogThetaMax(myevent,n_I)) #<-- should be the bin center for the second bin in Fig. 1a

myevent = ['f_1','phi','none']
n_I = 1
print("Number of emissions:",Nemissions(myevent,n_I))
print("log(theta_max):",LogThetaMax(myevent,n_I)) #<-- should be the bin center for the first bin in Fig. 1a

myevent = ['f_1','phi','phi']
n_I = 1
print("Number of emissions:",Nemissions(myevent,n_I))
print("log(theta_max):",LogThetaMax(myevent,n_I)) #<-- should be the bin center for the first bin in Fig. 1a

myevent = ['f_1','none','phi','none']
n_I = 1
print("Number of emissions:",Nemissions(myevent,n_I))
print("log(theta_max):",LogThetaMax(myevent,n_I))
