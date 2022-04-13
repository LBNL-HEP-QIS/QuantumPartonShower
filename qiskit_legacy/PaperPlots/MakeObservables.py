# One "event" is represented as a list of N+n_I particles corresponding to the N steps.  
# The possible symbols for the particles are f1, af1, f2, af2, phi, and 0 -- represented as strings.
# For example, an event might look like [f1, none, none, none] if there is a fermion that does not radiate at all.  
# Another example is [f1, 0, phi, 0] in which a fermion radiates a phi in the second step.

import math
import numpy as np

def Nemissions(myevent, n_I= 1):
	#This function returns the observable shown in Fig. 1b.
	mycount = 0
	for i in range(0, len(myevent) - n_I):
		if myevent[i]!= '0':
			mycount+= 1
			pass
		pass
	return mycount


def LogThetaMax(myevent, n_I= 1, eps= 0.001):
	#This function returns the observable shown in Fig. 1a.
    # Outputs -1 if there is no emission.

    N = len(myevent) - n_I
    firstemmit = -1
    for i in reversed(range(0, len(myevent) - n_I)):
        if myevent[i]!= '0':
            firstemmit= len(myevent) - n_I - i - 1
            break

    hist_bins = [math.log(eps**(j / N)) for j in range(N, -1, -1)]
    hist_bins = np.array(hist_bins)
    centers = (hist_bins[:-1] + hist_bins[1:]) / 2.

    if firstemmit != -1:
        return centers[::-1][firstemmit], centers, hist_bins
    else:
        return None, centers, hist_bins


def hist_bins(ni, N, eps):
	hb = [math.log(eps**(j / N)) for j in range(N, -1, -1)]
	hb = np.array(hb)
	centers = (hb[:-1] + hb[1:]) / 2.

	return hb, centers

if False:
    #Let's run some tests
    #myevent = ['f1', '0', 'phi']
    myevent = ['phi', '0', 'f1']
    n_I = 1
    print("Number of emissions: ", Nemissions(myevent, n_I))
    print("log(theta_max): ", LogThetaMax(myevent, n_I)) #<-- should be the bin center for the second bin in Fig. 1a

    #myevent = ['f1', 'phi', '0']
    myevent = ['0', 'phi', 'f1']
    n_I = 1
    print("Number of emissions: ", Nemissions(myevent, n_I))
    print("log(theta_max): ", LogThetaMax(myevent, n_I)) #<-- should be the bin center for the first bin in Fig. 1a

    #myevent = ['f1', 'phi', 'phi']
    myevent = ['phi', 'phi', 'f1']
    n_I = 1
    print("Number of emissions: ", Nemissions(myevent, n_I))
    print("log(theta_max): ", LogThetaMax(myevent, n_I)) #<-- should be the bin center for the first bin in Fig. 1a

    #myevent = ['f1', '0', 'phi', '0']
    myevent = ['0', 'phi', '0', 'f1']
    n_I = 1
    print("Number of emissions: ", Nemissions(myevent, n_I))
    print("log(theta_max): ", LogThetaMax(myevent, n_I))