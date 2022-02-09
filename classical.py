import numpy as np
import math

def MCMC(eps, g_a, g_b, na_i, nb_i, N, verbose=False):
    '''
    na_i, nb_i are the initial number of a and b fermions.

    '''

    n_a= na_i
    n_b= nb_i
    n_phi= 0

    n_emits= 0

    for i in range(N):
        # Compute time steps
        t_up = eps ** ((i) / N)
        t_mid = eps ** ((i + 0.5) / N)
        t_low = eps ** ((i + 1) / N)
        # Compute values for emission matrices
        Delta_a = Delta_f(t_low, g_a) / Delta_f(t_up, g_a)
        Delta_b = Delta_f(t_low, g_b) / Delta_f(t_up, g_b)
        Delta_phi = Delta_bos(t_low, g_a, g_b) / Delta_bos(t_up, g_a, g_b)
        P_a, P_b, P_phi = P_f(t_mid, g_a), P_f(t_mid, g_b), P_bos(t_mid, g_a, g_b)

        P_phi_a= P_bos_g(t_mid, g_a)
        P_phi_b= P_bos_g(t_mid, g_b)

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

def P(g):
    alpha = g**2 / (4 * math.pi)
    return alpha

def Delta(lnt, g):
    alpha = g**2 / (4 * math.pi)
    return math.exp(alpha * lnt)


def P_f(t, g):
    alpha = g ** 2 * Phat_f(t) / (4 * math.pi)
    return alpha

def Phat_f(t):
    return math.log(t)

def Phat_bos(t):
    return math.log(t)

def Delta_f(t, g):
    return math.exp(P_f(t, g))

def P_bos(t, g_a, g_b):
    alpha = g_a ** 2 *Phat_bos(t) / (4 * math.pi) + g_b ** 2 * Phat_bos(t) / (4 * math.pi)
    return alpha

def P_bos_g(t, g):
    return g ** 2 *Phat_bos(t) / (4 * math.pi)

def Delta_bos(t, g_a, g_b):
    return math.exp(P_bos(t, g_a, g_b))


# The analytical distribution of the hardest emission
def dsigma_d_t_max(lnt, lneps, g, normalized=False):
    if normalized: # Normalized to -log(θmax), i.e. "conditionally" normalized on emission occuring
        return P(g) * Delta(lnt, g) / (1 - Delta(lneps, g))
    else: # Normalized to -infinity, i.e. this gives the actual probabilities --> use this for plotting
        return P(g) * Delta(lnt, g)