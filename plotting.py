import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib
import math

from PaperPlots import MakeObservables as mo

def ptype(x):
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


def bar_plot2(counts, events, eps, g1, g2, counts2= None, save=True, wReg=True):

    width= 0.2
    firstisf1_y = np.zeros(6)
    firstisf1_ey = np.zeros(6)
    firstisf1_x = np.arange(1 - width, 7 - width, 1)

    firstisf2_y = np.zeros(6)
    firstisf2_ey = np.zeros(6)
    firstisf2_x = np.arange(1, 7, 1)

    if counts2 != None:
        firstisf1b_y = np.zeros(6)
        firstisf1b_ey = np.zeros(6)
        firstisf1b_x = np.arange(1 + width, 7 + width, 1)

        #firstisf2b_x = np.zeros(6)
        #firstisf2b_y = np.zeros(6)
        #firstisf2b_ey = np.arange(1, 7, 1)

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

    mycounter = 0
    for c in counts:
        print(mycounter, c, ptype(c.split()[-3 - wReg]), ptype(c.split()[-2 - wReg]), ptype(c.split()[-1 - wReg]), counts[c])
        mycounter+=1

        x= mymap[ptype(c.split()[-3 - wReg]), ptype(c.split()[-2 - wReg])] - 1
        if (ptype(c.split()[-1 - wReg])=='f1'):
            firstisf1_y[x]+= 100*counts[c]/events
            firstisf1_ey[x]+= 100*counts[c]**0.5/events
            pass
        if (ptype(c.split()[-1 - wReg])=='f2'):
            firstisf2_y[x]+= 100*counts[c]/events
            firstisf2_ey[x]+= 100*counts[c]**0.5/events
            pass

    if counts2 != None:
        for c in counts2:
            x= mymap[ptype(c.split()[-3 - wReg]), ptype(c.split()[-2 - wReg])] - 1
            if (ptype(c.split()[-1 - wReg])=='f1'):
                firstisf1b_y[x]+= 100*counts2[c]/events
                firstisf1b_ey[x]+= 100*counts2[c]**0.5/events
                #firstisf1b_x+=[0.2+mymap[ptype(c.split()[-4]),ptype(c.split()[-3])]]
                pass
            if (ptype(c.split()[-1 - wReg])=='f2'):
                firstisf2b_y[x]+= 100*counts2[c]/events
                firstisf2b_ey[x]+= 100*counts2[c]**0.5/events
                #firstisf2b_x+=[0.2+mymap[ptype(c.split()[-4]),ptype(c.split()[-3])]]
                pass



    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot(1, 1, 1)
    plt.ylim((100*1e-4, 100*5.))
    plt.xlim((1 - 3*width, 6 + 2*width))
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylabel('Probability [%]', size=20)
    bar1 = plt.bar(firstisf1_x, firstisf1_y, color='#228b22', width=width, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
    bar1b = plt.bar(firstisf2_x, firstisf2_y, color='#9AEE9A', width=width, label=r"$f' = f_{2}, g_{12}= 1$", hatch='//') #,yerr=firstisf2_ey)
    #n, bins, patches= plt.hist(firstisf1_x, weights=firstisf1_y, bins=np.arange(0.3, 6.31, 1), align= 'mid', color='#228b22', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
    #n, bins, patches= plt.hist(firstisf1_x, weights=firstisf1_y, bins=6, range=[0.3, 6.3], align= 'mid', color='#228b22', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
    #plt.hist(firstisf2_x, weights=firstisf2_y, bins=6, range=[0.5, 6.5], align= 'mid', color='#01B9FF', width=0.2, label=r"$f' = f_{1}, g_{12}= 1$", hatch='//')
    #print(n, bins, patches)
    ax.set_xticks(np.arange(1, 7, 1))
    ax.set_xticklabels( (r"$f_{1}\rightarrow f'$", r"$f_{1}\rightarrow f'\phi$", r"$f_{1}\rightarrow f'\phi\phi$",r"$f_{1}\rightarrow f' f_{1} \bar{f}_{1}$",r"$f_{1}\rightarrow f' f_{2} \bar{f}_{2}$",r"$f_{1}\rightarrow f' f_{1/2} \bar{f}_{2/1}$") )

    if counts2 != None:
        bar2 = plt.bar(firstisf1b_x, firstisf1b_y, color='#01B9FF', width=width, label=r"$f' = f_{1}, g_{12}= 0$", alpha=1.0) #,hatch="//")
        #plt.hist(firstisf1b_x, weights=firstisf1b_y, bins=6, range=[0.7, 6.7], align= 'mid', color='#FF4949', width=0.2, label=r"$f' = f_{1}, g_{12}= 0$", alpha=1.0) #,hatch="//")
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
    plt.text(0.6, 220, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=16)

    if save == True:
        f.savefig("sim2step_states_shots=%d.pdf" %(events))
    plt.show()







def bar_plot3(counts, events, eps, g1, g2, initialParticles, counts2= None, save=True):

    width= 0.2
    y_f1 = np.zeros(10)
    ey_f1 = np.zeros(10)
    x_f1 = np.arange(1 - width, 11 - width, 1)

    y_f2 = np.zeros(10)
    ey_f2 = np.zeros(10)
    x_f2 = np.arange(1, 11, 1)

    if counts2 != None:
        y2_f1 = np.zeros(10)
        ey2_f1 = np.zeros(10)
        x2_f1 = np.arange(1 + width, 11 + width, 1)

        #y2_f2 = np.zeros(10)
        #ey2_f2 = np.zeros(10)
        #x2_f2 = []


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


    #mycounter = 0
    for c in counts:
        x= mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])] - 1 # - 1 for zero-indexing
        pList= list((ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), ptype(c.split()[8])))
        #print(mycounter, c, pList, counts[c])
        if (ptype(c.split()[-2])=='f1'):
            y_f1[x]+= 100*counts[c]/events
            ey_f1[x]+= 100*counts[c]**0.5/events
            #x_f1+= [-1.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
            pass
        if (ptype(c.split()[-2])=='f2'):
            y_f2[x]+= 100*counts[c]/events
            ey_f2[x]+= 100*counts[c]**0.5/events
            #x_f2+= [-0.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
            pass
        pass            

    if counts2 != 0:
        for c in counts2:
            x= mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])] - 1
            pList= list((ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), ptype(c.split()[8])))
            #print(mycounter, c, pList, counts[c])
            if (ptype(c.split()[-2])=='f1'):
                y2_f1[x]+= 100*counts2[c]/events
                ey2_f1[x]+= 100*counts2[c]**0.5/events
                #x2_f1+= [0.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
                pass
            #if (ptype(c.split()[-2])=='f2'):
                #y2_f2+= [100*counts2[c]/events]
                #ey2_f2+= [100*counts2[c]**0.5/events]
                #x2_f2+= [1.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
                #pass
            pass


    f = plt.figure(figsize=(14, 10))
    ax = f.add_subplot(1, 1, 1)
    plt.ylim((100*1e-4, 100*5.))
    plt.xlim((1 - 3*width, 10 + 2*width))
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylabel('Probability [%]', size= 24)
    bar1_f1 = plt.bar(x_f1, y_f1, color='#228b22', width=width, label=r"$g_{12}= 1, f'= f_{1}$", hatch='//') #,yerr=firstisf1_ey)
    bar1_f2 = plt.bar(x_f2, y_f2, color='#9AEE9A', width=width, label=r"$g_{12}= 1, f'= f_{2}$", hatch='\\') #,yerr=firstisf1_ey)
    #plt.hist(x_f1, weights=y_f1, bins= np.arange(6-1.5*offset, 72-1.5*offset, 6), align= 'mid', color='#228b22', width=offset, label=r"$f' = f_{1}, g_{12}= 1$", hatch='//')
    #plt.hist(x_f2, weights=y_f2, bins= np.arange(6-0.5*offset, 72-0.5*offset, 6), align= 'mid', color='#9AEE9A', width=offset, label=r"$f' = f_{2}, g_{12}= 1$", hatch='\\')

    if counts2!= 0:
        bar2_f1 = plt.bar(x2_f1, y2_f1, color='#01B9FF', width=width, label=r"$g_{12}= 0, f'= f_{1}$")
        #bar2_f2 = plt.bar(x2_f2, y2_f2, color='#C0EDFE', width=offset, label=r"$g_{12}= 0, f'= f_{2}$")
        #plt.hist(x2_f1, weights=y2_f1, bins= np.arange(6+0.5*offset, 72+0.5*offset, 6), align= 'mid', color='#01B9FF', width=offset, label=r"$f' = f_{1}, g_{12}= 0$")
        #leg2 = ax.legend([bar1_f1, bar1_f2],[r'$f = f_{a}$',r'$f = f_{b}$'], loc='upper right', frameon=False, prop={'size': 12.5}, bbox_to_anchor=(1.,0.8))

    pmap= {'f1': r'$f_{1}$', 'af1': r'$f_{1}$', 'f2': r'$f_{2}$', 'af2': r'$f_{2}$', 'phi': r'$\phi$'}
    iP_str= ''
    for j in range(len(initialParticles)):
        iP_str+= pmap[ptype(initialParticles[j])]
        if j > 0: iP_str+= ', '

    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels((iP_str + r"$\rightarrow f'$", iP_str + r"$\rightarrow f'\phi$", iP_str + r"$\rightarrow f'\phi\phi$", iP_str + r"$\rightarrow f'\phi\phi\phi$",
                        iP_str + r"$\rightarrow f' f_{1} \bar{f}_{1}$", iP_str + r"$\rightarrow f' f_{2} \bar{f}_{2}$", iP_str + r"$\rightarrow f' f_{1/2} \bar{f}_{2/1}$",
                        iP_str + r"$\rightarrow f' f_{1} \bar{f}_{1} \phi$", iP_str + r"$\rightarrow f' f_{2} \bar{f}_{2} \phi$", iP_str + r"$\rightarrow f' f_{1/2} \bar{f}_{2/1} \phi$"), size= 10)
    plt.xlabel('Final State', size=24)

    plt.legend(loc='upper right',prop={'size': 20})



    #plt.text(-0.3, 55*4, r"3-step Full Quantum Simulation", fontsize=24)
    plt.title(r"3-step Full Quantum Simulation", fontsize=28)
    plt.text(2.8, 200, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=20)

    plt.text(2.8, 100, r"Initial: " + iP_str, fontsize=20)

    if save == True:
        f.savefig("sim3step_states_shots=%d.pdf" %(events))
    plt.show()















def bar_plot_thetamax(counts, events, eps, g1, g2, N, ni, counts2= None, save=True):
    # Plots counts vs. number of emissions

    # Can handle two different counts: counts and counts2
    hist_bins, centers= mo.hist_bins(ni, N, eps)
    width= (hist_bins[1] - hist_bins[0]) / 2.5

    tm_x= centers - width/2. # tm= theta max
    tm_ey= np.zeros(N)
    tm_y= np.zeros(N)

    for c in counts:
        emit_list= []
        for n in range(0, N + ni):
            emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

        #print(c, ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), counts[c])

        theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
        j= -1
        while hist_bins[j+1] < theta_max:
            j+= 1

        tm_y[j]+= 100*counts[c]/events
        tm_ey[j]+= 100*counts[c]**0.5/events
        #tm_x+= [-offset + theta_max]

    
    if counts2 != None:
    
        tm2_x= centers + width/2.
        tm2_ey= np.zeros(N)
        tm2_y= np.zeros(N)
    
        for c in counts2:
            emit_list= []
            for n in range(0, N + ni):
                emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

            theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
            j= -1
            while hist_bins[j+1] < theta_max:
                j+= 1

            tm2_y[j]+= 100*counts2[c]/events
            tm2_ey[j]+= 100*counts2[c]**0.5/events
            #tm2_x+= [offset + theta_max]

    #    print(sum(tm2_y))
    print(sum(tm2_y))
    print(tm2_y)
    #print(tm_x)
    #print(tm2_x)
    #print(centers)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot(1, 1, 1)
    #plt.ylim((0, 50))
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.xlim((hist_bins[0]-width, hist_bins[-1]+width))
    #ax.set_yscale("log", nonposy='clip')
    ax.set_ylabel('Probability [%]', fontsize=24, fontname= 'times new roman')
    bar1 = plt.bar(tm_x, tm_y, color='#228b22', width= width, label=r"$g_{12} = 1$") #,yerr=firstisf1_ey)

    plt.xticks(size= 18, fontname= 'times new roman')
    plt.yticks(size= 18, fontname= 'times new roman')

    if counts2 != None:
        bar2 = plt.bar(tm2_x, tm2_y, width= width, color='#FF4949', label=r"$g_{12} = 0$", alpha=1.0) #,hatch="//")

    plt.legend(loc='upper left', prop={'size': 16})
    plt.title(r"%d-step Full Quantum Simulation" %(N), fontsize=30, fontname= 'times new roman', pad= 10)
    plt.text(0.1, 0.7, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=18, transform=ax.transAxes)

    plt.xlabel(r'$\log(\theta_{\max})$', fontsize=24, fontname= 'times new roman')
    if save == True:
        f.savefig("sim%dstep_θmax_shots=%d.pdf" %(N, events))
    plt.show()













def bar_plot_emissions(counts, events, eps, g1, g2, N, ni, counts2= None, save=True, wReg=True):
        # Plots counts vs. number of emissions

        # Can handle two different counts: counts and counts2
        width= 0.4

        emissions_x= np.arange(0, N+4, 1) - width/2.
        emissions_ey= np.zeros(N+4)
        emissions_y= np.zeros(N+4)

        for c in counts:
            emit_list= []
            for n in range(N + ni):
                #print(c.split()[-1-ni-n])
                emit_list= [ptype(c.split()[-1 - wReg - n])] + emit_list

            #print(c, ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), counts[c])

            emit_N= int(mo.Nemissions(emit_list, n_I= ni))
            emissions_y[emit_N]+= 100*counts[c]/events
            emissions_ey[emit_N]+= 100*counts[c]**0.5/events
            #emissions_x+= [-offset + emit_N]

        
        if counts2 != None:
        
            emissions2_x= np.arange(0, N+4, 1) + width/2.
            emissions2_ey= np.zeros(N+4)
            emissions2_y= np.zeros(N+4)

            for c in counts2:
                emit_list= []
                for n in range(N + ni):
                    emit_list= [ptype(c.split()[-1 - wReg -n])] + emit_list

                #print(c, ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), counts2[c])
                #print(emit_list)

                emit_N= int(mo.Nemissions(emit_list, n_I= ni))

                emissions2_y[emit_N]+= 100*counts2[c]/events
                emissions2_ey[emit_N]+= 100*counts2[c]**0.5/events
                #emissions2_x+= [offset + emit_N]
    
            #print(sum(emissions2_y))
        #print(sum(emissions_y))
        #print(emissions_y)
        #print(emissions_x)

        f = plt.figure(figsize=(10, 8))
        ax = f.add_subplot(1, 1, 1)
        plt.xlim((-2*width, N + 2*width))
        ax.set_ylabel('Probability [%]', fontsize=24)
        bar1 = plt.bar(emissions_x, emissions_y, color='#228b22', width=width, label=r"$g_{12} = 1$") #,yerr=firstisf1_ey)

        plt.xticks(np.arange(0, N + 1), size= 18)

        if counts2 != None:
            bar2 = plt.bar(emissions2_x, emissions2_y, color='blue', width=width, label=r"$g_{12} = 0$", alpha=0.5)

        plt.legend(loc='upper left', prop={'size': 16})
        plt.title(r"%d-step Full Quantum Simulation" %(N), fontsize=24)
        plt.text(0.03, 0.7, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=18, transform=ax.transAxes)

        plt.xlabel('Number of emissions', fontsize=24)
        if save == True:
            f.savefig("sim%dstep_emissions_shots=%d.pdf" %(N, events))
        plt.show()








def master_plot_phisplit_emissions(g1, g2, N, ni, events, counts, counts2, mcmc= None, save=True):

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
    colors_hist = ["blue", "red", "green","blue", "red", "green", "black"]
    opacity = [.3, 1, 1,.3, 1, 1]
    line_width = [None, 3, 3,None, 3, 3]

    f = plt.figure(figsize=(10, 7))

    #gs = GridSpec(5, 1, width_ratios=[1], height_ratios=[3.5, 1, 0.9, 3.5, 1])
    gs = GridSpec(3, 1, width_ratios=[1], height_ratios=[4, 1, 1])
    ax1 = plt.subplot(gs[0])
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

    label_format = '{:,.1f}'
    ax1.set_yticks(np.arange(0, 0.7, 0.1))
    ax1.set_yticklabels([label_format.format(x) for x in np.arange(0, 0.7, 0.1)], size=16, fontname= 'times new roman')

    ######################    
    # simulation         #
    ######################
    emissions_y= np.zeros(N+1)
    emissions_ey= np.zeros(N+1)
    for c in counts:
        emit_list= []
        for n in range(N + ni):
            emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

        emit_N= int(mo.Nemissions(emit_list, n_I= ni))

        emissions_y[emit_N]+= counts[c]/events
        emissions_ey[emit_N]+= counts[c]**0.5/events

    ax1.bar(np.arange(0, N+1, 1), emissions_y, alpha=0.2, color = 'red', width= 1., label=label[4], tick_label= ['']*(N+1))

    emissions2_y= np.zeros(N+1)
    emissions2_ey= np.zeros(N+1)
    for c in counts2:
        emit_list= []
        for n in range(N + ni):
            emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

        emit_N= int(mo.Nemissions(emit_list, n_I= ni))

        emissions2_y[emit_N]+= counts2[c]/events
        emissions2_ey[emit_N]+= counts2[c]**0.5/events

    print(emissions_y)
    print(emissions2_y)
    print(np.arange(0, N+1, 1))
    plt.bar(np.arange(0, N+1, 1), emissions2_y, alpha=0.2, color = 'blue', width= 1., label=label[1], tick_label= ['']*(N+1))

    try:
        plt.scatter(np.arange(0, N+1, 1), mcmc, color= 'black', label=r'MCMC ($g_{12} = 0$)', marker='+', s=150)
    except:
        print('Invalid array input for MCMC.')

    plt.ylim((0, 0.65))

    font = matplotlib.font_manager.FontProperties(family='times new roman', size=16)

    plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $dN$', labelpad = 18, fontsize= 18)
    #plt.legend(loc='upper right', prop=font, frameon=False)
    plt.legend(loc='best', prop=font, frameon=False)
    #plt.text(-0.5, 0.77, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=10)
    #plt.text(-0.5, 0.65, r"4 steps",fontsize=10)
    #plt.text(-0.5, 0.52, r"$\phi\rightarrow f\bar{f}$ excluded", fontsize=10)
    #plt.plot(bs[len(bs)-1], newn, color = "black", linewidth = 0, marker="x", zorder=10)
    #plt.text(6.5, 0.78, r"(b)", fontsize=10)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    #ax1.set_xlabel('Number of emissions (N)', fontsize= 16, labelpad = 19, fontname= 'times new roman')

    ######################
    # ratio plot upper   #
    ######################
    ax2 = plt.subplot(gs[1])
    ax2.tick_params(bottom="True", right="False", top="False", left="True", direction='in')
    
    #ax2.get_xaxis().set_label_coords(0, N+1)
    label_format = '{:,.1f}'
    ax2.set_xticks(np.arange(-1, N+1, 1))
    ax2.set_yticks(np.arange(0, 3.5, 2))
    ax2.set_xticklabels(np.arange(-1, N+1, 1), size=20, fontname= 'times new roman')
    ax2.set_yticklabels([label_format.format(x) for x in np.arange(0, 3.5, 2)], size=16, fontname= 'times new roman')

    plt.step(np.arange(-0.5, N+0.5, 1), [1.]*(N+1), color='black', linestyle=':')
    ratio= emissions2_y / emissions_y
    plt.step(np.arange(-0.5, N+1.5, 1), np.concatenate((ratio[:1], ratio)), color='black')

    plt.ylim((0, 3.5))
    plt.ylabel('Classical /\n Quantum', fontsize= 18, labelpad = 18, fontname= 'times new roman')
    plt.xlabel('Number of emissions (N)', fontsize= 22, labelpad = 18, fontname= 'times new roman')
    #ax2.set_xlabel('Number of emissions (N)', fontsize= 16, labelpad = 19, fontname= 'times new roman')

    gs.update(wspace=0.025, hspace=0.07)
    gs.update(wspace=0.025, hspace=0.1)

    if save == True:
        plt.savefig('sim%dstep_emissions_shots=%d_paperStyle.pdf' %(N, events))
    plt.show()





def master_plot_phisplit_thetamax(g1, g2, N, ni, events, counts, counts2, analytical= None, save=True):

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
    colors_hist = ["blue", "red", "green","blue", "red", "green", "black"]
    opacity = [.3, 1, 1,.3, 1, 1]
    line_width = [None, 3, 3,None, 3, 3]

    f = plt.figure(figsize=(10, 7))

    #gs = GridSpec(5, 1, width_ratios=[1], height_ratios=[3.5, 1, 0.9, 3.5, 1])
    gs = GridSpec(3, 1, width_ratios=[1], height_ratios=[4, 1, 1])
    ax1 = plt.subplot(gs[0])
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

    label_format = '{:,.1f}'
    ax1.set_yticks(np.arange(0, 0.8, 0.1))
    ax1.set_yticklabels([label_format.format(x) for x in np.arange(0, 0.8, 0.1)], size=16, fontname= 'times new roman')

    ######################    
    # simulation         #
    ######################

    hist_bins, centers= mo.hist_bins(ni, N, eps)
    width= (hist_bins[1] - hist_bins[0]) / 2.5
    #print('hist_bins: ' + str(hist_bins))
    #print('centers: ' + str(centers))
    #print('\n')
    #x_axis= np.arange(math.floor(centers[0]-width), math.ceil(centers[-1]+width) + 2, 1)
    x_axis= np.arange(math.floor(hist_bins[0]), math.ceil(hist_bins[-1]) + 2, 1)
    ax1.set_xticks(x_axis, 1)
    ax1.set_xticklabels(['']*len(x_axis))

    #tm_x= centers - width/2. # tm= theta max
    tm_x= centers
    tm_ey= np.zeros(N)
    tm_y= np.zeros(N)

    for c in counts:
        emit_list= []
        for n in range(0, N + ni):
            emit_list= [ptype(c.split()[-1-ni-n])] + emit_list
        #print(c, emit_list)
        theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
        #print('thetamax= ' + str(theta_max))
        if theta_max != None:
            j= 0
            while hist_bins[j+1] < theta_max:
                j+= 1

            tm_y[j]+= counts[c]/events
            tm_ey[j]+= counts[c]**0.5/events
    #print('\n')
    #print('weights: ' + str(tm_y))
    ax1.hist(tm_x, weights=tm_y, bins=hist_bins, alpha=0.2, color = 'red', density = True, label=label[4], histtype = hist_type[0], linewidth = line_width[1])

    if counts2 != None:
    
        #tm2_x= centers + width/2.
        tm2_x= centers
        tm2_ey= np.zeros(N)
        tm2_y= np.zeros(N)
    
        for c in counts2:
            emit_list= []
            for n in range(0, N + ni):
                emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

            theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
            if theta_max != None:
                j= 0
                while hist_bins[j+1] < theta_max:
                    j+= 1

                tm2_y[j]+= counts2[c]/events
                tm2_ey[j]+= counts2[c]**0.5/events
        #print('weights2: ' + str(tm2_y))
        ax1.hist(tm_x, weights=tm2_y, bins=hist_bins, alpha=0.2, color = 'blue', density = True, label=label[1], histtype = hist_type[0], linewidth = line_width[1])

    try:
        plt.plot(analytical[0], analytical[1], color= 'black', label=r'Analytical ($g_{12} = 0$)', linestyle='--')
    except:
        print('Invalid array input for MCMC.')

    plt.ylim((0, 0.75))

    font = matplotlib.font_manager.FontProperties(family='times new roman', size=16)

    plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $d\log(\theta_{\max})$', labelpad = 18, fontsize= 18)
    #plt.legend(loc='upper right', prop=font, frameon=False)
    plt.legend(loc='best', prop=font, frameon=False)
    #plt.text(-0.5, 0.77, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=10)
    #plt.text(-0.5, 0.65, r"4 steps",fontsize=10)
    #plt.text(-0.5, 0.52, r"$\phi\rightarrow f\bar{f}$ excluded", fontsize=10)
    #plt.plot(bs[len(bs)-1], newn, color = "black", linewidth = 0, marker="x", zorder=10)
    #plt.text(6.5, 0.78, r"(b)", fontsize=10)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)


    ######################
    # ratio plot         #
    ######################
    ax2 = plt.subplot(gs[1])
    ax2.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

    ax2.set_xticks(x_axis, 1)
    ax2.set_xticklabels(x_axis-1, size=16, fontname= 'times new roman')
    #ax2.set_xticklabels(size=16, fontname= 'times new roman')

    ax2.set_yticks(np.arange(0, 3.5, 2))
    ax2.set_yticklabels([label_format.format(x) for x in np.arange(0, 3.5, 2)], size=16, fontname= 'times new roman')

    plt.step(hist_bins, [1.]*len(hist_bins), color='black', linestyle=':')
    ratio= (tm2_y / np.sum(tm2_y)) / (tm_y / np.sum(tm_y))

    plt.step(hist_bins, np.concatenate((ratio[:1], ratio)), color='black')

    plt.ylim((0, 3.5))
    plt.ylabel('Classical /\n Quantum', fontsize= 18, labelpad = 18, fontname= 'times new roman')
    plt.xlabel(r'$\log(\theta_{\max})$', fontsize= 22, labelpad = 18, fontname= 'times new roman')
    #ax2.set_xlabel('Number of emissions (N)', fontsize= 16, labelpad = 19, fontname= 'times new roman')

    gs.update(wspace=0.025, hspace=0.07)
    gs.update(wspace=0.025, hspace=0.1)

    if save == True:
        plt.savefig('sim%dstep_thetamax_shots=%d_paperStyle.pdf' %(N, events))
    plt.show()




















def master_plot_nophisplit(g1, g2, N, ni, events, counts, counts2):

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
    colors_hist = ["blue", "red", "green","blue", "red", "green", "black"]
    opacity = [.3, 1, 1,.3, 1, 1]
    line_width = [None, 3, 3,None, 3, 3]





    f = plt.figure(figsize=(10, 10))

    gs = GridSpec(5, 1, width_ratios=[1], height_ratios=[3.5, 1, 0.9, 3.5, 1])
    ax1 = plt.subplot(gs[0])
    #ax1.set_xticklabels( () )
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')
    ax1.get_xaxis().set_label_coords(0, N+1)


    ######################
    # Classical MCMC     #
    ######################
    lnt_max_list={}
    vals_classical = np.load("PaperPlots/vals_classical.npy")
    lnt_max_list[0]=[]
    lnt_max_list[100]=[]
    for k in range(6):
        quantum_0 = np.load("PaperPlots/quantum_0_"+str(k)+".npy")
        quantum_100 = np.load("PaperPlots/quantum_100_"+str(k)+".npy")
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

    mysum = sum(ns[len(bs)-1])*2*bs2[len(bs)-1][0]
    newn = []
    for j in range(len(bs[len(bs)-1])):
        newn += [ns[len(bs)-1][j] / (mysum)]
        pass
    newn[len(newn)-1]=-999
    newn[len(newn)-2]=-999

    print(newn)
    print(bs[len(bs)-1])
    plt.plot(bs[len(bs)-1][:N+1], newn[:N+1], label='Classical MCMC', color = "black", linewidth = 0,marker="x")

    plt.hist(lnt_max_list[0][1], bins=mybins2, alpha=0.2, color = 'blue', density = True, label=label[1], histtype = hist_type[0], linewidth = line_width[1])
    plt.hist(lnt_max_list[0][4], bins=mybins2, alpha=0.2, color = 'red', density = True, label=label[4], histtype = hist_type[3], linewidth = line_width[4])

    ######################
    # IBMQ data ??       #
    ######################
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





    ######################    
    # simulation         #
    ######################
    #plt.hist(lnt_max_list[0][0], bins=mybins, alpha=0.3, color = 'blue', density = True, label=label[0], histtype = hist_type[0], linewidth = line_width[0])
    #xx,yy,zz = plt.hist(lnt_max_list[0][1], bins=mybins2, alpha=0.2, color = 'blue', density = True, label=label[1], histtype = hist_type[0], linewidth = line_width[1])
    #plt.hist(lnt_max_list[0][3], bins=mybins, alpha=0.3, color = 'red', density = True, label=label[3], histtype = hist_type[3], linewidth = line_width[3])
    emissions_y= np.zeros(N+1)
    emissions_ey= np.zeros(N+1)
    for c in counts:
        emit_list= []
        for n in range(N + ni):
            emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

        emit_N= int(mo.Nemissions(emit_list, n_I= ni))

        emissions_y[emit_N]+= counts[c]/events
        emissions_ey[emit_N]+= counts[c]**0.5/events

    #plt.bar(np.arange(0, N+1, 1), emissions_y, alpha=0.2, color = 'red', width= 1., label=label[4], tick_label= ['']*(N+1))

    emissions2_y= np.zeros(N+1)
    emissions2_ey= np.zeros(N+1)
    for c in counts2:
        emit_list= []
        for n in range(N + ni):
            emit_list= [ptype(c.split()[-1-ni-n])] + emit_list

        emit_N= int(mo.Nemissions(emit_list, n_I= ni))

        emissions2_y[emit_N]+= counts2[c]/events
        emissions2_ey[emit_N]+= counts2[c]**0.5/events

    print(emissions_y)
    print(emissions2_y)
    print(np.arange(0, N+1, 1))
    #plt.bar(np.arange(0, N+1, 1), emissions2_y, alpha=0.2, color = 'blue', width= 1., label=label[1], tick_label= ['']*(N+1))

    plt.ylim((0, 0.75))



    plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $dN$')
    plt.legend(loc='upper right', prop={'size': 9}, frameon=False)
    #plt.text(-0.5, 0.77, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=10)
    #plt.text(-0.5, 0.65, r"4 steps",fontsize=10)
    #plt.text(-0.5, 0.52, r"$\phi\rightarrow f\bar{f}$ excluded", fontsize=10)
    #plt.plot(bs[len(bs)-1], newn, color = "black", linewidth = 0, marker="x", zorder=10)
    #plt.text(6.5, 0.78, r"(b)", fontsize=10)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    ######################
    # ratio plot upper   #
    ######################
    ax2 = plt.subplot(gs[1])
    ax2.set_xticklabels( np.arange(-1, N+1, 1) )
    ax2.tick_params(bottom="True", right="False", top="False", left="True", direction='in')
    ax2.get_xaxis().set_label_coords(0, N+1)

    plt.step(np.arange(-0.5, N+0.5, 1), [1.]*(N+1), color='black', linestyle=':')
    ratio= emissions2_y / emissions_y
    plt.step(np.arange(-0.5, N+1.5, 1), np.concatenate((ratio[:1], ratio)), color='black')

    plt.ylim((0, 3.))
    plt.ylabel('Classical /\n Quantum', labelpad = 19)
    plt.xlabel(r'Number of emissions (N)')

    gs.update(wspace=0.025, hspace=0.07)

    #f.align_xlabels()
    
    
    ######################
    # ratio plot lower   #
    ######################
    ax1 = plt.subplot(gs[3])
    ax1.set_xticklabels( () )
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')
    plt.ylabel(r'$1 / \sigma$ $d\sigma$ $/$ $dN$')
    #plt.bar(nvalsfull,nvalsg120,width=1,label=r'simulation ($g_{12}=0$)',fill=False,color='black',linestyle=':')
    #plt.bar(nvalsfull,nvalsg121,width=1,label=r'simulation ($g_{12}=1$)',fill=False,color='black')
    #plt.legend(loc='upper right',prop={'size': 9.},frameon=False)
    #plt.text(2.8, 0.38-0.04, r"2 steps", fontsize=10)
    #plt.text(6.5, 0.67, r"(d)", fontsize=10)
    #plt.text(2.8, 0.28-0.04, r"$\phi\rightarrow f\bar{f}$ included", fontsize=10)

    ax1 = plt.subplot(gs[4])
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

    plt.step([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5], [1.,1.,1.,1.,1.,1.,1.,1.], color='black', linestyle=':')  

    plt.ylabel('Classical /\n Quantum', labelpad = 19)

    #plt.plot([-0.5,4.5],xx2,color='black',linestyle=':')
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)
    plt.ylim((0, 3.))
    plt.xlabel(r'Number of emissions (N)')

    gs.update(wspace=0.025, hspace=0.1)

    plt.show()