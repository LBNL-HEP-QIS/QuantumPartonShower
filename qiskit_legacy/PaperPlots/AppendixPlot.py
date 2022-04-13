import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def mylog(x):
  return int(np.log(x)/np.log(2.)+0.999) #This does the ceiling function.

def c(m,nI):
  return (1./6.)*(m+1.)*(m*m+3*m*nI+5*m+3*nI*nI+9*nI+6)

def C(n,b):
  return 64*b*b-94*b+32*n+3

def S1(m,nI):
  return 873*mylog(m+nI)-968

def S2(m,nI):
  return c(m,nI)*(96*mylog(m+nI)-27)

def S3(m,nI):
  mysum = 0.
  for j in range(1,m+nI+1):
    mysum+=(S1(m,nI))+3*c(m,nI)*C(4+3*mylog(m+nI-j+1),mylog(j))
    pass
  return mysum

def S4(m,nI):
  return (m+nI)*(224*mylog(m+nI)+143)

def SS1(N,nI):
  mysum = 0.
  for m in range(0,N):
    mysum+=S1(m,nI)
    pass
  return mysum

def SS2(N,nI):
  mysum = 0.
  for m in range(0,N):
    mysum+=S2(m,nI)
    pass
  return mysum

def SS3(N,nI):
  mysum = 0.
  for m in range(0,N):
    mysum+=S3(m,nI)
    pass
  return mysum

def SS4(N,nI):
  mysum = 0.
  for m in range(0,N):
    mysum+=S4(m,nI)
    pass
  return mysum

def func(x, a, b, c, d, e, f):
  myout = []
  for xx in x:
    myout+=[(a*xx**5+b*xx**4+c*xx**3+d*xx**2+e*xx+f)*np.log(xx)]
    pass
  return np.array(myout)

ni = 1 #number of initial particles.

x = []
y = []
y2 = []
x_forfit = []
y_forfit = []
y2_forfit = []
for i in range(1,50):
  x+=[i*1.]
  y+=[SS1(i,ni)+SS2(i,ni)+SS3(i,ni)+SS4(i,ni)]
  y2+=[SS1(i,ni)]
  if (i > 15):
    x_forfit+=[i*1.]
    y_forfit+=[SS1(i,ni)+SS2(i,ni)+SS3(i,ni)+SS4(i,ni)]
    y2_forfit+=[SS1(i,ni)]
    pass
  pass

popt, pcov = curve_fit(func, x_forfit, y_forfit)

f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(1, 1, 1)
plt.plot(np.array(x),np.array(y), label='Calculation', color = '#89B9DD', linewidth = 0,marker="o")
#plt.plot(np.array(x),np.array(y2), label='Repeated Measurements', color = '#E1816D', linewidth = 0,marker="v")
#plt.ylim((0, 0.6))
ax.set_yscale("log", nonposy='clip')
plt.ylabel(r'Number of Standard Gates')
plt.xlabel(r'Number of steps (N)')
#plt.plot(np.array(x_forfit),np.array(y_new),color='black',label="quartic fit",linestyle='--')
#plt.plot(np.array(x_forfit),np.array(y_new2),color='black',label="quadratic fit")
plt.subplots_adjust(left=0.15,right=0.9,top=0.95,bottom=0.15)
plt.plot(x_forfit, func(x_forfit, *popt), color='black',label=r"fit to $N^{5}\log(N)$",linestyle='--') #, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend(loc='lower right',prop={'size': 9.5},frameon=False)
f.savefig("appendixplot.pdf")

