#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from termcolor import cprint

###########################
## Isentropic Formulas   ##
###########################

#SI Units
def density(ps,ts,Rgas):
    return ps/(Rgas*ts)

def mach(U,ts,gam,Rgas):
    return U/np.sqrt(gam*Rgas*ts)

def max_mass(pt,A,Rgas,tt,gam):
    const = gam**0.5*(2./(gam+1))**(0.5*(gam+1.)/(gam-1.))
    return const*pt*A/(Rgas*tt)**0.5    

def qdyn(rho,U):
    return 0.5*rho*U**2. #for consistency with empirical correlations

def correctedflow(Wobs,pamb,tamb):
    return Wobs*np.sqrt(tamb/288.15)*(101325./pamb)

#good up to 36089 ft = 11 km
#https://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
def Pambient(alt): #[m]
    #p0 = 101352.9
    p0 = 101325.
    c=2.25577e-5
    return p0*(1-c*alt)**5.25588 #[Pa]

def ambientPT(alt,ISA):
    #https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
    if alt <= 11000: #troposphere
        tamb = 273.15+15.04 -0.00649*alt +ISA
        pamb = 101.29*(tamb/288.08)**5.256 -0.16845
    elif alt > 11000 and alt < 25000: #lower stratosphere
        tamb = 273.15 -56.46 +ISA
        pamb = 22.65*np.exp(1.73-0.000157*alt)
    else: #upper stratosphere
        tamb = 273.15 -131.21 + 0.00299*alt +ISA
        pamb = 2.488*(tamb/216.6)**(-11.388)
    return tamb,pamb*1000.

def ttts(xM,gam):
    return (1+0.5*(gam-1)*xM**2.)

def ptps(xM,gam):
    return (1+0.5*(gam-1)*xM**2.)**(gam/(gam-1))

def mach_ptps(pt,ps,gam,inc):
    if pt >= ps:
        tmp = np.sqrt(2./(gam-1)*((pt/ps)**((gam-1)/(gam))-1))*inc[2] #inc[2]=Wsign
        ps=ps
        pt=pt
    elif inc[0] < 10:
        tmp = 0.02
    else:
        tmp= -np.sqrt(2./(gam-1)*((ps/pt)**((gam-1)/(gam))-1))*inc[2]
        cprint('CS%d ps>pt at inc %d reversing flow xM %1.2e' % (inc[1], inc[0],tmp),'red')
        ps=pt
        pt=ps
        #print(ps,pt)
        #tmp=0.
        #ps=pt
    return tmp,pt,ps

def mach_ttts(tt,ts,gam,inc):
    if tt >= ts:
        tmp = np.sqrt(2./(gam-1)*(tt/ts-1))
    else:
        tmp=0.0
        cprint('CS%d error ts>tt at inc %d' % (inc[1], inc[0]),'red')
    return tmp



print('isentropic relationships defined')


# In[ ]: