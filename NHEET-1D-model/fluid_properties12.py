#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from numpy import loadtxt
from termcolor import cprint
import sys
#from pathlib import Path
workingDir = os.getcwd()
if os.name == 'nt':
    filepath = workingDir + r'\NHEET-1D-model\coeffs\\'
else:
    filepath = workingDir + r'/coeffs/'
R_u = 8.31451

# % species order (new)
#   0  CHO unburnt
# % 1  CO2    CH4
# % 2  H20    C2H6
# % 3  O2     C3H8
# % 4  N2     C4H10
# % 5  H2     C5H12
# % 6  CO     C6H14
# % 7  OH     C7H16
# % 8  H      C8H18
# % 9  O      C9H20
# % 10 N      C10H22
# % 11 NO     C11H24
# % 12 NO2    C12H24

# ### Combustion stuff
# #Stoichiometric reaction of methane rxC=1,rxH=4,rxO=0
# #for air rXC=0.rxH=0,rxO=0
# equivalence ratio (fuel rich/fuel lean) ER =1
# combustion efficiency eta_c =1
# oxidant ='0=air' '1=oxy'

## Coefficients
ahighh = loadtxt(filepath+r'ahighh12.txt', comments="#", delimiter="\t", unpack=False)
ahighhorg = loadtxt(filepath+r'ahighhorg.txt', comments="#", delimiter="\t", unpack=False)
ahighlamda = loadtxt(filepath+r'ahighlamda12.txt', comments="#", delimiter="\t", unpack=False)
ahighmu = loadtxt(filepath+r'ahighmu12.txt', comments="#", delimiter="\t", unpack=False)
alowh = loadtxt(filepath+r'alowh12.txt', comments="#", delimiter="\t", unpack=False)
alowhorg = loadtxt(filepath+r'alowhorg.txt', comments="#", delimiter="\t", unpack=False)
alowlamda = loadtxt(filepath+r'alowlamda12.txt', comments="#", delimiter="\t", unpack=False)
alowlamdaorg = loadtxt(filepath+r'alowlamdaorg.txt', comments="#", delimiter="\t", unpack=False)
alowmu = loadtxt(filepath+r'alowmu12.txt', comments="#", delimiter="\t", unpack=False)
alowmuorg = loadtxt(filepath+r'alowmuorg.txt', comments="#", delimiter="\t", unpack=False)

print('coefficients loaded')

def chem_rxn(ER,eta_comb,oxidant,rxC,rxH,rxO):
# %(ER, eta_c, oxidant, rC, rH, rO);
# %
# % CHEM_RXN: finds product mole fractions to ideal 1-step HC rxn
# %           and corresponding molar mass

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %  Chemical Reaction - May 19,23,24   %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# global ER eta_c oxidant rxC rxH rxO R_u;

# %stoichiometric #kmols air 
    z_st = (rxC + rxH/4 - rxO/2);
    if rxH ==0:
        pCO2 = 0;
        pH2O = 0;
        pO2 = 1.;
        pN2 = 3.76;
        pCHO = 0;
        pCO=0.
    elif ER <=1. and oxidant == 0: #air
    #% assume complete combustion
        rO2 = 1.*eta_comb/ER*z_st
        rN2 = 3.76*eta_comb/ER*z_st
        pN2 = rN2;
        #%Products
        pCO2 = eta_comb*rxC;
        pH2O = eta_comb*rxH/2.;
        pO2 = z_st*eta_comb*(1.-ER)/ER;
        pCHO = (1.-eta_comb)
        pCO=0.
    elif ER <=1. and oxidant == 1: #oxygen
        rO2 = 1.*eta_comb/ER*z_st;
        pN2 = 0;
        #%Products
        pCO2 = eta_comb*rxC;
        pH2O = eta_comb*rxH/2.;
        pO2 = z_st*eta_comb*(1-ER)/ER;
        pCHO = (1. -eta_comb)
        pCO=0.
    elif ER >1. and oxidant == 0:
        rO2 = eta_comb/ER*(rxC/2*(ER+1)+rxH/4*ER)
        rN2 = 3.76*eta_comb/ER*(rxC/2*(ER+1)+rxH/4*ER)
        pN2 = rN2;
        #%Products
        pCO2 = eta_comb/ER*rxC;
        pH2O = eta_comb*rxH/2;
        #pO2 = (1-eta_comb)*z_st/ER;
        pO2 = 0.
        pCHO = (1-eta_comb)
        pCO= eta_comb*(ER-1)/ER*rxC
    elif ER >1. and oxidant == 1:
        rO2 = 1/ER*z_st;
        pN2 = 0;
        #%Products
        pCO2 = eta_comb/ER*rxC;
        pH2O = eta_comb*rxH/2;
        pO2 = 0.
        pCHO = (1-eta_comb); 
        pCO=0.
    
#     pCO2 = ds[0]*pCO2
#     pCO = (1-ds[0])/ds[0]*pCO2
    
#     pH2O = ds[1]*pH2O
#     pH2 = 0.5*(1-ds[1])/ds[1]*pH2O
#     pOH = 0.5*(1-ds[1])/ds[1]*pH2O
 
    #no dissociation
    pH2=0.
    pOH=0.
    pH=0.
    pO=0.
    pN=0.
    pNO=0.
    pNO2=0.
    
    #products #[CHO,CO2,H2O,O2,N2,H2,CO,OH,H,O,N,NO,NO2]
    #ptot = pCHO+ pCO2 + pH2O + pO2 + pN2 +pH2 +pCO +pOH +pH +pO +pN +pNO +pNO2
    p_mol = [pCHO,pCO2, pH2O, pO2, pN2, pH2, pCO,pOH,pH,pO,pN,pNO,pNO2]
    ptot = sum(p_mol)
    p_frc = np.divide(p_mol,ptot);
    
    MW = MolWt(p_frc, rxC)[0] #%[kg/kmol]
    #print(p_mol,ptot,p_frc,MW)
    Rgas = 1000*R_u/MW; #%[J/kg-K]

#     #reactants
#     if rxC !=0:
#         rCO2 =0
#         rH2O =0
#         rCHO = rxC
#         rtot = rO2 + rN2 +rCHO
#         r_frc = np.divide([0,rCO2, rH2O, rO2, rN2, rCHO,0,0,0],rtot);
#     else: 
#         r_frc = p_frc
    
    return MW, Rgas, p_frc, z_st, p_mol, ptot

def MolWt (p_frc, rxC):
    #M = [0,28.010, 44.011, 2.016, 1.008, 17.007, 18.016, 28.013, 14.007, 30.006, 46.006, 31.999, 16.000] old order
    M = [0,44.011,18.016,31.999,28.013,2.016,28.010,17.007,1.008,16.000,14.007,30.006,46.006]
    M2 = [0,16.043, 30.069, 44.096, 58.123, 72.150, 86.177, 100.203, 114.230, 128.257, 142.284, 156.311, 168.322] #from Turns
    if rxC ==0:
        M3 = M[5];
    else:
        M3 = M2[rxC]
    #print(M3,p_frc,M)
    MWspecies = [M3]+M[1:]
    MWall = np.concatenate(([M3*p_frc[0]],np.multiply(p_frc[1:],M[1:])))
    #print(sum(MWall)/sum(p_frc))
    return sum(MWall)/sum(p_frc),MWspecies

#MW=MolWt(p_frc,rxC)
#print(MW)
#print (R_u/MW*1000.)


# # In[107]:
# def enthalpy_of_formation(p_frc,rC,liqtogas):
#     Hfref = [0.,-110541.,-393546.,0.,217977.,38985.,-241845.,0.,472629.,90297.,33098.,0.,249197.] #kJ/kmol
#     Hfref2 = [0.,-74831.,-84667.,-103847.,-124733.,-146440.,-167193.,-187820.,-208447.,-229032.,-249659.,-270286.,-165352.] #[kJ/kmol]
#     hfg2 = [0.,509.,488.,425.,386.,358.,335.,316.,300.,295.,277.,265.,251.] #[kJ/kg] Kerosene from Cengel and Boles

#     if rC ==0:
#         h3 = Hfref[3];
#         hfg3 =0.
#     else:
#         h3 = Hfref2[rC]
#         hfg3 = hfg2[rC]
    
#     MWf = MolWt (p_frc, rC)
#     hfgmol = MWf*hfg3*liqtogas
    
#     Hf = [0.,p_frc[1]*Hfref[2], p_frc[2]*Hfref[6], p_frc[3]*Hfref[11], \
#         p_frc[4]*Hfref[7], p_frc[5]*(h3+hfgmol),Hfref[1]*p_frc[6],Hfref[3]*p_frc[7],Hfref[5]*p_frc[8]]
#     return Hf #[kJ/kmol]

def specheat(T, p_frc, rC):
# %
# %SPECHEAT  finds constant pressure specific heat of mixture given number 
# %   of kmoles of species on a [kJ/kg-K] basis
# %   uses NASA coefficients
# %
# % Output: Cp [kJ/kmol-K]

# %R_u = 8.31451; %[kJ/kmol-K]
    #MW = MolWt(p_frc, rC);
    
    if T < 1000.:
        A = alowh
        A2 = alowhorg
    else:    
        A = ahighh
        A2 = ahighhorg
    
    b2 = R_u*np.array([[1.], [T], [T**2.], [T**3.], [T**4.], [0.], [0.]]);
    
    if rC ==12:
        b22 = R_u*np.array([[1.], [T], [T**2.], [T**3.], [T**4.], [0.], [0.], [0.], [0.]]);
    elif rC==9 or rC==10 or rC==11:
        b22 = np.array([[0],[0],[1.], [T], [T**2.], [T**3.], [T**4.],[0],[0]]);
    else:
        b22 = R_u*np.array([[T**(-2)], [T**(-1)], [1.],[T],[T**2.], [T**3.], [T**4.], [0.], [0.]]);
    

    Cp = np.vstack(([0],A.dot(b2))); #[kJ/kmol-K]
    #print(Cp)
    Cp2 = np.vstack(([0],A2.dot(b22)));

    if rC ==0:
        Cp3 = Cp[5]; #H2
    else:
        Cp3 = Cp2[rC]
    
#     print(Cp)
#     Cprow = np.array(Cp)
#     print(Cprow)
#     map(list, zip(*Cprow))
#     print(Cprow)
    
    Cps= np.concatenate(([Cp3[0]*p_frc[0]],p_frc[1:]*Cp[1:].T[0]))
    #Cps = [Cp3[0]*p_frc[0],(p_frc[1:]*(Cp[1:].T)[0][0])]
    #Cps = [0,Cp[2],Cp[6],Cp[11],Cp[7],Cp3]
    #print(Cps)
    return (sum(Cps)) #/MW; #[kJ/kg-K]

def specheatk(T, p_frc, rxC):
    Cp = specheat(T, p_frc, rxC)
    #MW = MolWt(p_frc, rxC);
    Cv = Cp - R_u #/MW;
    return Cp/Cv;

def enthalpy(T, p_frc, rC, H0onoff,liqtogas):
#     %
#     %ENTHALPY  finds enthalpy of mixture given number of kmoles of species
#     %   on a [kJ/kmol-prod] basis May24
#     %   uses NASA coefficients
#     %   H0 tables in kJ/mol
#     % Output: h [kJ/kmol]
#     %
#NASA 211556              1         2       3     4      5       6        7     8       9      10    11     12
    #H0 = np.multiply([0,-119.206,-402.875,-8.468,211.801,28.465,-251.73, -8.67,466.483,82.092,23.985,-8.68,242.45],1000.*H0onoff) #old order
    
    H0 = np.multiply([0,-402.875,-251.73,-8.68,-8.67,-8.468,-119.206,28.465,211.801,242.45,466.483,82.092,23.985],1000.*H0onoff) #new order
    H02 = np.multiply([0,-84.616,-95.743,-119.421,-145.019,-170.944,-195.622,-221.001, -246.530,0.,0.,0.,-385.765*1.18],1000.*H0onoff)
    hfg2 = np.multiply([0.,509.,488.,425.,386.,358.,335.,316.,300.,295.,277.,265.,247.],liqtogas) #[kJ/kg] Kerosene=251 from Cengel and Boles, 247 from Wang
    
    #print(H0)

    #MW = MolWt(p_frc, rC);
    if T < 1000.:
        A = alowh
        A2 = alowhorg
    else:    
        A = ahighh
        A2 = ahighhorg
        
    # evaluate integral of Cp_bar dt from 0 to T
    b2 = R_u*np.array([[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.], [1.], [0.]]);
    #b22 = np.array([[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.]]);
    if rC ==12:
        b22 = R_u*np.array([[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5],[1.], [0.], [0.],[0.]]);
    elif rC==9 or rC==10 or rC==11:
        b22 = np.array([[0],[0],[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.],[0],[0]]);
    else:
        b22 = R_u*np.array([[-T**(-1)], [np.log(T)], [T],[T**2/2.],[T**3./3.], [T**4./4.], [T**5./5.], [1.], [0.]]);
                            
    h = np.vstack(([0],A.dot(b2))).T[0]; #[kJ/kmol]
    #print(h)
    #h=(h.T[0])
    #print(h[1:])
    #print(H0[1:])
    #print(np.multiply(p_frc[1:],(h[1:]-H0[1:])))
    h2 = np.vstack(([0],A2.dot(b22)));

    if rC ==0 or p_frc[0]==0.:
        h3 = h[5];
        H03=H0[5]
    else:
        h3 = h2[rC][0]
        H03 = H02[rC] - hfg2[rC]*MolWt(p_frc, rC)[0]
#     print(h)
#     print(H0)
#     print(h.T[0]-H0)
    h_bar= np.concatenate(([(h3-H03)*p_frc[0]],np.multiply(p_frc[1:],(h[1:]-H0[1:])))) #[kJ/kmol-prod] #NB .T = transpose
    #print('fp85',h_bar)    
#     #p_frc = [pCO2, pH2O, pO2, pN2, pCHO];
#     h_bar = [0,p_frc[1]*(h[2]-H0[2]), p_frc[2]*(h[6]-H0[6]), p_frc[3]*(h[11]-H0[11]), \
#         p_frc[4]*(h[7]-H0[7]), p_frc[5]*(h3-H03),(h[1]-H0[1])*p_frc[6],(h[3]-H0[3])*p_frc[7],(h[5]-H0[5])*p_frc[8]]; #[kJ/kmol-prod]
    #print(h_bar,h[11],H0[11])
    #h_bar = np.array(h_bar).T[0]
    
    #Hf = np.array(enthalpy_of_formation(p_frc,rC,1))
    #map(list, zip(*Hf))
    #print(h_bar,Hf)
    #h_bar2 = h_bar-Hf
    return sum(h_bar) #kJ/kmol] #*1000./MW #[J/kg]

def enthalpy_Tnew(T, h_ref, p_frc, rC, H0onoff,liqtogas):
# %
# %ENTHALPY_T  isolates T in enthalpy eqn in order to find mixture 
# %   temperature May24, Jun9, Mar1
# %   uses NASA coefficients
#   H0onoff 0=combustion formation, 1= thermo (cp*T)
# %
# % Output: T [K] (Implicit)
# %

#NASA 211556
    H0 = np.multiply([0,-402.875,-251.73,-8.68,-8.67,-8.468,-119.206,28.465,211.801,242.45,466.483,82.092,23.985],1000.*H0onoff) #new order
    H02 = np.multiply([0,-84.616,-95.743,-119.421,-145.019,-170.944,-195.622,-221.001, -246.530,0.,0.,0.,-385.765*1.18],1000.*H0onoff)
    hfg2 = np.multiply([0.,509.,488.,425.,386.,358.,335.,316.,300.,295.,277.,265.,247.],liqtogas) #[kJ/kg] 

    #MW = MolWt(p_frc, rC);
    if T < 1000.:
        A = alowh
        A2 = alowhorg
    else:    
        A = ahighh
        A2 = ahighhorg

    b2 = R_u*np.array([[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.], [1.], [0.]]);
    b_no_a1 = R_u*np.array([[0.], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.], [1.], [0.]]);
    #b22 = np.array([[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.]]);
    if rC ==12:
        b22 = R_u*np.array([[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5],[1.], [0.], [0.],[0.]]);
    elif rC==9 or rC==10 or rC==11:
        b22 = np.array([[0],[0],[T], [T**2./2.], [T**3./3.], [T**4./4.], [T**5./5.],[0],[0]]);
    else:
        b22 = R_u*np.array([[-T**(-1)], [np.log(T)], [T],[T**2/2.],[T**3./3.], [T**4./4.], [T**5./5.], [1.], [0.]]);

    h = np.vstack(([0],A.dot(b2))).T[0]; #[kJ/kmol-K]

    h_no_a1 = np.vstack(([0],A.dot(b_no_a1))).T[0];
    h2 = np.vstack(([0],A2.dot(b22)));

    if rC ==0 or p_frc[0]==0.:
        h3 = h[5];
        H03 = H0[5]
    else:
        h3 = h2[rC][0]
        H03 = H02[rC] - hfg2[rC]*MolWt(p_frc, rC)[0]
       
    h_ref_mol = h_ref #*MW/1000.; #%[kJ/kmol]

    #print(h3,H03,p_frc[0])
    #%p_frc = [pCO2, pH2O, pO2, pN2, pCHO];

#     h_bar = [p_frc[1]*h[2], p_frc[2]*h[6], p_frc[3]*h[11], \
#              p_frc[4]*h[7], p_frc[5]*h3,h[1]*p_frc[6],h[3]*p_frc[7],h[5]*p_frc[8]]
#     h_bar = [0,p_frc[1]*(h[2]-H0[2]), p_frc[2]*(h[6]-H0[6]), p_frc[3]*(h[11]-H0[11]), \
#         p_frc[4]*(h[7]-H0[7]), p_frc[5]*(h3-H03),(h[1]-H0[1])*p_frc[6],(h[3]-H0[3])*p_frc[7],(h[5]-H0[5])*p_frc[8]]; #[kJ/kmol-prod]
    #print(h3,H03,h,H0)
    h_bar= np.concatenate(([(h3-H03)*p_frc[0]],np.multiply(p_frc[1:],(h[1:]-H0[1:])))) #[kJ/kmol-prod]
    h_sum = sum(h_bar); #%[kJ/kmol-prod]
    #print(h_bar,h_sum)

    if p_frc[4] != 0.: #N2
    #         T = (h_ref - (h_sum -p_frc[4]*h[7])-p_frc[4]*h_no_a1[7])/...
#             (p_frc[4]*R_u*A[7,1]); #A[6,0]?
        T = (h_ref_mol - (h_sum -p_frc[4]*(h[4]-H0[4]))-p_frc[4]*(h_no_a1[4]-H0[4]))/ \
            (p_frc[4]*R_u*A[3,0]);
    elif p_frc[4] ==0 and p_frc[2] >= p_frc[1]: #H2O
#         T = (h_ref - (h_sum -p_frc[2]*h[6])-p_frc[2]*h_no_a1[6])/...
#             (p_frc[2]*R_u*A[6,1]) #check A r,c should be [5,0]?
        T = (h_ref_mol - (h_sum -p_frc[2]*(h[2]-H0[2]))-p_frc[2]*(h_no_a1[2]-H0[2]))/ \
            (p_frc[2]*R_u*A[1,0]) 
    elif p_frc[4] ==0 and p_frc[2] < p_frc[1]: #CO2
#         T = (h_ref - (h_sum -p_frc[1]*h[2])-p_frc[1]*h_no_a1[2])/...
#             (p_frc[1]*R_u*A[2,1]); #A[1,0]?
        T = (h_ref_mol - (h_sum -p_frc[1]*(h[1]-H0[1]))-p_frc[1]*(h_no_a1[1]-H0[1]))/ \
            (p_frc[1]*R_u*A[0,0]); 
    else:
            cprint('enthalpy_T fluid molars not programmed','red')
            sys.exit('error')
    return T

def enthalpy_T(T,h_ref,p_frc,rC,H0onoff,liqtogas): #not stable??
    errT =1.
    inc =1
    while abs(errT)>1.e-2: #%solve for T by defn h = int(Cp dT)
        errlast=errT
        #print(T,h_ref,p_frc,rC)
        errT = enthalpy_Tnew(T,h_ref,p_frc,rC,H0onoff,liqtogas) -T
        #print(T,enthalpy_Tnew(T,h_ref,p_frc,rC,H0onoff),h_ref,errT,errlast)
        #errTpr = 1.-specheat(T,p_frc,rC)*1000. #not correct
        #print(inc,T,enthalpy_Tnew(T,h_ref,p_frc,rC),errT)
#         if abs(errT) > abs(errlast):
#             T -= 0.1*errT #%relaxation needed when T1=~T2
#         else:
#             T -= 1.*errT
        T += 0.5*errT
        #T -= errT/errTpr
        #print(inc)
        #T = T[0]
        if inc == 100:
            cprint('enthalpy_T not converged err %1.2e' % errT,'red')
            break
        #print(inc,T,err)
        inc +=1
    return T

def entropy(T, p_frc, rC):
#     %
#     %ENTROPY  finds entropy of mixture given number of kmoles of species
#     %   on a [kJ/kmol-prod] basis May24
#     %   uses NASA coefficients
#     %
#     % Output: h [kJ/kmol]
#     %
    #MW = MolWt(p_frc, rC);
    if T < 1000.:
        A = alowh
        A2 = alowhorg
    else:    
        A = ahighh
        A2 = ahighhorg
    
    # evaluate integral of Cp_bar dt from 0 to T
    b2 = R_u*np.array([[np.log(T)], [T], [T**2./2.], [T**3./3.], [T**4./4.], [0.], [1.]]);
    #b22 = np.array([[np.log(T)], [T], [T**2./2.], [T**3./3.], [T**4./4.]]);
    if rC ==12:
        b22 = R_u*np.array([[np.log(T)], [T], [T**2./2.], [T**3./3.], [T**4./4.], [0.], [1.], [0.], [0.]]);
    elif rC==9 or rC==10 or rC==11:
        b22 = np.array([[0],[0],[np.log(T)], [T], [T**2./2.], [T**3./3.], [T**4./4.],[0],[0]]);
    else:
        b22 = R_u*np.array([[-T**(-2.)/2.], [-T**(-1)], [np.log(T)],[T],[T**2/2.],[T**3./3.], [T**4./4.],[0.], [1.]]);

    h = np.vstack(([0],A.dot(b2))).T[0]; #[kJ/kmol]

    h2 = np.vstack(([0],A2.dot(b22)));

    if rC ==0:
        h3 = h[5];
    else:
        h3 = h2[rC][0]

    #p_frc = [pCO2, pH2O, pO2, pN2, pCHO];
    h_bar= np.concatenate(([(h3)*p_frc[0]],np.multiply(p_frc[1:],(h[1:])))) #[kJ/kmol-prod]
#     h_bar = [p_frc[1]*h[2], p_frc[2]*h[6], p_frc[3]*h[11], \
#         p_frc[4]*h[7], p_frc[5]*h3,h[1]*p_frc[6],h[3]*p_frc[7],h[5]*p_frc[8]]; #[kJ/kmol-prod]
    return sum(h_bar) #*1000./MW #[J/kg]

def viscosity(T, p_frc, rC):
# %
# %VISCOSITY  finds viscosity of mixture given number of kmoles of species
# %   on a [kg/m-s] basis
# %   uses NASA and Yaws coefficients
# %
# % Output: mu [kg/m-s]
# %

    A2 = alowmuorg
    if T < 1000.:
        A = alowmu;
    else:
        A = ahighmu;
    b = np.array([[np.log(T)], [1./T], [1./T**2.], [1.]]);
    b2 = np.array([[1.], [T], [T**2.]])

    mu = np.vstack(([0],np.multiply(np.exp(A.dot(b)),10**-7))); #%[kg/m-s]
    #print(mu)
    mu2 = np.vstack(([0],np.multiply(A2.dot(b2),10**-7))); #%[kg/m-s]

    if rC ==0:
        mu3 = mu[5];
    else:
        mu3 = mu2[rC]

    mu_bar= np.concatenate(([(mu3[0])*p_frc[0]],p_frc[1:]*(mu[1:]).T[0])) #[kJ/kmol-prod]
    return sum(mu_bar); #%[kg/m-s]

def conductivity(T, p_frc, rC):
# %
# %CONDUCTIVITY  finds conductivity of mixture given number of kmoles of
# %   specied on a [W/m-K] basis
# %   uses NASA and Yaws coefficients
# %
# % Output: lamda [W/m-K]
# %


    A2 = alowlamdaorg;
    if T < 1000.:
        A = alowlamda
    else:
        A = ahighlamda
    b = np.array([[np.log(T)], [1./T], [1./T**2.], [1.]]);
    b2 = np.array([[1.], [T], [T**2.]])
        
    la = np.vstack(([0],np.multiply(np.exp(A.dot(b)),10**-4))); #%[W/m-K]
    #print(mu)
    la2 = np.vstack(([0],A2.dot(b2))); #%[W/m-K]

    if rC ==0:
        la3 = la[5];
    else:
        la3 = la2[rC]
    
    la_bar= np.concatenate(([(la3[0])*p_frc[0]],p_frc[1:]*(la[1:]).T[0])) #[kJ/kmol-prod]
    return sum(la_bar) #%[W/m-K]


def density_liq_kerosene(T,rho0): #T in K
    #incompressible so not dependent on pressure
    #rho0 = 800 kg/m3 = 0.8kg/litre expected
    a0 = 1.2691005
    a1 = -0.0009349366
    return rho0*(a0+a1*T) #kg/m3
        


# T=500

# MW,Rgas,p_frc = chem_rxn()
# print(MW,Rgas,p_frc)

# cp = specheat(T, p_frc, rxC)
# print(cp)

# gam = specheatk(T, p_frc, rxC)
# print(gam)

# mu=viscosity(T, p_frc, rxC)
# print(mu)

# la=conductivity(T, p_frc, rxC)
# print(la)

print('fluid properties loaded from path', filepath)

# In[ ]: