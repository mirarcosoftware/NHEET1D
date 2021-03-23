##### TITLE BLOCK #####
# Created by D Cerantola, Mar.22 2021
#
#

import numpy as np
from fluid_properties12 import chem_rxn,MolWt,specheat,specheatk, \
    enthalpy, enthalpy_Tnew, enthalpy_T, entropy,viscosity,conductivity

def reaction_balance():
    #global MW,Rgas,p_frc,z_st,p_mol,ptot,rxC
    #number moles of species for air to give 21% O2 and 79% N2 (by mol)
    rxC =0; rxH =0; rxO =0
    
    R_u = 8.31451
    eta_comb =1. #no unburnt hydrocarbons
    oxidant =int(0) #0='air' or 1='oxy'
    liqtogas =1

    MW,Rgas,p_frc,z_st,p_mol,ptot = chem_rxn(1.,eta_comb,oxidant,rxC,rxH,rxO)#air
    
    rxn_consts=[MW,Rgas,p_frc,z_st,p_mol,ptot,rxC]
    return(rxn_consts)

def ptps(xM,gam):
    return (1+0.5*(gam-1)*xM**2.)**(gam/(gam-1))

def factors_fn(inputs):
    M = len(inputs)
    NN = np.array([0]*M)
    for i in range(0,M):
        NN[i] = np.size(inputs[i])
    NNtot = np.prod(NN)
    factors = np.zeros((NNtot,M))

    for i in range(0,M): #outputs a 2D array
        Ni = int(NNtot/np.prod(NN[:i+1]))
        tmp1=[]
        for j in range(0,NN[i]):
            Nj = int(NNtot/np.prod(NN[i:]))
            tmp1.append([inputs[i][j]]*Nj)
        tmp = np.reshape(tmp1,(1,np.product(np.shape(tmp1))))[0]
        col =[tmp]*Ni 
        col = np.reshape(col,(1,np.product(np.shape(col))))[0]

        factors[:,i] = col
    return factors,NN,NNtot

def ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,x,V0,rhof_mean,muf_mean):   
    Duse = Dp*Dpcorr
    delta_p = x*(150.*muf_mean/Duse**2*(1-eps)**2/eps**3*V0 + 1.75*rhof_mean/Duse*(1-eps)/eps**3*V0**2)
    return delta_p

def Koekemoer_PressureLoss(Dp,Dpcorr,phi,eps,x,V0,rhof_mean,muf_mean):    
    k1 = 77.4
    k2 = 2.8
    Duse = phi*Dp*Dpcorr
    delta_p = x*(k1*muf_mean/Duse**2*(1-eps)**2/eps**3*V0 + k2*rhof_mean/Duse*(1-eps)/eps**3*V0**2)
    return delta_p

def NusseltNumber(eps,Rep,Pr,Nu_method):
    if Nu_method ==1:
        tmp = ((1.18*Rep**0.58)**4 + (0.23*(Rep/(1 - eps))**0.75 )**4)**0.25
    elif Nu_method ==2:
        tmp = 1.15*Rep**0.6*Pr**(1./3.)
    return tmp

def VoidFraction(Dp,delDp,Dpcorr): #not used
    Duse = Dp*Dpcorr
    aeps=0.43
    beps=0.061
    ceps=-0.030
    return aeps+beps*Duse+c*delDp