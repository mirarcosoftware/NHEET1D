#######################################
###                                 ###
#######     Schumanns Model     #######
###                                 ###
#######################################

#############################
#####     Libraries     #####
#############################
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import sys
import csv
import pandas as pd
from termcolor import cprint
from pathlib import Path
#from scipy.interpolate import griddata
print('Libraries loaded')
#import time
from tictoc import tic, toc

#A fluid properties solver intended for combustion applications is used here to obtain relationships for SI properties of dry air as a function of temperature.
#Inputs below are selected so that properties of dry air are provided.
from fluid_properties12 import chem_rxn,MolWt,specheat,specheatk, \
    enthalpy, enthalpy_Tnew, enthalpy_T, entropy,viscosity,conductivity

from isentropic_formulas import density, mach, max_mass,qdyn, correctedflow, \
    ambientPT, ttts, ptps, mach_ptps, mach_ttts

def reaction_balance():
    #number moles of species for air to give 21% O2 and 79% N2 (by mol)
    rxC =0; rxH =0; rxO =0
    #jet fuel ~kerosene, use C12H24 https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20000065620.pdf
    rfC=12; rfH=24; rfO=0

    R_u = 8.31451
    eta_comb =1. #no unburnt hydrocarbons
    oxidant =int(0) #0='air' or 1='oxy'
    liqtogas =1

    MW = [0.]*5
    Rgas = [0.]*5
    p_frc = [[0.]*13]*5
    z_st = [0.]*5
    p_mol = [[0.]*13]*5
    ptot = [0.]*5

    #number moles of fuel for chemical reaction
    p_frc[0] = [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    p_mol[0] = p_frc[0]
    ptot[0] = sum(p_mol[0])
    MW[0] = MolWt(p_frc[0], rfC)[0]
    MW[1],Rgas[1],p_frc[1],z_st[1],p_mol[1],ptot[1] = chem_rxn(1.,eta_comb,oxidant,rxC,rxH,rxO)#air
    #print(rfC,MW[1],Rgas[1],p_frc[1],z_st[1],p_mol[1],ptot[1])
    return MW,Rgas,p_frc,z_st,p_mol,ptot,rxC

#Schumann Model Functions
def getA_Scf(rhof,cpf):

    A_Scf = eps*rhof*cpf
    
    return A_Scf

def getB_Scf(G,cpf):

    B_Scf = G*cpf
    
    return B_Scf

def getC_Scf(G, Dp, muf, kf, eps):
    
    #Heat Transfer Correlations
    Rep = G*Dp/muf
    ##Verify which hp
    Nup = ((1.18*Rep**0.58)**4 + (0.23*(Rep/(1 - eps))**0.75 )**4)**0.25
    hp = Nup*kf/Dp
    #hp = 17.773    #Abdel-Ghaffar, E. A.-M., 1980. PhD Thesis
    #hp=1/(Dp/(Nup*kf) + Dp/(10*ks)) #Effective heat transfer coefficient accounting for Biot Number Effects
    Ap = 6*(1-eps)/(Dp)
    C_Scf = hp*Ap 

    return C_Scf

def getD_Scf(kf,muf,cpf):  #For Wall Heat Loss - Might add this later... Assumed adiabatic for now. 

    #hw = (kf_in/Dp)*(2.576*Rep_in**(1./3.)*Pr**(1./3.)+0.0936*Rep_in**0.8*Pr**0.4)
    #h2 = np.log((0.5*np.sqrt(Ly**2+Lz**2)+ithk)/(0.5*np.sqrt(Ly**2+Lz**2)))/(2*np.pi*L*0.06)
    #hx = 1.42*(75./L)**0.25
    #Uw = 1./(1./(hw*(2*Ly*L+2*Lz*L)) + h2 + 1./(hx*(2*Ly*L+2*Lz*L))) 
    #D_Scf=-UwAw
    D_Scf = 0.0

    return D_Scf

def getA_Scs(rhos,cps):

    A_Scs = (1-eps)*rhos*cps #Constant
    
    return A_Scs

def getB_Scs(C_Scf):

    B_Scs = C_Scf #They are the same
    
    return B_Scs

def T_inlet(new_time):
    if T_inlet_method == 1:
        T = Tprof(new_time)
        return T
    elif T_inlet_method == 2: 
        T = Ttable(new_time)
        return T
    else:
        print('Error... Invalid Inlet Temperature Method Selection')

def Tprof(new_time):
    T_new = a*np.sin(2.*np.pi/lamda*new_time+ps)+b
    return T_new

def Ttable(new_time):

    #input Table: time [hours], Temperature [C]
    if new_time < 1E-8:
        T = weather_df[0, 1] + 273.15 #[K]
        return T
    else:
        time_hours = new_time/3600. 
        table_row_index = int(np.floor(time_hours/table_intervals)) 
        T_lin_intrpl = (weather_df[table_row_index + 1, 1] - weather_df[table_row_index, 1])/table_intervals*(time_hours - weather_df[table_row_index, 0]) \
            + weather_df[table_row_index,1]

        return T_lin_intrpl + 273.15

def ErgunEQ_PressureLoss():

    if T_inlet_method == 1:
        T_mean = b
    elif T_inlet_method == 2:
        T_max = np.amax(weather_df[:,1]) + 273.15 #K
        T_min = np.amin(weather_df[:,1]) + 273.15 #K
        T_mean = 0.5*(T_max + T_min)
    else:
        print('Cannot calculate pressure loss... Invalid inlet temperature specification')

    rhof_mean = pamb/Rgas[1]/T_mean #kg/m^3
    #cpf_mean = specheat(T_mean, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_mean = viscosity(T_mean, p_frc[1], rxC) #kg/m-s
    #kf_mean = conductivity(T_mean, p_frc[1], rxC) #W/m-K    #Unused
    G_mean = rhof_mean*V_in
    
    delta_p = L*(150.*muf_mean/Dp**2*(1-eps)**2/eps**3*V_in + 1.75*rhof_mean/Dp*(1-eps)/eps**3*V_in**2)

    return delta_p

def Schumman_Backward_Euler1(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc):

    #Note: Twice the size for each matrix and array due to coupling of Ts and Tf (Fluid and Solid combined)
    AA_Sc = np.array([[0.]*2*N]*2*N)
    RHS_Sc =np.array([[0.]*1]*2*N)

    #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
    AA_Sc[0,0]=1.

    #Matrix Inputs
    #Fluid
    # D_Scf = np.array([0.0]*N) #Ts, Tf
    ##Solid
    A_Scs = getA_Scs(rhos,cps)#dTs/dt #Constant 

    #Update Fluid BC at Left End
    RHS_Sc[0,0]=Tinf_new

    rhof_Sc = pamb/Rgas[1]/Tinf_old #kg/m^3
    cpf_Sc = specheat(Tinf_old, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_Sc = viscosity(Tinf_old, p_frc[1], rxC) #kg/m-s
    kf_Sc = conductivity(Tinf_old, p_frc[1], rxC) #W/m-K    #Unused
    G = rhof_Sc*V_in
        
    C_Scf = getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
    B_Scs = getB_Scs(C_Scf)
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs/dt_Sc + B_Scs
    AA_Sc[N,0] = -B_Scs
    RHS_Sc[N] = delta_Sc[N]*(A_Scs/dt_Sc)

    for j in range(1,N): #Begin for loop
        
        #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)
        T_cell_Scf=T_Sc[inc][j] 
        rhof_Sc=p_Sc[j]/Rgas[1]/T_cell_Scf #kg/m^3
        cpf_Sc=specheat(T_cell_Scf, p_frc[1], rxC)*1000./MW[1] #J/kg-K
        muf_Sc=viscosity(T_cell_Scf, p_frc[1], rxC) #kg/m-s
        kf_Sc=conductivity(T_cell_Scf, p_frc[1], rxC) #W/m-K    #Unused
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf = getA_Scf(rhof_Sc,cpf_Sc)
        B_Scf = getB_Scf(G, cpf_Sc)
        C_Scf = getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
        B_Scs = getB_Scs(C_Scf)
        
        #Matrix Coefficients and RHS
        if j < N-1: 
            #Second Order Accurate Central Differencing Scheme
            #Fluid  
            AA_Sc[j,j-1] = -B_Scf/(2.*dx_Sc) 
            AA_Sc[j,j] = A_Scf/dt_Sc + C_Scf
            AA_Sc[j,j+1] = B_Scf/(2.*dx_Sc)
            AA_Sc[j,j+N] = -C_Scf
            RHS_Sc[j] = delta_Sc[j]*(A_Scf/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N] = A_Scs/dt_Sc + B_Scs
            AA_Sc[j+N,j] = -B_Scs
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs/dt_Sc)
        else:
            #First Order Accurate Backward Differencing Scheme for Cells at Right End
            #Fluid 
            AA_Sc[j,j-1] = -B_Scf/dx_Sc
            AA_Sc[j,j] = A_Scf/dt_Sc + B_Scf/dx_Sc + C_Scf
            AA_Sc[j,j+N] = -C_Scf
            RHS_Sc[j] = delta_Sc[j]*(A_Scf/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N] = A_Scs/dt_Sc + B_Scs
            AA_Sc[j+N,j] = -B_Scs
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs/dt_Sc) #End for loop

    return AA_Sc, RHS_Sc, G

def Schumman_Crank_Nicholson(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc):
    
    #Note: Twice the size for each matrix and array due to coupling of Ts and Tf (Fluid and Solid combined)
    AA_Sc = np.array([[0.]*2*N]*2*N)
    RHS_Sc =np.array([[0.]*1]*2*N)

    #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
    AA_Sc[0,0]=1.

    #Matrix Inputs
    #Fluid
    # D_Scf = np.array([0.0]*N) #Ts, Tf
    ##Solid
    A_Scs = getA_Scs(rhos,cps)#dTs/dt #Constant 

    #Update Fluid BC at Left End
    RHS_Sc[0,0]=Tinf_new

    rhof_Sc = pamb/Rgas[1]/Tinf_old #kg/m^3
    cpf_Sc = specheat(Tinf_old, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_Sc = viscosity(Tinf_old, p_frc[1], rxC) #kg/m-s
    kf_Sc = conductivity(Tinf_old, p_frc[1], rxC) #W/m-K    #Unused
    G = rhof_Sc*V_in
    
    C_Scf = getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
    B_Scs = getB_Scs(C_Scf)
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs/dt_Sc + B_Scs/2.0
    AA_Sc[N,0] = -B_Scs/2.0
    RHS_Sc[N] = delta_Sc[N]*(A_Scs/dt_Sc - B_Scs/2.0) + delta_Sc[0]*B_Scs/2.0

    for j in range(1,N): #Begin for loop
        
        #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)
        T_cell_Scf = T_Sc[inc][j] 
        rhof_Sc = p_Sc[j]/Rgas[1]/T_cell_Scf #kg/m^3
        cpf_Sc = specheat(T_cell_Scf, p_frc[1], rxC)*1000./MW[1] #J/kg-K
        muf_Sc = viscosity(T_cell_Scf, p_frc[1], rxC) #kg/m-s
        kf_Sc = conductivity(T_cell_Scf, p_frc[1], rxC) #W/m-K    #Unused
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf = getA_Scf(rhof_Sc,cpf_Sc)
        B_Scf = getB_Scf(G,cpf_Sc)
        C_Scf =getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
        B_Scs = getB_Scs(C_Scf)
        
        #Matrix Coefficients and RHS
        if j < N-1: 
            #Second Order Accurate Central Differencing Scheme
            #Fluid  
            AA_Sc[j,j-1] = -B_Scf/(4.0*dx_Sc) 
            AA_Sc[j,j] = A_Scf/dt_Sc + C_Scf/2.0
            AA_Sc[j,j+1] = B_Scf/(4.0*dx_Sc)
            AA_Sc[j,j+N] = -C_Scf/2.0
            RHS_Sc[j] = delta_Sc[j]*(A_Scf/dt_Sc - C_Scf/2.0) + (delta_Sc[j-1] - delta_Sc[j+1])*B_Scf/(4.0*dx_Sc) + delta_Sc[j+N]*C_Scf/2.0
            #Solid
            AA_Sc[j+N,j+N] = A_Scs/dt_Sc + B_Scs/2.0
            AA_Sc[j+N,j] = -B_Scs/2.0
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs/dt_Sc - B_Scs/2.0) + delta_Sc[j]*B_Scs/2.0
        else:
            #First Order Accurate Backward Differencing Scheme for Cells at Right End
            #Fluid 
            AA_Sc[j,j-1] = -B_Scf/(2.0*dx_Sc)
            AA_Sc[j,j] = A_Scf/dt_Sc + B_Scf/(2.0*dx_Sc) + C_Scf/2.0
            AA_Sc[j,j+N] = -C_Scf/2.0
            RHS_Sc[j] = delta_Sc[j]*(A_Scf/dt_Sc - B_Scf/(2.0*dx_Sc) - C_Scf/2.0) + delta_Sc[j-1]*B_Scf/(2.0*dx_Sc) + delta_Sc[j+N]*C_Scf/2.0
            #Solid
            AA_Sc[j+N,j+N] = A_Scs/dt_Sc + B_Scs/2.0
            AA_Sc[j+N,j] = -B_Scs/2.0
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs/dt_Sc - B_Scs/2.0) + delta_Sc[j]*B_Scs/2.0

    return AA_Sc, RHS_Sc, G

def Enhanced_Schumman_Backward_Euler1(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc):

    #Note: Twice the size for each matrix and array due to coupling of Ts and Tf (Fluid and Solid combined)
    AA_Sc = np.array([[0.]*2*N]*2*N)
    RHS_Sc =np.array([[0.]*1]*2*N)

    #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
    AA_Sc[0,0]=1.

    #Matrix Inputs
    #Fluid
    # D_Scf = np.array([0.0]*N) #Ts, Tf
    ##Solid
    A_Scs = getA_Scs(rhos,cps)#dTs/dt #Constant 
    #C_Scs = ks  #Enhanced Schumann #Constant
    C_Scs = ks*(1 - eps) #In line with Fluent
    #Update Fluid BC at Left End
    RHS_Sc[0,0]=Tinf_new

    rhof_Sc = pamb/Rgas[1]/Tinf_old #kg/m^3
    cpf_Sc = specheat(Tinf_old, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_Sc = viscosity(Tinf_old, p_frc[1], rxC) #kg/m-s
    kf_Sc = conductivity(Tinf_old, p_frc[1], rxC) #W/m-K   
    G = rhof_Sc*V_in
        
    C_Scf = getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
    B_Scs = getB_Scs(C_Scf)
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs/dt_Sc + B_Scs
    AA_Sc[N,0] = -B_Scs
    RHS_Sc[N] = delta_Sc[N]*(A_Scs/dt_Sc)

    for j in range(1,N): #Begin for loop
        
        #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)
        T_cell_Scf=T_Sc[inc][j] 
        rhof_Sc=p_Sc[j]/Rgas[1]/T_cell_Scf #kg/m^3
        cpf_Sc=specheat(T_cell_Scf, p_frc[1], rxC)*1000./MW[1] #J/kg-K
        muf_Sc=viscosity(T_cell_Scf, p_frc[1], rxC) #kg/m-s
        kf_Sc=conductivity(T_cell_Scf, p_frc[1], rxC) #W/m-K   
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf = getA_Scf(rhof_Sc,cpf_Sc)
        B_Scf = getB_Scf(G, cpf_Sc)
        C_Scf = getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
        #E_Scf = kf_Sc #Enhanced Schumann
        E_Scf = kf_Sc*eps #In line with Fluent
        B_Scs = getB_Scs(C_Scf)
        
        #Matrix Coefficients and RHS
        if j < N-1: 
            #Second Order Accurate Central Differencing Scheme
            #Fluid  
            AA_Sc[j,j-1] = -B_Scf/(2.*dx_Sc) -E_Scf/dx_Sc**2.
            AA_Sc[j,j] = A_Scf/dt_Sc + C_Scf + 2.*E_Scf/dx_Sc**2.
            AA_Sc[j,j+1] = B_Scf/(2.*dx_Sc) - E_Scf/dx_Sc**2.
            AA_Sc[j,j+N] = -C_Scf
            RHS_Sc[j] = delta_Sc[j]*(A_Scf/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N-1] = -C_Scs/dx_Sc**2.
            AA_Sc[j+N,j+N] = A_Scs/dt_Sc + B_Scs +2.*C_Scs/dx_Sc**2.
            AA_Sc[j+N,j+N+1] = -C_Scs/dx_Sc**2.
            AA_Sc[j+N,j] = -B_Scs
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs/dt_Sc)
        else:
            #First Order Accurate Backward Differencing Scheme for Cells at Right End
            #Fluid 
            AA_Sc[j,j-2] = -E_Scf/dx_Sc**2.
            AA_Sc[j,j-1] = -B_Scf/dx_Sc + 2.*E_Scf/dx_Sc**2.
            AA_Sc[j,j] = A_Scf/dt_Sc + B_Scf/dx_Sc + C_Scf - E_Scf/dx_Sc**2.
            AA_Sc[j,j+N] = -C_Scf
            RHS_Sc[j] = delta_Sc[j]*(A_Scf/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N-2] = -C_Scs/dx_Sc**2
            AA_Sc[j+N,j+N-1] = 2.*C_Scs/dx_Sc**2
            AA_Sc[j+N,j+N] = A_Scs/dt_Sc + B_Scs - C_Scs/dx_Sc**2
            AA_Sc[j+N,j] = -B_Scs
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs/dt_Sc) #End for loop

    return AA_Sc, RHS_Sc, G

def NextTimeStep(CFL, G, t_Sc, dx_Sc, T_Sc, dt_old):
    
    T_max = np.amax(T_Sc[len(t_Sc)-1])
    rhof_min = pamb/Rgas[1]/T_max #kg/m^3
    Vsuperficial_max = G/rhof_min
    dt_new = dt_old
    dt_new = CFL*dx_Sc/Vsuperficial_max
    if dt_new/dt_old > 1.2:
        dt_new = 1.2*dt_old
    #elif dt_new/dt_old < 0.8:
    #    dt_new = 0.8*dt_old
    #if dt_new < 60.:
    #    dt_new = 60.
    
    return dt_new

def runtime_Update(current_time, max_time, num_runtime_updates):
    if current_time > 0.25*max_time and num_runtime_updates == 0:
        print('Solution Progress: 25%')
        num_runtime_updates += 1
    if current_time > 0.5*max_time and num_runtime_updates <= 1:
        print('Solution Progress: 50%')
        num_runtime_updates += 1
    if current_time > 0.75*max_time and num_runtime_updates <= 2:
        print('Solution Progress: 75%')
        num_runtime_updates += 1
    return num_runtime_updates

def post_process():
    
    time_hours = t_Sc/3600.
    time_days = t_Sc/(3600.*24.)
    time_years = t_Sc/(3600.*24.*365.)

    if time_increments_post_processing_figures == 1:
        time_Post = time_hours
    elif time_increments_post_processing_figures == 2:
        time_Post = time_days
    elif time_increments_post_processing_figures == 3:
        time_Post = time_years
    else:
        print('Invalid selection for time increments in post processing')

    #Cell Index at X-locations of 0*L, 0.25*L, 0.5*L, 0.75*L, and 1*L 
    #Fluid
    cell_f0 = 0
    cell_f1 = int(np.ceil(N/4))
    cell_f2 = int(np.ceil(N/2))
    cell_f3 = int(np.ceil(3*N/4))
    cell_f4 = N-1 #int(np.ceil(47*N/48))
    #Solid
    cell_s0 = 0 + N
    cell_s1 = int(np.ceil(N/4)) + N
    cell_s2 = int(np.ceil(N/2)) + N
    cell_s3 = int(np.ceil(3*N/4)) + N
    cell_s4 = 2*N - 1 #int(np.ceil(47*N/48)) + N

    cell_ID = np.array([cell_f0, cell_f1, cell_f2, cell_f3, cell_f4, cell_s0, cell_s1, cell_s2, cell_s3, cell_s4])
    cell_phase = ['fluid', 'fluid', 'fluid', 'fluid', 'fluid', 'solid', 'solid', 'solid', 'solid', 'solid']

    x_Post = np.array([0.]*len(cell_ID))
    for i in range(0, len(cell_ID)):
        if i < len(cell_ID)/2:
            x_Post[i] = cell_ID[i]*L/N 
        else:
            x_Post[i] = cell_ID[i]*L/N - L

    T_Post = np.array([[0.]*len(x_Post)]*len(T_Sc))
    for i in range(0, len(x_Post)):
        for j in range(0, len(T_Sc)):
            T_Post[j,i] = T_Sc[j,cell_ID[i]] - 273.15

    #Heating/Cooling Rate 
    #Rock Mass
    dTsdt_Post = np.array([[0.]*N]*len(t_Sc))
    for i in range(0,(len(t_Sc)-1)):
        for j in range(0,N):
            dTsdt_Post[i+1,j] = (T_Sc[i+1,j+N]-T_Sc[i,j+N])/(t_Sc[i+1]-t_Sc[i])
    dTsdt_Post[-1,:] = dTsdt_Post[-2,:]
    dTsdt_Post[0,:] = dTsdt_Post[1,:]
    dQdt_Post = np.sum(dTsdt_Post, axis=1)*area*dx_Sc*(1-eps)*rhos*cps
    #Air Enthalpy Change (Equivalent Expression)
    #dQdt_Post2 = np.array([0.]*len(t_Sc))
    #for i in range(0,(len(t_Sc)-1)):
    #    Tfave_Post2 = 0.5*(T_Sc[i,N-1] + T_Sc[i,0])
    #    cpave_Post2 = specheat(Tfave_Post2, p_frc[1], rxC)*1000./MW[1]
    #    dmdt_Post2 = Q_in*(p_Sc[0]/Rgas[1]/T_Sc[i,0])
    #    dQdt_Post2[i] = dmdt_Post2*cpave_Post2*(T_Sc[i,N-1]-T_Sc[i,0])
    #dQdt_Post2[-1] = dQdt_Post2[-2] 
    
    #For plotting
    #Available lines: 'solid', 'dashed', 'dashdot', 'dotted'
    plot_lines = ['solid','solid','solid','solid','solid','dashdot','dashdot','dashdot','dashdot','dashdot'] 
    #Available colours: blue ('b'), green ('g'), red ('r'), cyan ('c', magenta ('m'), yellow ('y'), black ('k'), white ('w')
    plot_colors = ['r','b','g','y','c','r','b','g','y','c']
    #Available markers: 'o','d','s','*','<','>','^','v','1','2','3','4']
    #plot_markers = []
    Case_Description_Output()
    writeCSV(T_Post, time_hours)
    plot(x_Post, T_Post, time_Post, plot_lines, plot_colors, dQdt_Post)

def Case_Description_Output():
    
    workingDir = os.getcwd()
    if os.name == 'nt':
        dirName = workingDir + r'\Solutions\Case_Description'
    else:
        dirName = workingDir + r'/Solutions/Case_Description'

    file = open(dirName+'\\'+fileName+'_Description.txt', 'w')
    file.write('Case Description for ' + fileName + ':\n')
    if T_inlet_method == 1:
        file.write('Inlet Temperature Specified by User-Defined Function \n')
    elif T_inlet_method == 2:
        file.write('Inlet Temperature Specified by CSV Table Input \n')
    if time_integrator == 1:
        file.write('Backward Euler 1st Order Time Integration \n')
    elif time_integrator == 2:
        file.write('Crank Nicholson (i.e. Trapezoidal Rule) 2nd Order Time Integration \n')
    file.write('Column Diameter or Side Length: %.3f m \nColumn Height: %.3f m \nParticle Diameter: %.3f m \nVoid Fraction: %.3f \n' % (D, L, Dp, eps))
    file.write('Flow rate: %.4f m^3/s or %.1f cfm \nSuperficial Velocity: %.3f m/s \nReynolds Number (Ave): %.1f \n' % (Q_in, Qcfm_in, V_in, Re_mean))
    file.write('Heat Transfer Coefficient and Effective HTC (Ave): %.2f W/(m^2K) and %.2f W/(m^2K)\nBiot Number (Ave): %.2f \n' % (hp_mean, hp_effective_mean, Bi_mean))
    file.write('CFL Number: %.2f \nAverage Time Step Size: %.2f s \n' % (CFL, CFL*dx_Sc/V_in))
    if time_increments_post_processing_figures == 1:
        file.write('Maximum Solution Time: %.2f hr \n' % (max_time/3600.))
    elif time_increments_post_processing_figures == 2:
        file.write('Maximum Solution Time: %.2f days\n' % (max_time/(3600.*24.)))
    elif time_increments_post_processing_figures == 3:
        file.write('Maximum Solution Time: %.2f yr\n' % (max_time/(3600.*24.*365.)))
    file.write('Ergun EQ Pressure Loss: %.1f Pa \n' % (delta_p))
    file.close()
  
    return 

def writeCSV(T_Post, time_hours):
    print('Writing CSV Files...')

    workingDir = os.getcwd()
    if os.name == 'nt':
        dirName = workingDir + r'\Solutions\CSV'
    else:
        dirName = workingDir + r'/Solutions/CSV'
    #path = Path(dirName)
    #path.mkdir(parents=True)

    with open(dirName+'\\'+fileName+'_CSV.csv', mode='w', newline='') as CSV_file:
        fieldnames = ['time_hours', 'Tf_0*L', 'Tf_0.25*L', 'Tf_0.50*L', 'Tf_0.75*L', 'Tf_1*L', 'Ts_0*L', 'Ts_0.25*L', 'Ts_0.50*L', 'Ts_0.75*L', 'Ts_1*L']
        write_file = csv.DictWriter(CSV_file, fieldnames=fieldnames)
        write_file.writeheader()
        for i in range(0,len(T_Sc)):
            write_file.writerow({'time_hours': '%.6f' %time_hours[i], 'Tf_0*L': '%.2f' %T_Post[i,0], 'Tf_0.25*L': '%.2f' %T_Post[i,1], 'Tf_0.50*L': '%.2f' %T_Post[i,2], 
                                       'Tf_0.75*L': '%.2f' %T_Post[i,3], 'Tf_1*L': '%.2f' %T_Post[i,4], 'Ts_0*L': '%.2f' %T_Post[i,5], 
                                       'Ts_0.25*L': '%.2f' %T_Post[i,6], 'Ts_0.50*L': '%.2f' %T_Post[i,7], 'Ts_0.75*L': '%.2f' %T_Post[i,8], 
                                       'Ts_1*L': '%.2f' %T_Post[i,9]})
               
    ##For Full Data Set, Uncomment Below
    #
    #Full_Data_Set = np.array([[0.]*(len(T_Sc[0]) + 1)]*len(t_Sc))
    #Full_Data_Set[:,0] = t_Sc[:] 
    #for i in range(0, len(t_Sc)):
    #    for j in range(0, len(T_Sc[0])):
    #        Full_Data_Set[i,j+1] = T_Sc[i,j]

    #fullDataSet = open(dirName+'\\'+fileName+'_FullDataSet.csv', mode='w', newline='')

    #with fullDataSet:

    #    writer = csv.writer(fullDataSet)
    #    writer.writerows(Full_Data_Set)

    print('CSV Files Complete')
    return

def plot(x_Post, T_Post, time_Post, plot_lines, plot_colors, dQdt_Post):
    ## plotting
    print('Plotting...')
    
    workingDir = os.getcwd()
    if os.name == 'nt':
        dirName = workingDir + r'\Solutions\Figures'
    else:
        dirName = workingDir + r'/Solutions/Figures'
    # fileName = 'test.png'

    if time_increments_post_processing_figures == 1:
        plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f hours, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [hr]' 
        ylabel1= 'T [C]'
    elif time_increments_post_processing_figures == 2:
        plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f days, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [days]' 
        ylabel1= 'T [C]'
    elif time_increments_post_processing_figures == 3:
        plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f years, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [yr]' 
        ylabel1= 'T [C]'
    
    fig, ax1 = plt.subplots()
    ax1title=(plot_title)
    plt.title(ax1title)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1) 
    ax1.tick_params(axis='y') 

    for i in range(0,len(x_Post)):
        ax1.plot(time_Post,T_Post[:,i],color=plot_colors[i],linestyle=plot_lines[i],label='x = %.2f' % x_Post[i]) 

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)

    #rootdir = r'C:\Users\pgareau\Source\repos\NHEET-1D-model\NHEET-1D-model\Figures'
    workingDir = os.getcwd()
    if os.name == 'nt':
        dirName = workingDir + r'\Solutions\Figures'
    else:
        dirName = workingDir + r'/Solutions/Figures'
    # fileName = 'test.png'
    plt.savefig(dirName+'/'+fileName+'_T',bbox_inches = 'tight') #dpi=100

    ####Second Plot - Heating and Cooling rate####
    if time_increments_post_processing_figures == 1:
        plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f hours, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [hr]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 2:
        plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f days, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [days]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 3:
        plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f years, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [yr]' 
        ylabel1= 'dQ/dt [W]'

    fig, ax1 = plt.subplots()
    ax1title=(plot_title)
    plt.title(ax1title)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1) 
    ax1.tick_params(axis='y') 

    ax1.plot(time_Post,dQdt_Post,color=plot_colors[0],linestyle=plot_lines[0],label='dQ/dt_Solids') 
    #ax1.plot(time_Post,dQdt_Post2,color=plot_colors[0],linestyle=plot_lines[0],label='dQ/dt_Solids') 

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)

    plt.savefig(dirName+'/'+fileName+'_dQdt',bbox_inches = 'tight') #dpi=100

    print('Plots Complete')
    print('Post Processing Complete')
    
    plt.show()
    plt.close(fig)
    return

def Mine_HC_Potential():
    #Inputs
    T_Mine = 273.15 + 40.

    T_ambient = T_Sc[:,0]
    T_NHEET_Outlet = T_Sc[:,N-1]
    cp_ambient = np.array([0.0]*len(T_Sc))
    cp_NHEET_Outlet = np.array([0.0]*len(T_Sc))
    HC_potential_NHEET = np.array([0.0]*len(T_Sc)) 
    HC_potential_ambient = np.array([0.0]*len(T_Sc)) 
    for x in range(0,len(T_Sc)):
        cp_ambient[x] = specheat(T_ambient[x], p_frc[1], rxC)*1000./MW[1]
        cp_NHEET_Outlet[x] = specheat(T_NHEET_Outlet[x], p_frc[1], rxC)*1000./MW[1]
        HC_potential_ambient[x] = Q_in*p_Sc[0]/Rgas[1]/T_ambient[x]*cp_ambient[x]*(T_ambient[x] - T_Mine) #dm/dt*cp*deltaT
        HC_potential_NHEET[x] = Q_in*p_Sc[0]/Rgas[1]/T_NHEET_Outlet[x]*cp_NHEET_Outlet[x]*(T_NHEET_Outlet[x] - T_Mine) #dm/dt*cp*deltaT

    time_hours = t_Sc/3600.
    time_years = t_Sc/(3600.*24.)
    time_years = t_Sc/(3600.*24.*365.)

    if time_increments_post_processing_figures == 1:
        time_Post = time_hours
    elif time_increments_post_processing_figures == 2:
        time_Post = time_days
    elif time_increments_post_processing_figures == 3:
        time_Post = time_years
    else:
        print('Invalid selection for time increments in post processing')

    print('Plotting Mine Heating and Cooling Potential...')

    if time_increments_post_processing_figures == 1:
        plot_title=fileName+'\nMine Heating and Cooling Potential \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f hours, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [hr]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 2:
        plot_title=fileName+'\nMine Heating and Cooling Potential \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f days, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [days]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 3:
        plot_title=fileName+'\nMine Heating and Cooling Potential \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f years, dt_final = %.2f s' % (V_in, Qcfm_in, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [yr]' 
        ylabel1= 'dQ/dt [W]'

    fig, ax1 = plt.subplots()
    ax1title=(plot_title)
    plt.title(ax1title)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1) 
    ax1.tick_params(axis='y') 

    ax1.plot(time_Post,HC_potential_ambient,color='r',linestyle='solid',label='Ambient Air')
    ax1.plot(time_Post,HC_potential_NHEET,color='b',linestyle='solid',label='NHEET')

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)

    workingDir = os.getcwd()
    if os.name == 'nt':
        dirName = workingDir + r'\Solutions\Figures'
    else:
        dirName = workingDir + r'/Solutions/Figures'

    plt.savefig(dirName+'/'+fileName+'_MHCP',bbox_inches = 'tight') #dpi=100

    plt.show()
    plt.close(fig)
    return

##########################
#####     INPUTS     #####
##########################

### Output File Name ###
fileName = 'Exp2-Fluent-Comparison-Enhanced-Schumann-Test3'

### Ambient Pressure ###
pamb = 101325. #[Pa]

### Initial Rock Temperature ###
T0 = 293.15 #[K]

### Inlet Temperature ###
#Choose Between Two Methods: 1 = User-Defined Function, 2 = Input CSV Table
T_inlet_method = 2
### Method 1 (Profile) ###
#Constants: a = amplitude, b= average value, ps= phase shift, lamda = wavelength (cycle duration) #time is in seconds
a = 10. #[K]
b = 293.15 #[K]
lamda = 3600.0*24. #[s]
ps = np.arcsin((T0-b)/a) #[rad]
### Method 2 (Input CSV Table) ###
#Temperature from table - #input Table must have a header, and its first and second column must be time [hr] and Temperature [C]
weather_df = np.genfromtxt('Input_Hourly_Weather_Data_July2019.csv', delimiter=",", dtype = float, skip_header = 1)

### Volumetric Flow Rate (constant) ###
Qcfm_in = 525.0 #cfm

#### Geometry ###
#Rock Mass
D = 14.*0.3048 #m #ID of Sch80 18 in pipe is 0.407 m
L = 8.*0.3048 #m
area = D**2 #np.pi/4.0*D**2 #m^2
#Particle Diameter
Dp = 0.02
#Insulation Thickness # Not using for now
#ithk = 0.1 #m
#Porosity
eps = 0.3

#Solid Material Properties
### Granite
rhos = 2635. #kg/m^3 #check 2418.788
cps = 790. #J/kg-K
ks = 2.6 #W/m-K

### Spatial Discretization: N = # of points ###
N = 150+1

### Transient Solution Parameters ###
#Temporal Discretization 
first_time_step_size = 120. #3600.*0.25
### Maximum Time - Method 1 (Profile) ###
max_time = 3600.*24.*15.
### Maximum Time - Method 2 (Input CSV Table) ###
if T_inlet_method == 2:
    Final_hour = weather_df[-2,0] #last data point is clipped to ensure interpolation does not fail
    table_intervals = weather_df[1,0] - weather_df[0,0] #intervals must be evenly spaced
    max_time = Final_hour*3600.0 *10./31. #seconds
### Courant Friedrichs Lewy Number
CFL = 100.

### Include Thermal Conductivity (i.e. Enhanced Schumann) ###
    # 0 = Do not include (Schumann Model)
    # 1 = Include (Enhanced Schumann Model) - Cannot use Crank Nicholson Time Integrator right now. Note: Enhanced Schumann conductivity is not the same as Fluent conductivity, which includes void fraction.
Enhanced_Schumann = 1

### Time Integrator ###
## Choose Between Two Methods: 
    # 1 = Backward Euler - First Order
    # 2 = Crank Nicholson (i.e. Trapezoidal Rule) - Second Order
time_integrator = 1

### Time units in post-processed figures ###
## Choose Between Two Units: 
    # 1 = Hours
    # 2 = Days
    # 3 = Years
time_increments_post_processing_figures = 2

### #Abdel-Ghaffar, E. A.-M., 1980. PhD Thesis - Here for now for quick overwrite of input variables
#a = 44.25 #[K]
#b = 274.9 #[K]
#lamda = 3600.0*4. #[s]
#ps = np.arcsin((T0-b)/a) #[rad]
#
### Geometry 
#Rock Mass
#D = 0.3048 #m
#L= 1.2192 #m
#area = D**2 #m^2
##Particle Diameter
#Dp = 0.028
#eps = 0.4537
#
### Solid Material Properties
#rhos = 2418.79 #kg/m^3
#cps = 908.54 #J/kg-K
#ks = 1.2626 #W/m-K
###

Vsuperficial = Qcfm_in/2118.88/area
Re_mean = 1.125*Vsuperficial*Dp/1.81E-5 #m/s
Nup_mean = ((1.18*Re_mean**0.58)**4 + (0.23*(Re_mean/(1 - eps))**0.75 )**4)**0.25
hp_mean = Nup_mean*0.02436/Dp
hp_effective_mean = 1/(Dp/(Nup_mean*0.02436) + Dp/(10*ks)) #Effective heat transfer coefficient accounting for Biot Number Effects
Bi_mean = hp_mean*Dp/ks

##############################################
#####     Setup and Initialization     #######
##############################################

print('Setting Up and Initializing . . .')

MW,Rgas,p_frc,z_st,p_mol,ptot,rxC = reaction_balance()

#Flow rate calculations at Inlet
Q_in = Qcfm_in/2118.88 #m^3/s
V_in = Q_in/area #m/s
#Ergun EQ delta_p
delta_p = ErgunEQ_PressureLoss()

#Fluid Properties Initialization
p_Sc = np.array([pamb]*N) #Pressure gradient - Assumed constant everywhere for now
Tinf0 = T_inlet(0.)

#At Inlet at t=0
rhof_in = p_Sc[0]/Rgas[1]/Tinf0 #kg/m^3
cpf_in=specheat(Tinf0, p_frc[1], rxC)*1000./MW[1] #J/kg-K
muf_in=viscosity(Tinf0, p_frc[1], rxC) #kg/m-s
kf_in=conductivity(Tinf0, p_frc[1], rxC) #W/m-K
G_0=rhof_in*V_in

#Particle Reynolds and Prandtl at Inlet
Rep_in = G_0*Dp/muf_in
Pr_in = muf_in*cpf_in/kf_in

#Initialize a time array 
t_Sc=np.array([0.])

#Mesh points in space
x_Sc = np.linspace(0.,L,num=N) #Linear space spanning 0 to L with N equally spaced points
dx_Sc = x_Sc[1]-x_Sc[0] #space between points 

########## Matrix and Array Initialization ##########

T_Sc = np.array([[T0]*2*N]) #initialization of T0 everywhere
T_Sc[0,0] = Tinf0 #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
delta_Sc = np.array([[T0]*1]*2*N) #delta is a temporary array of T at time t-1 - initialization
delta_Sc[0,0] = Tinf0

print('Setup and Initialization Complete\n')

print('Setup Description:')
print('Filename: ' +fileName) 
if T_inlet_method == 1:
    print('Inlet Temperature Specified by User-Defined Function')
elif T_inlet_method == 2:
    print('Inlet Temperature Specified by CSV Table Input')
else:
    print('Error - Invalid Inlet Temperature Method Selection')
if time_integrator == 1:
    print('Backward Euler 1st Order Time Integration')
elif time_integrator == 2:
    print('Crank Nicholson (i.e. Trapezoidal Rule) 2nd Order Time Integration')
else:
    print('Error - Invalid Time Integration Method Selection')
print('Column Diameter or Side Length: %.3f m \nColumn Height: %.3f m \nParticle Diameter: %.3f m \nVoid Fraction: %.3f' % (D, L, Dp, eps))
print('Flow rate: %.4f m^3/s or %.1f cfm \nSuperficial Velocity: %.3f m/s \nReynolds Number (Ave): %.1f' % (Q_in, Qcfm_in, V_in, Re_mean))
print('Heat Transfer Coefficient and Effective HTC (Ave): %.2f W/(m^2K) and %.2f W/(m^2K)\nBiot Number (Ave): %.2f ' % (hp_mean, hp_effective_mean, Bi_mean))
print('CFL Number: %.2f \nAverage Time Step Size: %.2f s' % (CFL, CFL*dx_Sc/V_in))
print('Maximum Solution Time: %.2f hr' % (max_time/3600.))
print('Ergun EQ Pressure Loss: %.1f Pa or %.3f in. w.g.\n' % (delta_p, delta_p/248.84))

######################################
#####     Transient Solver     #######
######################################

print('Solving . . .')

tic() #Start timekeeping (tictoc) to determine how long computation takes
current_time = t_Sc[-1]
new_time = current_time + first_time_step_size
dt_Sc = first_time_step_size
Tinf_old = Tinf0
Tinf_new = Tinf0
inc = 0
num_runtime_updates = 0

while current_time < max_time:

    ##### At Inlet #####
    Tinf_old = Tinf_new
    Tinf_new = T_inlet(new_time)
    
    ##### Get Matrix AA and Vector RHS #####
    if Enhanced_Schumann == 1:
        if time_integrator == 1:
            AA_Sc,RHS_Sc, G = Enhanced_Schumman_Backward_Euler1(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc)
        else:
            print("Incorrect Time Integrator Selection")
            break
    elif Enhanced_Schumann == 0:
        if time_integrator == 1:
            AA_Sc,RHS_Sc, G = Schumman_Backward_Euler1(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc)
        elif time_integrator == 2:
            AA_Sc,RHS_Sc, G = Schumman_Crank_Nicholson(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc)
        else:
            print("Incorrect Time Integrator Selection")
            break
    else:
        print("Incorrect Model Selection")
        break
    ##### Solve temperature at updated time "new_time" #####
    delta_Sc=np.linalg.lstsq(AA_Sc,RHS_Sc,rcond=-1)[0] #Returns the least-squares solution to a linear matrix equation

    T_Sc=np.vstack([T_Sc,delta_Sc.T]) #Stack transpose of delta array (now horizontal) to T(t) solution matrix
    t_Sc = np.append(t_Sc, new_time) #Append time values to the end of an array

    dt_Sc = NextTimeStep(CFL, G, t_Sc, dx_Sc, T_Sc, dt_Sc)
    current_time = t_Sc[-1]
    new_time = current_time + dt_Sc
    inc+=1
    num_runtime_updates = runtime_Update(current_time, max_time, num_runtime_updates)

print('Solving Complete. \nFinal solution time: %.2f hr' % (current_time/3600.))
toc() #End timekeeping 

#####################################
#####     Post-Processing     #######
#####################################

print('Post Processing . . .')
post_process()
#Mine_HC_Potential() #Mine heating and cooling potential

####### END #######
print('END')