##### TITLE BLOCK #####
# Created by P Gareau, 2020
#
# Description: 
# -functions to be used by NHEET_main_looper to obtain solutions within a parametric study
# -completes a 1D conservation of energy analysis on a packed bed
#
# Input functions:
# -fluid_properties12.py 
# -isentropic_formulas.py (not used?)
#
# Output: fluid and solid temperatures at axial location and time
#
### Revisions ###
# Jan 15,2021 by D Cerantola
# -explicitly defined global variables
# -added functions Koekemoer_PressureLoss, NusseltNumber, VoidFraction (not used), inputs_const, inputs_var, findminmax, post_process_responses 
# -turned iterative part of code into a function iterate
# -modified iterative while loop to end when downstream end of bed obtains a periodic solution (sinusoidal inlet temperature profile required)
# -time updates given for periods
#
# Feb10,2021
# - inputs_const: velocity and number of timesteps inputs
# - post_process_responses: file export
#
# Feb11,2021
# - removed reaction_balance arrays
# - made constant massflow
# - added velocity, static pressure, and mach number outputs
# - predictor corrector inner loop to update material properties with current temperature
#
# Mar 20,2021
# - Constant Nu(x) evaluated using T(x=0,t)
# - addded Enhanced_Schumman_Backward_Euler1
# 
#
# Notes
# in jupyter lab .py file: ctrl + [ is de-indent
#
# Copyright 2020

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
import sys
import csv
from termcolor import cprint
from pathlib import Path
from tictoc import tic, toc

#A fluid properties solver intended for combustion applications is used here to obtain relationships for SI properties of dry air as a function of temperature.
#Inputs below are selected so that properties of solely dry air are provided.
from fluid_properties12 import chem_rxn,MolWt,specheat,specheatk, \
    enthalpy, enthalpy_Tnew, enthalpy_T, entropy,viscosity,conductivity

from NHEET_Tprofile_fns import findminmax, post_process_responses
from NHEET_other_fns import reaction_balance,ptps,ErgunEQ_PressureLoss,Koekemoer_PressureLoss,NusseltNumber,VoidFraction

print('Libraries loaded')

#Schumann Model Functions
def getA_Scf(rhof,cpf):
    A_Scf = eps*rhof*cpf
    return A_Scf

def getB_Scf(G,cpf):
    B_Scf = G*cpf
    return B_Scf

def getC_Scf(Rep,Pr, kf, eps,Nu_method):    
    Nup = NusseltNumber(eps,Rep,Pr,Nu_method)
    hp = Nup*kf/Dp
    #hp = 17.773    #Abdel-Ghaffar, E. A.-M., 1980. PhD Thesis
    #hp=1/(Dp/(Nup*kf) + Dp/(10*ks)) 
    #hp=1/(Dp/(Nup*kf) + Dp/(10*3.))
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

def T_inlet(new_time,Tprof_consts):
    if T_inlet_method == 1:
        T = Tprof(new_time,Tprof_consts)
        return T
    elif T_inlet_method == 2: 
        T = Ttable(new_time)
        return T
    else:
        print('Error... Invalid Inlet Temperature Method Selection')

def Tprof(new_time,Tprof_consts):
    a,b,T0,lamda = Tprof_consts
    ps = np.arcsin((T0-b)/a) #[rad]
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

def Schumman_Backward_Euler1(dt_Sc, Tinf_new,delta_Sc,p_Sc,V,fluidprops):
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
   
    rhof_Sc = fluidprops[0,0] #Nu requires Tf(y=0)
    cpf_Sc = fluidprops[1,0]
    muf_Sc = fluidprops[2,0]
    kf_Sc = fluidprops[3,0]
    
    G = rhof_Sc*V[0]
    Rep = G*Dp/muf_Sc
    Pr = muf_Sc*cpf_Sc/kf_Sc
        
    C_Scf = getC_Scf(Rep,Pr,kf_Sc, eps,Nu_method)
    B_Scs = getB_Scs(C_Scf)
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs/dt_Sc + B_Scs
    AA_Sc[N,0] = -B_Scs
    RHS_Sc[N] = delta_Sc[N]*(A_Scs/dt_Sc)

    for j in range(1,N): #Begin for loop  
#         #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)    
        rhof_Sc = fluidprops[0,j] #Nu requires Tf(y=0)
        cpf_Sc = fluidprops[1,j]
        muf_Sc = fluidprops[2,j]
        kf_Sc = fluidprops[3,j]
             
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf = getA_Scf(rhof_Sc,cpf_Sc)
        B_Scf = getB_Scf(G, cpf_Sc)
        C_Scf = getC_Scf(Rep,Pr, kf_Sc, eps,Nu_method)
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

def Schumman_Crank_Nicholson(dt_Sc, Tinf_new,delta_Sc,p_Sc,V,fluidprops): 
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

    #Update Fluid BC at LE
    RHS_Sc[0,0]=Tinf_new

    rhof_Sc = fluidprops[0,0]
    cpf_Sc = fluidprops[1,0]
    muf_Sc = fluidprops[2,0]
    kf_Sc = fluidprops[3,0]    
    
    G = rhof_Sc*V[0]
    Rep = G*Dp/muf_Sc
    Pr = muf_Sc*cpf_Sc/kf_Sc
    
    C_Scf = getC_Scf(Rep,Pr, kf_Sc, eps,Nu_method)
    B_Scs = getB_Scs(C_Scf)
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs/dt_Sc + B_Scs/2.0
    AA_Sc[N,0] = -B_Scs/2.0
    RHS_Sc[N] = delta_Sc[N]*(A_Scs/dt_Sc - B_Scs/2.0) + delta_Sc[0]*B_Scs/2.0

    for j in range(1,N): #Begin for loop       
#         #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)        
        rhof_Sc = fluidprops[0,j]
        cpf_Sc = fluidprops[1,j]
        muf_Sc = fluidprops[2,j]
        kf_Sc = fluidprops[3,j]
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf = getA_Scf(rhof_Sc,cpf_Sc)
        B_Scf = getB_Scf(G,cpf_Sc)
        C_Scf =getC_Scf(Rep,Pr, kf_Sc, eps,Nu_method)
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

#def Enhanced_Schumman_Backward_Euler1(Tinf_old, Tinf_new, p_Sc, T_Sc, delta_Sc):
def Enhanced_Schumman_Backward_Euler1(dt_Sc, Tinf_new,delta_Sc,p_Sc,V,fluidprops):

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

    rhof_Sc = fluidprops[0,0]
    cpf_Sc = fluidprops[1,0]
    muf_Sc = fluidprops[2,0]
    kf_Sc = fluidprops[3,0]  
    
    G = rhof_Sc*V[0]
    Rep = G*Dp/muf_Sc
    Pr = muf_Sc*cpf_Sc/kf_Sc
    
    C_Scf = getC_Scf(Rep,Pr,kf_Sc, eps,Nu_method)
    B_Scs = getB_Scs(C_Scf)
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs/dt_Sc + B_Scs
    AA_Sc[N,0] = -B_Scs
    RHS_Sc[N] = delta_Sc[N]*(A_Scs/dt_Sc)

    for j in range(1,N): #Begin for loop
        
        #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)
        rhof_Sc = fluidprops[0,j]
        cpf_Sc = fluidprops[1,j]
        muf_Sc = fluidprops[2,j]
        kf_Sc = fluidprops[3,j]
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf = getA_Scf(rhof_Sc,cpf_Sc)
        B_Scf = getB_Scf(G, cpf_Sc)
        #C_Scf = getC_Scf(G, Dp, muf_Sc, kf_Sc, eps)
        C_Scf =getC_Scf(Rep,Pr, kf_Sc, eps,Nu_method)
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

def NextTimeStep(CFL, G, t_Sc, dx_Sc, T_Sc, dt_old,rxn_consts):
    MW,Rgas,p_frc,z_st,p_mol,ptot,rxC=rxn_consts
    T_max = np.amax(T_Sc[len(t_Sc)-1])
    rhof_min = pamb/Rgas/T_max #kg/m^3
    Vsuperficial_max = G/rhof_min
    
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

def post_process(n,dirName):
    global time_Post
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
    cell_f4 = N-1
    #Solid
    cell_s0 = 0 + N
    cell_s1 = int(np.ceil(N/4)) + N
    cell_s2 = int(np.ceil(N/2)) + N
    cell_s3 = int(np.ceil(3*N/4)) + N
    cell_s4 = 2*N - 1

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
    Case_Description_Output(n,dirName)
    writeCSV(T_Post, time_hours,n,dirName)
    plot(x_Post, T_Post, time_hours, plot_lines, plot_colors, dQdt_Post,n,dirName)

def Case_Description_Output(n,dirName):
    #file = open('./Solutions/' + fileName+'_Description'+n+'.txt', 'w')
    file = open(dirName + fileName+'_Description'+n+'.txt', 'w')
    file.write('Case Description for ' + fileName + n +':\n')
    if T_inlet_method == 1:
        file.write('Inlet Temperature Specified by User-Defined Function \n')
    elif T_inlet_method == 2:
        file.write('Inlet Temperature Specified by CSV Table Input \n')
    if time_integrator == 1:
        file.write('Backward Euler 1st Order Time Integration \n')
    elif time_integrator == 2:
        file.write('Crank Nicholson (i.e. Trapezoidal Rule) 2nd Order Time Integration \n')
    file.write('Column Diameter or Side Length: %.3f m \nColumn Height: %.3f m \nParticle Diameter: %.3f m \nVoid Fraction: %.3f \n' % (D, L, Dp, eps))
    file.write('Flow rate: %.4f m^3/s or %.1f cfm \nSuperficial Velocity: %.3f m/s \nReynolds Number (Ave): %.1f \n' % (Q_in, Q_in*2118.88, V_in, Re_mean))
    file.write('CFL Number: %.2f \nAverage Time Step Size: %.2f s \n' % (CFL, CFL*dx_Sc/V_in))
    #file.write('Maximum Solution Time: %.2f hr \n' % (max_time/3600.))
    if time_increments_post_processing_figures == 1:
        file.write('Maximum Solution Time: %.2f hr \n' % (max_time/3600.))
    elif time_increments_post_processing_figures == 2:
        file.write('Maximum Solution Time: %.2f days\n' % (max_time/(3600.*24.)))
    elif time_increments_post_processing_figures == 3:
        file.write('Maximum Solution Time: %.2f yr\n' % (max_time/(3600.*24.*365.)))
    file.write('Ergun EQ Pressure Loss: %.1f Pa \n' % (delta_p))
    file.write('Koekemoer EQ Pressure Loss: %.1f Pa \n' % (delta_p2))
    file.close()
  
    return 

def writeCSV(T_Post, time_hours,n,dirName):
    print('Writing CSV Files...')

    #dirName = r'C:\Users\pgareau\source\repos\NHEET-1D-model\NHEET-1D-model\Solutions\Data'
    #with open('./Solutions/' + fileName+n+'_CSV.csv', mode='w', newline='') as CSV_file:
    with open(dirName + fileName+n+'_CSV.csv', mode='w', newline='') as CSV_file:
        fieldnames = ['time_hours', 'Tf_0*L', 'Tf_0.25*L', 'Tf_0.50*L', 'Tf_0.75*L', 'Tf_1*L', 'Ts_0*L', 'Ts_0.25*L', 'Ts_0.50*L', 'Ts_0.75*L', 'Ts_1*L']
        write_file = csv.DictWriter(CSV_file, fieldnames=fieldnames)
        write_file.writeheader()
        for i in range(0,len(T_Sc)):
            write_file.writerow({'time_hours': '%.6f' %time_hours[i], 'Tf_0*L': '%.2f' %T_Post[i,0], 'Tf_0.25*L': '%.2f' %T_Post[i,1], 'Tf_0.50*L': '%.2f' %T_Post[i,2], 
                                       'Tf_0.75*L': '%.2f' %T_Post[i,3], 'Tf_1*L': '%.2f' %T_Post[i,4], 'Ts_0*L': '%.2f' %T_Post[i,5], 
                                       'Ts_0.25*L': '%.2f' %T_Post[i,6], 'Ts_0.50*L': '%.2f' %T_Post[i,7], 'Ts_0.75*L': '%.2f' %T_Post[i,8], 
                                       'Ts_1*L': '%.2f' %T_Post[i,9]})
               
    #For Full Data Set, Uncomment
    #Full_Data_Set = np.array([[0.]*(len(T_Sc[0]) + 1)]*len(t_Sc))
    #Full_Data_Set[:,0] = t_Sc[:] 
    #for i in range(0, len(t_Sc)):
    #    for j in range(0, len(T_Sc[0])):
    #        Full_Data_Set[i,j+1] = T_Sc[i,j]

    #fullDataSet = open(fileName+'_FullDataSet.csv', mode='w', newline='')

    #with fullDataSet:

    #    writer = csv.writer(fullDataSet)
    #    writer.writerows(Full_Data_Set)

    print('CSV Files Complete')
    return

def plot(x_Post, T_Post, time_hours, plot_lines, plot_colors, dQdt_Post,n,dirName):
    ## plotting
    print('Plotting...')
    
    #rootdir = r'C:\Users\pgareau\Source\repos\NHEET-1D-model\NHEET-1D-model\Figures'
#     workingDir = os.getcwd()
#     if os.name == 'nt':
#         dirName = workingDir + r'\Solutions\it' + n
#     else:
#         dirName = workingDir + r'/Solutions/it' + n
#     if not os.path.exists(dirName):
#         os.makedirs(dirName)
#     # fileName = 'test.png'

#     plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm \nTotal time = %.1f hours, dt_final = %.5f' % (V_in, Q_in*2118.88,time_hours[-1], t_Sc[-1]-t_Sc[-2])
#     # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
#     xlabel= 't [hr]' 
#     ylabel1= 'T [C]'

    if time_increments_post_processing_figures == 1:
        plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f hours, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [hr]' 
        ylabel1= 'T [C]'
    elif time_increments_post_processing_figures == 2:
        plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f days, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [days]' 
        ylabel1= 'T [C]'
    elif time_increments_post_processing_figures == 3:
        plot_title=fileName+'\nTemperature at Various Column Depths \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f years, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
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
        ax1.plot(time_hours,T_Post[:,i],color=plot_colors[i],linestyle=plot_lines[i],label='x=%.2f' % x_Post[i]) 

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)

    plt.savefig(dirName+fileName+n+'_T',bbox_inches = 'tight') #dpi=100

    ####Second Plot - Heating and Cooling rate####
#     plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm \nTotal time = %.1f hours, dt_final = %.5f' % (V_in, Q_in*2118.88,time_hours[-1], t_Sc[-1]-t_Sc[-2])
#     # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
#     xlabel= 't [hr]' 
#     ylabel1= 'dQ/dt [W]'
    
    if time_increments_post_processing_figures == 1:
        plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f hours, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [hr]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 2:
        plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f days, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [days]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 3:
        plot_title=fileName+' \nRock Mass Heating and Cooling Rate \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f years, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [yr]' 
        ylabel1= 'dQ/dt [W]'
    

    fig, ax1 = plt.subplots()
    ax1title=(plot_title)
    plt.title(ax1title)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1) 
    ax1.tick_params(axis='y') 

    ax1.plot(time_hours,dQdt_Post,color=plot_colors[0],linestyle=plot_lines[0],label='dQ/dt_Solids') 
    #ax1.plot(time_hours,dQdt_Post2,color=plot_colors[0],linestyle=plot_lines[0],label='dQ/dt_Solids') 

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)

    plt.savefig(dirName+fileName+n+'_dQdt',bbox_inches = 'tight') #dpi=100

    print('Plots Complete')
    print('Post Processing Complete')
    print('END %s' % n)
    plt.show()
    plt.close(fig)
    return

def Mine_HC_Potential(n,dirName): #save path may be broken
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

    print('Plotting Mine Heating and Cooling Potential...')

    if time_increments_post_processing_figures == 1:
        plot_title=fileName+'\nMine Heating and Cooling Potential \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f hours, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [hr]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 2:
        plot_title=fileName+'\nMine Heating and Cooling Potential \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f days, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
        # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
        xlabel= 't [days]' 
        ylabel1= 'dQ/dt [W]'
    elif time_increments_post_processing_figures == 3:
        plot_title=fileName+'\nMine Heating and Cooling Potential \nVs = %.3f m/s, Q = %.1f cfm, Rep = %.1f \nErgun EQ Pressure Loss = %.2f Pa or %.2f in. w.g. \nTotal time = %.1f years, dt_final = %.2f s' % (V_in, Q_in*2118.88, Re_mean, delta_p, delta_p*0.00401865, time_Post[-1], dt_Sc)
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

#     workingDir = os.getcwd()
#     if os.name == 'nt':
#         dirName = workingDir + r'\Solutions\Figures'
#     else:
#         dirName = workingDir + r'/Solutions/Figures'

    plt.savefig(dirName+'/'+fileName+n+'_MHCP',bbox_inches = 'tight') #dpi=100

    plt.show()
    plt.close(fig)
    return

def inputs_const(inlet_bc,Dcont,Lin,const_input_vals,rxn_consts):
    #Inputs not considered in parametric study
    global fileName,pamb,T_inlet_method,weather_df,N,\
             first_time_step_size,max_time,Final_hour,table_intervals,CFL,time_integrator,\
            Enhanced_Schumann, time_increments_post_processing_figures,\
            Nu_method,rhos,cps,ks

    ##########################
    #####     INPUTS     #####
    ##########################
    MW,Rgas,p_frc,z_st,p_mol,ptot,rxC=rxn_consts
    ### Output File Name ###
    #fileName = 'Lab-Design-Test'
    fileName = const_input_vals[0]

    inlet_bc_type = const_input_vals[1]
    
    ### Ambient Pressure ###
    pamb = 101325. #[Pa]

    ### Initial Rock Temperature ###
    #T0 = 283.0 #[K]   
    T0=const_input_vals[3]
    dTampin = const_input_vals[4]
    dlamdain = const_input_vals[5]
    
    ### Inlet Temperature ###
    #Choose Between Two Methods: 1 = User-Defined Function, 2 = Input CSV Table
    #T_inlet_method = 1
    T_inlet_method = const_input_vals[2]
    
    #method 1=, 2=Allen2015
    #Nu_method=1
    Nu_method = const_input_vals[6]
    
    #Constants: a = amplitude, b= average value, ps= phase shift, lamda = wavelength (cycle duration) #time is in seconds
    a = dTampin #[K]
    b = T0 #[K]
    lamda = 3600.0*dlamdain #[s]
    Tprof_consts=[a,b,T0,lamda]
    #ps = np.arcsin((T0-b)/a) #[rad]
    ### Method 2 (Input CSV Table) ###
    #Temperature from table - #input Table must have a header, and its first and second column must be time [hr] and Temperature [C]
    weather_df = np.genfromtxt('Input_Hourly_Weather_Data_2010-2019.csv', delimiter=",", dtype = float, skip_header = 1)

    dt_tmp = lamda/const_input_vals[8]
    
    area=Dcont**2 #np.pi/4.0*Dcont**2 #m^2
    if inlet_bc_type==0:
        V_in = inlet_bc
        #Q_in = V_in*area
        #W0 = V_in*pamb/Rgas/Tinf0*area  #mass flow, kg/s
        #inlet_bc_label='V_in'
    elif inlet_bc_type==1:
        #Flow rate calculations at Inlet
        Q_in = inlet_bc/2118.88
        V_in = Q_in/area
        #W0 = V_in*pamb/Rgas/Tinf0*area  #mass flow, kg/s
        #inlet_bc_label='Qcfm'
    elif inlet_bc_type==2:
        W0 = inlet_bc
        V_in = W0/(pamb/Rgas/T0)/area
        #Q_in = V_in*area
        #inlet_bc_label='W0'
    
    ### Courant Friedrichs Lewy Number
    CFL=const_input_vals[7]
    dx_tmp = V_in*dt_tmp/CFL
    N = max(int(np.ceil(Lin/dx_tmp)+1),20)
    print("Number of spaces, N",N)
    ### Spatial Discretization: N = # of points ###
    #N = 200+1

    ### Transient Solution Parameters ###
    #Temporal Discretization 
    #first_time_step_size = 10. #3600.*0.25
    first_time_step_size=const_input_vals[9]

    if T_inlet_method == 1:
        Final_hour = 0
        table_intervals = 0
        ### Maximum Time - Method 1 (Profile) ###
        max_time = 3600.*1000.     
    if T_inlet_method == 2:
        ### Maximum Time - Method 2 (Input CSV Table) ###
        Final_hour = weather_df[-2,0] #last data point is clipped to ensure interpolation does not fail
        table_intervals = weather_df[1,0] - weather_df[0,0] #intervals must be evenly spaced
        max_time = Final_hour*3600.0 / 40.*1 /50. #seconds
    
    ### Include Thermal Conductivity (i.e. Enhanced Schumann) ###
    # 0 = Do not include (Schumann Model)
    # 1 = Include (Enhanced Schumann Model) - Cannot use Crank Nicholson Time Integrator right now. Note: Enhanced Schumann conductivity is not the same as Fluent conductivity, which includes void fraction.
    #Enhanced_Schumann = 0
    Enhanced_Schumann =const_input_vals[10]
    
    ### Time Integrator ###
    ## Choose Between Two Methods: 
        # 1 = Backward Euler - First Order
        # 2 = Crank Nicholson (i.e. Trapezoidal Rule) - Second Order
    #time_integrator = 2
    time_integrator=const_input_vals[11]

    ### Time units in post-processed figures ###
    ## Choose Between Two Units: 
    # 1 = Hours
    # 2 = Days
    # 3 = Years
    time_increments_post_processing_figures = 1
    
    #Solid Properties (Constant)
    ### Granite
    #rhos = 2635. #kg/m^3
    #cps = 790. #J/kg-K
    #ks = 2.6 #W/m-K
    
    #rhos = 2700. #kg/m^3
    #cps = 900. #J/kg-K
    #ks = 3. #W/m-K
    rhos,cps,ks=const_input_vals[12:]
    
    ### Abdel-Ghaffar, E. A.-M., 1980. PhD Thesis
    #rhos = 2418.79 #kg/m^3
    #cps = 908.54 #J/kg-K
    #ks = 1.2626 #W/m-K
    
    print('Inputs defined\n')
    
    return(Tprof_consts)

def inputs_var(factors):
    #Inputs considered in parametric study
    global inlet_bc,D,L,area,Dp,delDp,Dpcorr,eps,phi
    ### Volumetric Flow Rate (constant) ###
    #Qcfm_in = 300. #cfm Qcfm_in = 2850 #cfm
    inlet_bc = factors[0] #V,Qcfm,or W0

    #### Geometry ###
    #Rock Mass
    #D = 0.4572 #m
    D = factors[1]
    #L = D*5.0 #m
    L = D*factors[2]
    
    #constainer shape
    #area = np.pi/4.0*D**2 #m^2 ### cylinder
    area = D**2 #m^2            ### square
    
    #Particle Diameter
    #Dp = 0.01
    Dp=factors[3] #spherical equivalent diameter
    delDp=factors[4]
    #
    ### Geometry ### #Abdel-Ghaffar, E. A.-M., 1980. PhD Thesis
    ##Rock Mass
    #D = 0.3048 #m
    #L= 1.2192 #m
    #area = D**2 #np.pi/4.0*D**2 #m^2
    ##Particle Diameter
    #Dp = 0.028

    #Vsuperficial = Qcfm_in/2118.88/area
    #Re_mean = 1.125*Vsuperficial*Dp/1.81E-5 #m/s

    #Insulation Thickness # Not using for now
    #ithk = 0.1 #m
    #Void fraction
    #eps = 0.4537 #Abdel-Ghaffar, E. A.-M., 1980. PhD Thesis
    eps = factors[7]
    
    #sphericity
    phi = factors[5]
    
    #diameter corrector, ie if want to use d32 in pressure loss
    Dpcorr = factors[6]
 
    return()

def iterate(n,inlet_bc_type,const_input_vals,rxn_consts,Tprof_consts):
    global Q_in,V_in,inc,dx_Sc,t_Sc,T_Sc,delta_p,delta_p2,Re_mean,\
        dt_Sc, Nup_mean,hp_mean,hp_effective_mean,Bi_mean
    ##############################################
    #####     Setup and Initialization     #######
    ##############################################
    print('Setting Up and Initializing . . .')
    MW,Rgas,p_frc,z_st,p_mol,ptot,rxC=rxn_consts

    Tinf0 = T_inlet(0.,Tprof_consts)
    if inlet_bc_type==0:
        V_in = inlet_bc
        Q_in = V_in*area
        W0 = V_in*pamb/Rgas/Tinf0*area  #mass flow, kg/s
        inlet_bc_label='V_in'
    elif inlet_bc_type==1:
        #Flow rate calculations at Inlet
        Q_in = inlet_bc/2118.88
        V_in = Q_in/area
        W0 = V_in*pamb/Rgas/Tinf0*area  #mass flow, kg/s
        inlet_bc_label='Qcfm'
    elif inlet_bc_type==2:
        W0 = inlet_bc
        V_in = W0/(pamb/Rgas/Tinf0)/area
        #V_in = W0/(pamb/Rgas/300.)/area
        Q_in = V_in*area
        inlet_bc_label='W0'
        
    if T_inlet_method == 1:
        T_mean = Tinf0
        #T_mean = Tprof_consts[0]+Tprof_consts[1]
        print(T_mean)
    elif T_inlet_method == 2:
        T_max = np.amax(weather_df[:,1]) + 273.15 #K
        T_min = np.amin(weather_df[:,1]) + 273.15 #K
        T_mean = 0.5*(T_max + T_min)
    else:
        print('Cannot calculate pressure loss... Invalid inlet temperature specification')

    rhof_mean = pamb/Rgas/T_mean #kg/m^3
    #cpf_mean = specheat(T_mean, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_mean = viscosity(T_mean, p_frc, rxC) #kg/m-s
    #kf_mean = conductivity(T_mean, p_frc[1], rxC) #W/m-K    #Unused
    #G_mean = rhof_mean*V_in
    Re_mean = rhof_mean*V_in*Dp/muf_mean
    Nup_mean = ((1.18*Re_mean**0.58)**4 + (0.23*(Re_mean/(1 - eps))**0.75 )**4)**0.25
    hp_mean = Nup_mean*0.02436/Dp
    hp_effective_mean = 1/(Dp/(Nup_mean*0.02436) + Dp/(10*ks)) #Effective heat transfer coefficient accounting for Biot Number Effects
    Bi_mean = hp_mean*Dp/ks
    #print(L,V_in,rhof_mean,muf_mean)
    #Ergun EQ delta_p
    delta_p = ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,L,V_in,rhof_mean,muf_mean)
    delta_p2= Koekemoer_PressureLoss(Dp,Dpcorr,phi,eps,L,V_in,rhof_mean,muf_mean)

    #Fluid Properties Initialization
    p_Sc = np.array([pamb]*N) #static pressure
    Pt = np.array([pamb]*N) #total pressure
    ps = np.array([pamb]*N) #static pressure
    delta_ps= np.array([0.]*N)
    delta_ps_tmp= np.array([0.]*N)
    V = np.array([V_in]*N)
    Vtmp = np.array([V_in]*N)
    
    gam=specheatk(Tinf0, p_frc, rxC)
    
    xM = np.array([V_in/np.sqrt(gam*Rgas*Tinf0)]*N) #mach number
    Pt = pamb*ptps(xM,gam)
    
    rhof = np.array([0.]*(N))
    cpf = np.array([0.]*(N))
    muf = np.array([0.]*(N))
    kf = np.array([0.]*(N))

    #Initialize a time array 
    t_Sc=np.array([0.])

    #Mesh points in space
    x_Sc = np.linspace(0.,L,num=N) #Linear space spanning 0 to L with N equally spaced points
    dx_Sc = x_Sc[1]-x_Sc[0] #space between points 

    ########## Matrix and Array Initialization ##########

    T_Sc = np.array([[Tinf0]*2*N]) #initialization of T0 everywhere, 2x since array of fluid and solid temperatures
    T_Sc[0,0] = Tinf0 #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
    Tnew = np.array([[Tinf0]*1]*2*N) #delta is a temporary array of T at time t-1 - initialization
    Tnew[0,0] = Tinf0
    
    for i in range(0,N):
        rhof[i] = pamb/Rgas/Tnew.T[0][i] #kg/m^3
        cpf[i] = specheat(Tnew.T[0][i], p_frc, rxC)*1000./MW #J/kg-K
        muf[i] = viscosity(Tnew.T[0][i], p_frc, rxC) #kg/m-s
        kf[i] = conductivity(Tnew.T[0][i], p_frc, rxC) #W/m-K    #Unused
        Vtmp[i]=W0/area/rhof[i]
        #delta_ps_tmp[i]=Koekemoer_PressureLoss(Dp,Dpcorr,phi,eps,L-x_Sc[i],Vtmp[0],rhof[0],muf[0])
        delta_ps_tmp[i]=ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,L,Vtmp[0],rhof[0],muf[0])-ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,x_Sc[i],Vtmp[0],rhof[0],muf[0])
        
    fluidprops=np.array([rhof,cpf,muf,kf])
    

    print('Setup and Initialization Complete\n')

    print('Setup Description:')
    print('Filename: ' +fileName) 
#     if T_inlet_method == 1:
#         print('Inlet Temperature Specified by User-Defined Function')
#     elif T_inlet_method == 2:
#         print('Inlet Temperature Specified by CSV Table Input')
#     else:
#         print('Error - Invalid Inlet Temperature Method Selection')
#     if time_integrator == 1:
#         print('Backward Euler 1st Order Time Integration')
#     elif time_integrator == 2:
#         print('Crank Nicholson (i.e. Trapezoidal Rule) 2nd Order Time Integration')
#     else:
#         print('Error - Invalid Time Integration Method Selection')
#     print('Column Diameter or Side Length: %.3f m \nColumn Height: %.3f m \nParticle Diameter: %.3f m \nVoid Fraction: %.3f' % (D, L, Dp, eps))
#     print('Flow rate: %.4f m^3/s or %.1f cfm \nSuperficial Velocity: %.3f m/s \nReynolds Number (Ave): %.1f' % (Q_in, Q_in*2118.88, V_in, Re_mean))
    print('CFL Number: %.2f \nAverage Time Step Size: %.2f s' % (CFL, CFL*dx_Sc/V_in))
    print('Maximum Solution Time: %.2f hr' % (max_time/3600.))
    print('Ergun EQ Pressure Loss: %.1f Pa or %.3f in. w.g.' % (delta_p, delta_p/248.84))
    print('Koekemoer EQ Pressure Loss: %.1f Pa or %.3f in. w.g.\n' % (delta_p2, delta_p2/248.84))
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
    Told = T_Sc[0]
    errT=1.
    inc = 0
    num_runtime_updates = 0

    ### initializers for where to find local min and max
    cell_ID=(np.concatenate(([0],(np.ceil(np.array(range(1,4))*N/4)),[N-1]))).astype(int)

    Nextrema=50
    Iminmax = np.array([[0]*len(cell_ID)]*Nextrema)
    tminmax = np.array([[0.]*len(cell_ID)]*Nextrema)
    Tminmax = np.array([[0.]*len(cell_ID)]*Nextrema)
    MM=1
    QQ=0 #switch to end while loop for a T(t) function input
    ###
    
    while current_time < max_time:

        ##### At Inlet #####
        Tinf_old = Tinf_new
        Tinf_new = T_inlet(new_time,Tprof_consts)
        
        for predcorr in range(0,2): #alternatively while max(abs(err))>1.e-2:      
#             ##### Get Matrix AA and Vector RHS ##### 
            if Enhanced_Schumann == 1:
                if time_integrator == 1:
                    AA_Sc,RHS_Sc, G = Enhanced_Schumman_Backward_Euler1(dt_Sc, Tinf_new,Told,p_Sc,Vtmp,fluidprops)
                else:
                    print("Incorrect Time Integrator Selection")
                    break
            elif Enhanced_Schumann == 0:
                if time_integrator == 1:
                    AA_Sc,RHS_Sc, G = Schumman_Backward_Euler1(dt_Sc, Tinf_new,Told,p_Sc,Vtmp,fluidprops)
                elif time_integrator == 2:
                    AA_Sc,RHS_Sc, G = Schumman_Crank_Nicholson(dt_Sc, Tinf_new,Told,p_Sc,Vtmp,fluidprops)
                else:
                    print("Incorrect Time Integrator Selection")
                    break
            else:
                print("Incorrect Model Selection")
                break

            ##### Solve temperature at updated time "new_time" #####
            err = np.linalg.lstsq(AA_Sc,RHS_Sc,rcond=-1)[0] -Tnew #Returns the least-squares solution to a linear matrix equation
            #print(err)
            Tnew = Tnew + err
            #Tnew = np.linalg.lstsq(AA_Sc,RHS_Sc,rcond=-1)[0]
            
            for i in range(0,N):
                rhof[i] = pamb/Rgas/Tnew.T[0][i] #kg/m^3
                cpf[i] = specheat(Tnew.T[0][i], p_frc, rxC)*1000./MW #J/kg-K
                muf[i] = viscosity(Tnew.T[0][i], p_frc, rxC) #kg/m-s
                kf[i] = conductivity(Tnew.T[0][i], p_frc, rxC) #W/m-K    
                Vtmp[i]=W0/area/rhof[i]
                #delta_ps_tmp[i]=Koekemoer_PressureLoss(Dp,Dpcorr,phi,eps,L-x_Sc[i],Vtmp[0],rhof[0],muf[0]) 
                #delta_ps_tmp[i]=ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,L,Vtmp[0],rhof[0],muf[0])-ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,x_Sc[i],Vtmp[0],rhof[0],muf[0])
            for i in range(0,N):
                delta_ps_tmp[i]=ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,L,Vtmp[-1],rhof[-1],muf[-1])-ErgunEQ_PressureLoss(Dp,Dpcorr,phi,eps,x_Sc[i],Vtmp[-1],rhof[-1],muf[-1])
            fluidprops=np.array([rhof,cpf,muf,kf])
#       ### END PREDICTOR-CORRECTOR LOOP
            
        errT = np.append(errT, max(abs(err)))
        T_Sc=np.vstack([T_Sc,Tnew.T]) #Stack transpose of delta array (now horizontal) to T(t) solution matrix
        t_Sc = np.append(t_Sc, new_time) #Append time values to the end of an array
        Told=Tnew.T[0]
        V = np.vstack([V,Vtmp])
        xM = np.vstack([xM,Vtmp/np.sqrt(gam*Rgas*T_Sc[-1,:N])])
        Pt = np.vstack([Pt,pamb*ptps(xM[-1,-1],gam)+delta_ps_tmp])
        ps = np.vstack([ps,Pt[-1,:]/ptps(xM[-1,:],gam)])
        delta_ps = np.vstack([delta_ps,delta_ps_tmp])
        #print(ps[0],delta_ps[0])
        ###
        
        dt_Sc = NextTimeStep(CFL, G, t_Sc, dx_Sc, T_Sc, dt_Sc,rxn_consts)
        current_time = t_Sc[-1]
        new_time = current_time + dt_Sc
        
        if T_inlet_method == 1:
            ### find local minimums and maximums to determine convergence, phase, phase shift
            Iminmax,tminmax,Tminmax,MM,QQ = findminmax(t_Sc,T_Sc,inc,MM,QQ,cell_ID,Iminmax,tminmax,Tminmax)
            if inc > 10000 or QQ==1: 
                break
        elif T_inlet_method == 2:
            num_runtime_updates = runtime_Update(current_time, max_time, num_runtime_updates) #good if while loop goes to max_time        
        inc+=1
            
    print('Solving Complete. \nFinal solution time: %.2f hr' % (current_time/3600.))
    toc() #End timekeeping 
    #print(errT)
    #####################################
    #####     Post-Processing     #######
    #####################################

    print('Post Processing . . .')
    workingDir = os.getcwd()
    if os.name == 'nt':
        dirItName = workingDir + r'\Solutions\it' + n + '/'
    else:
        dirItName = workingDir + r'/Solutions/it' + n + '/'
    if not os.path.exists(dirItName):
        os.makedirs(dirItName)
    # fileName = 'test.png'
    
    post_process(n,dirItName)
    #Mine_HC_Potential(n,dirItName) #Mine heating and cooling potential
    geom_vals=[Dp,L,area,eps,W0,N]
    Matrix = [x_Sc,t_Sc,T_Sc,delta_ps,ps,Pt,xM,V]

    return pamb,dirItName,x_Sc,T_Sc,t_Sc,Iminmax,tminmax,Tminmax,geom_vals,Matrix,current_time,delta_p,delta_p2

    ####### END #######