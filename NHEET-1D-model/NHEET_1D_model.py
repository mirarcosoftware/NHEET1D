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
#from scipy.interpolate import griddata
print('libraries loaded')
#import time
from tictoc import tic, toc

#A fluid properties solver intended for combustion applications is used here to obtain relationships for SI properties of dry air as a function of temperature.
#Inputs below are selected so that properties of solely dry air are provided.
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

def getB_Scf(cpf):

    B_Scf = G_0*cpf
    
    return B_Scf

def getC_Scf(rhof,muf):
    
    #Heat Transfer Correlations
    #Rep = G*Dp/muf
    #Nup=((1.18*Rep**0.58)**4 + (0.23*(1/(1-eps)*Rep)**0.75)**4)**0.25
    #Verify which hp
    #hp=1/(Dp/(Nup*kf_in) + Dp/(10*ks)) -> more correct kf_in value, but using original value from Ansys Fluent for now
    #hp=1/(Dp/(Nup*rhof) + Dp/(10*rhos))
    hp= 17.773 #10.89797907 #Assumed constant for now
    Ap= 6*(1-eps)/(Dp)
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

def getdm_dt(rhof_in):

    dm_dt = rhof_in*area*V_in
    
    return dm_dt

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

def post_process():
    
    time_hours = t_Sc/3600.
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

    #For plotting
    #Available lines: 'solid', 'dashed', 'dashdot', 'dotted'
    plot_lines = ['solid','solid','solid','solid','solid','dashdot','dashdot','dashdot','dashdot','dashdot'] 
    #Available colours: blue ('b'), green ('g'), red ('r'), cyan ('c', magenta ('m'), yellow ('y'), black ('k'), white ('w')
    plot_colors = ['r','b','g','y','c','r','b','g','y','c']
    #Available markers: 'o','d','s','*','<','>','^','v','1','2','3','4']
    #plot_markers = []

    writeCSV(T_Post, time_hours)
    plot(x_Post, T_Post, time_hours, plot_lines, plot_colors)

def writeCSV(T_Post, time_hours):
    print('Writing CSV Files...')

    #dirName = r'C:\Users\pgareau\source\repos\NHEET-1D-model\NHEET-1D-model\Solutions\Data'
    with open(fileName+'.csv', mode='w', newline='') as CSV_file:
        fieldnames = ['time_hours', 'Tf_0*L', 'Tf_0.25*L', 'Tf_0.50*L', 'Tf_0.75*L', 'Tf_1*L', 'Ts_0*L', 'Ts_0.25*L', 'Ts_0.50*L', 'Ts_0.75*L', 'Ts_1*L']
        write_file = csv.DictWriter(CSV_file, fieldnames=fieldnames)
        write_file.writeheader()
        for i in range(0,len(T_Sc)):
            write_file.writerow({'time_hours': '%.6f' %time_hours[i], 'Tf_0*L': '%.2f' %T_Post[i,0], 'Tf_0.25*L': '%.2f' %T_Post[i,1], 'Tf_0.50*L': '%.2f' %T_Post[i,2], 
                                       'Tf_0.75*L': '%.2f' %T_Post[i,3], 'Tf_1*L': '%.2f' %T_Post[i,4], 'Ts_0*L': '%.2f' %T_Post[i,5], 
                                       'Ts_0.25*L': '%.2f' %T_Post[i,6], 'Ts_0.50*L': '%.2f' %T_Post[i,7], 'Ts_0.75*L': '%.2f' %T_Post[i,8], 
                                       'Ts_1*L': '%.2f' %T_Post[i,9]})

    print('CSV Files Complete')
    return

def plot(x_Post, T_Post, time_hours, plot_lines, plot_colors):
    ## plotting
    print('Plotting...')
    
    plot_title=fileName+'\nVs = %.3f m/s, Q = %.1f cfm \nTotal time = %.1f hours' % (V_in, Qcfm_in,time_hours[-1])
    # plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
    xlabel= 'time, hr' 
    ylabel1= 'T, C'

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

    #rootdir = r'C:\Users\pgareau\Source\repos\NHEET-1D-model\NHEET-1D-model\Figures'
    workingDir = os.getcwd()
    if os.name == 'nt':
        dirName = workingDir + r'\Solutions\Figures'
    else:
        dirName = workingDir + r'/Solutions/Figures'
    # fileName = 'test.png'
    plt.savefig(dirName+'/'+fileName,bbox_inches = 'tight') #dpi=100

    print('Plots Complete')
    print('Post Processing Complete')
    print('END')
    plt.show()
    plt.close(fig)
    return

##########################
#####     INPUTS     #####
##########################

#Ambient Pressure
pamb = 101325.

#Initial Rock Temperature
T0 = 275.65 #[K]

#Inlet Temperature
#Choose A Method: 1 = Defined Profile, 2 = Input CSV Table
T_inlet_method = 2
#Method 1 (Profile)
#Constants: a = amplitude, b= average value, ps= phase shift, lamda = wavelength (cycle duration) #time is in seconds
a = 42.5 #[K]
b = 275.65 #[K]
lamda = 3600.0*0.5 #[s]
ps = np.arcsin((T0-b)/a) #[rad]
#Method 2 (Input CSV Table)
#Temperature from table - #input Table must have a header, and its first and second column must be time [hr] and Temperature [C]
weather_df = np.genfromtxt('Test_TempData.csv', delimiter=",", dtype = float, skip_header = 1)

#Geometry 
#Rock Mass
D=0.3048 #m
L= 1.2192 #1.0*D #m
area = D**2 #np.pi/4.0*D**2 #m^2
#Particle Diameter
Dp = 0.028 #m
#Insulation Thickness # Not using for now
#ithk = 0.1 #m
#Porosity
eps = 0.4537

#Volumetric Flow Rate (constant)
Qcfm_in = 27.13 #cfm

#Spatial Discretization: N = # of points
N = 400+1

#Transient Solution Parameters
dt_Sc = 300. #time step size in seconds
max_time = 1.0*3600. # set a max time constraint for time marching if T_inlet Profile (Method 0) is used
if T_inlet_method == 1:
    Final_hour = weather_df[-2,0] #last data point is clipped to ensure interpolation does not fail
    table_intervals = weather_df[1,0] - weather_df[0,0]
    max_time = Final_hour*3600.0

#Output File Name
fileName = 'Cleanup-Schumann-Validation_Abdel-Ghaffar-1980-PhD'

##############################################
#####     Setup and Initialization     #######
##############################################

print('Setting Up and Initializing')

MW,Rgas,p_frc,z_st,p_mol,ptot,rxC = reaction_balance()
print('Fluid property relationships defined')

#Flow rate calculations at Inlet
Q_in = Qcfm_in/2118.88
V_in = Q_in/area

#Fluid Properties 
#At Inlet at t=0
weather_inc = 0
#Tinf0 = Tprof(a,b,ps,lamda,0.0)
#Tinf0 = weather_df[0, 1] + 273.15 #[K]
Tinf0 = T_inlet(0.0)
rhof_in = pamb/Rgas[1]/Tinf0 #kg/m^3
cpf_in=specheat(Tinf0, p_frc[1], rxC)*1000./MW[1] #J/kg-K
muf_in=viscosity(Tinf0, p_frc[1], rxC) #kg/m-s
kf_in=conductivity(Tinf0, p_frc[1], rxC) #W/m-K
G_0=rhof_in*V_in
mdot_0 = rhof_in*area*V_in
#Elsewhere at t=0
rhof_0 = pamb/Rgas[1]/T0 #kg/m^3
cpf_0=specheat(T0, p_frc[1], rxC)*1000./MW[1] #J/kg-K
muf_0=viscosity(T0, p_frc[1], rxC) #kg/m-s
kf_0=conductivity(T0, p_frc[1], rxC) #W/m-K

#Solid Properties (Constant)
rhos = 2418.79 #2635. #kg/m^3
cps = 908.54 #790. #J/kg-K
ks = 1.2626 #2.6 #W/m-K

#Particle Reynolds and Prandtl at Inlet
Rep_in = G_0*Dp/muf_in
Pr_in = muf_in*cpf_in/kf_in 

#Schumanns Fluid property initialization at temperature T0 and left end fluid cell at Tinf
rhof_Sc = np.array([rhof_0]*N) #kg/m^3
cpf_Sc = np.array([cpf_0]*N) #J/kg-K
muf_Sc = np.array([muf_0]*N) #kg/m-s
kf_Sc = np.array([kf_0]*N) #W/m-K
rhof_Sc[0] = rhof_in #kg/m^3
cpf_Sc[0] = cpf_in #J/kg-K
muf_Sc[0] = muf_in #kg/m-s
kf_Sc[0] = kf_in #W/m-K

#Pressure gradient initialization - Assumed constant everywhere for now
p_Sc = np.array([pamb]*N)

#Initialize a time array 
t_Sc=np.array([0.])

#Mesh points in space
x_Sc = np.linspace(0.,L,num=N) #Linear space spanning 0 to L with N equally spaced points
dx_Sc = x_Sc[1]-x_Sc[0] #space between points 

########## Matrix and Array Initialization ##########

#Note: Twice the size for each matrix and array due to coupling of Ts and Tf (Fluid and Solid combined)
AA_Sc = np.array([[0.]*2*N]*2*N)
RHS_Sc =np.array([[0.]*1]*2*N)

T_Sc = np.array([[T0]*2*N]) #initialization of T0 everywhere
T_Sc[0,0] = Tinf0 #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
delta_Sc = np.array([[T0]*1]*2*N) #delta is a temporary array of T at time t-1 - initialization
delta_Sc[0,0] = Tinf0

#Arrays
#Fluid
A_Scf = np.array([0.0]*N) #dTf/dt
B_Scf = np.array([0.0]*N) #dTf/dx
C_Scf = np.array([0.0]*N) #Ts, Tf
# D_Scf = np.array([0.0]*N) #Ts, Tf
##Solid
A_Scs = np.array([getA_Scs(rhos,cps)]*N) #dTs/dt #Constant
B_Scs = np.array([0.0]*N) #Ts, Tf 

#Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
AA_Sc[0,0]=1.

print('Setup and Initialization Complete')

######################################
#####     Transient Solver     #######
######################################

print('Solving')
Tinf_old = Tinf0
Tinf_new = Tinf0

tic() #Start timekeeping (tictoc) to determine how long computation takes
current_time = t_Sc[-1]
new_time = current_time + dt_Sc
inc = 0

while current_time < max_time:
             
    #At Inlet
    Tinf_old = Tinf_new
    #Tinf_new = Tprof(a,b,ps,lamda,dt_Sc*(inc+1))
    Tinf_new = T_inlet(new_time)
    
    #Update Fluid BC at Left End
    RHS_Sc[0,0]=Tinf_new

    rhof_Sc[0] = pamb/Rgas[1]/Tinf_old #kg/m^3
    cpf_Sc[0] = specheat(Tinf_old, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_Sc[0] = viscosity(Tinf_old, p_frc[1], rxC) #kg/m-s
    kf_Sc[0] = conductivity(Tinf_old, p_frc[1], rxC) #W/m-K    #Unused
    dm_dt = getdm_dt(rhof_Sc[0])
    G = dm_dt/area
    
    C_Scf[0]=getC_Scf(rhof_Sc[0],muf_Sc[0])
    B_Scs[0]=getB_Scs(C_Scf[0])
    
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs[0]/dt_Sc + B_Scs[0]
    AA_Sc[N,0] = -B_Scs[0]
    RHS_Sc[N] = delta_Sc[N]*(A_Scs[0]/dt_Sc)

    for j in range(1,N): #Begin for loop
        
        #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)
        T_cell_Scf=T_Sc[inc][j] 
        rhof_Sc[j]=p_Sc[j]/Rgas[1]/T_cell_Scf #kg/m^3
        cpf_Sc[j]=specheat(T_cell_Scf, p_frc[1], rxC)*1000./MW[1] #J/kg-K
        muf_Sc[j]=viscosity(T_cell_Scf, p_frc[1], rxC) #kg/m-s
        kf_Sc[j]=conductivity(T_cell_Scf, p_frc[1], rxC) #W/m-K    #Unused
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf[j]=getA_Scf(rhof_Sc[j],cpf_Sc[j])
        B_Scf[j]=getB_Scf(cpf_Sc[j])
        C_Scf[j]=getC_Scf(rhof_Sc[j],muf_Sc[j])
        B_Scs[j]=getB_Scs(C_Scf[j])
        
        #Matrix Coefficients and RHS
        if j < N-1: 
            #Second Order Accurate Central Differencing Scheme
            #Fluid  
            AA_Sc[j,j-1]=-B_Scf[j]/(2.*dx_Sc) 
            AA_Sc[j,j]=A_Scf[j]/dt_Sc +C_Scf[j]
            AA_Sc[j,j+1]=B_Scf[j]/(2.*dx_Sc)
            AA_Sc[j,j+N] = -C_Scf[j]
            RHS_Sc[j]=delta_Sc[j]*(A_Scf[j]/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N]=A_Scs[j]/dt_Sc +B_Scs[j]
            AA_Sc[j+N,j] = -B_Scs[j]
            RHS_Sc[j+N]=delta_Sc[j+N]*(A_Scs[j]/dt_Sc)
        else:
            #First Order Accurate Backward Differencing Scheme for Cells at Right End
            #Fluid 
            AA_Sc[j,j-1] = -B_Scf[j]/dx_Sc
            AA_Sc[j,j] = A_Scf[j]/dt_Sc +B_Scf[j]/dx_Sc +C_Scf[j]
            AA_Sc[j,j+N] = -C_Scf[j]
            RHS_Sc[j] = delta_Sc[j]*(A_Scf[j]/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N] = A_Scs[j]/dt_Sc +B_Scs[j]
            AA_Sc[j+N,j] = -B_Scs[j]
            RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs[j]/dt_Sc) #End for loop
        
    delta_Sc=np.linalg.lstsq(AA_Sc,RHS_Sc,rcond=-1)[0] #Returns the least-squares solution to a linear matrix equation

    T_Sc=np.vstack([T_Sc,delta_Sc.T]) #Stack transpose of delta array (now horizontal) to T(t) solution matrix
    t_Sc = np.append(t_Sc,dt_Sc*(inc+1)) #Append time values to the end of an array
    current_time = t_Sc[-1]
    new_time = current_time + dt_Sc
    inc+=1
    

print('Solving Complete. Final solution time: %.2f hr' % (current_time/3600.))
toc() #End timekeeping 

#####################################
#####     Post-Processing     #######
#####################################

print('Post Processing')
post_process()

####### END #######