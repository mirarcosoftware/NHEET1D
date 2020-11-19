#
# Schumanns Model
#
###########################
## Libraries             ##
###########################
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from fluid_properties12 import chem_rxn,MolWt,specheat,specheatk, \
    enthalpy, enthalpy_Tnew, enthalpy_T, entropy,viscosity,conductivity
#isentropic relationships
from isentropic_formulas import density, mach, max_mass,qdyn, correctedflow, \
    ambientPT, ttts, ptps, mach_ptps, mach_ttts

#other
import sys
from termcolor import colored, cprint
from scipy.interpolate import griddata
#import time
from tictoc import tic, toc
#import inspect
import os
#import csv

R_u = 8.31451
print('libraries loaded')

###########################
## Fluid properties SI   ##
###########################
#A fluid properties solver intended for combustion applications is used here to obtain properties of dry air as a function of temperature.
#Inputs below are selected so that properties of purely dry air are provided.

#number moles of species for air to give 21% O2 and 79% N2 (by mol)
rxC =0; rxH =0; rxO =0
#jet fuel ~kerosene, use C12H24 https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20000065620.pdf
rfC=12; rfH=24; rfO=0

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

print('fluid property relationships defined')
print('Setting Up and Initializing')
#Inputs

#Ambient Pressure and Temperature at Inlet and t=0
pamb = 101325.
Tinf = 275.65
T0 = 275.65

#Function for temperature inlet T(t)
#Constants: a = amplitude, b= average value, ps= phase shift, lamda = wavelength (cycle duration) #time is in seconds
a = 42.5 #[K]
b = 275.65 #[K]
lamda = 3600.0*0.5 #[s]
ps = np.arcsin((T0-b)/a) #[rad]
def Tprof(a,b,ps,lamda,t):
    return a*np.sin(2.*np.pi/lamda*t+ps)+b

#Geometry 
#Rock Mass
D=0.3048 #m
L= 1.2192 #1.0*D #m
area = D**2 #np.pi/4.0*D**2 #m^2
#Particle Diameter
Dp = 0.028 #m
#Insulation Thickness # Not using for now
# ithk = 0.1 #m

#Porosity
eps = 0.4537

#Flow 
Qcfm_in = 27.13 #cfm
#Flow rate calculations at Inlet
Q_in = Qcfm_in/2118.88
V_in = Q_in/area

#Spatial Discretization: N = # of points
N = 400+1

#Fluid Properties 
#At Inlet at t=0
Tinf0 = Tprof(a,b,ps,lamda,0.0)
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
Pr_in = muf_in*cpf_in/kf_in #Delete?

#Heat Transfer Correlations #Variable
#Nusselt Number - Note: fluid properties change with T, thus so should Rep, Nup, and hp
Nup=((1.18*Rep_in**0.58)**4 + (0.23*(1/(1-eps)*Rep_in)**0.75)**4)**0.25
#hp=1/(Dp/(Nup*kf_in) + Dp/(10*ks)) -> more correct kf_in value, but using original value from Ansys Fluent for now
hp= 17.773 #10.89797907
Ap= 6*(1-eps)/(Dp)
Cconst_Scf = hp*Ap #Assumed constant for now

### To Clean Up ###
#Alternative Nusselt Number equations below - 
# if Rep_in > 40:
#     Nup = 2.42*Rep_in**(1./3.)*Pr**(1./3.) + 0.129*Rep_in**0.8*Pr**0.4 + 1.4*Rep_in**0.4
# else:
#     Nup = 3.22*Rep_in**(1./3.)*Pr**(1./3.) + 0.117*Rep_in**0.8*Pr**0.4
# hp = Nup*kf_in/Dp

#For Wall Heat Loss - Assumed adiabatic for now
# hw = (kf_in/Dp)*(2.576*Rep_in**(1./3.)*Pr**(1./3.)+0.0936*Rep_in**0.8*Pr**0.4)
# h2 = np.log((0.5*np.sqrt(Ly**2+Lz**2)+ithk)/(0.5*np.sqrt(Ly**2+Lz**2)))/(2*np.pi*L*0.06)
# hx = 1.42*(75./L)**0.25
# Uw = 1./(1./(hw*(2*Ly*L+2*Lz*L)) + h2 + 1./(hx*(2*Ly*L+2*Lz*L)))

#Schumann Model Functions

def getA_Scf(rhof,cpf):

    A_Scf = eps*rhof*cpf
    
    return A_Scf

def getB_Scf(cpf):

    B_Scf = G_0*cpf
    
    return B_Scf

def getC_Scf(rhof,muf):
    
#     Rep = G*Dp/muf
#     Nup=((1.18*Rep**0.58)**4 + (0.23*(1/(1-eps)*Rep)**0.75)**4)**0.25
#     hp=1/(Dp/(Nup*rhof) + Dp/(10*rhos))
    #Ap= 6*(1-eps)/(Dp) #Constant
#     Cvar_Scf = hp*Ap
    C_Scf = Cconst_Scf 
        
    return C_Scf

# def getD_Scf(kf,muf,cpf):  #Might add this later
#     Uw = 1./(1./(hw*(2*Ly*L+2*Lz*L)) + h2 + 1./(hx*(2*Ly*L+2*Lz*L)))
#     D_Scf=-UwAw
        
#     return D_Scf

def getA_Scs(rhos,cps):

    A_Scs = (1-eps)*rhos*cps #Constant
    
    return A_Scs

def getB_Scs(C_Scf):

    B_Scs = C_Scf #They are the same
    
    return B_Scs

def getdm_dt(rhof_in):

    dm_dt = rhof_in*area*V_in
    
    return dm_dt

#Pressure left out for now - Fluid properties are assumed to be a function of temperature only
# def getDeltaP():
#     DeltaP = 1
#     #In progress
#     return deltaP

#Schumanns Fluid property initialization at temperature T0 and left end fluid cell at Tinf
rhof_Sc = np.array([rhof_0]*N) #kg/m^3
cpf_Sc = np.array([cpf_0]*N) #J/kg-K
muf_Sc = np.array([muf_0]*N) #kg/m-s
kf_Sc = np.array([kf_0]*N) #W/m-K
rhof_Sc[0] = rhof_in #kg/m^3
cpf_Sc[0] = cpf_in #J/kg-K
muf_Sc[0] = muf_in #kg/m-s
kf_Sc[0] = kf_in #W/m-K

#Pressure gradient initialization 
#STILL IN PROGRESS - Assumed constant everywhere for now
p_Sc = np.array([pamb]*N)
# alpha_inv=1/(Dp**2/150*eps**3/(1-eps)**2)
# ebd=2*Ly*Lz/(Ly+Lz)
# c_f=0.55*(1-5.5*Dp/ebd)
# R_inertial=2*c_f*alpha_inv**0.5

# delta_ptot = (-muf_0*alpha_inv*V_in -R_inertial/2*rhof_0*V_in**2)*L
# delta_p = delta_ptot*dx/L
# print(delta_p, delta_ptot)

#for j in range(0,N):
#    p_Sc[j] = pamb + delta_ptot + delta_p*j

#Initialize a time array - Must reinitialize with every run
t_Sc=np.array([0.])
dt_Sc = 100. #time step size in seconds
max_time = 1.0*3600. # set a max time constraint for time marching

#Mesh points in space
x_Sc = np.linspace(0.,L,num=N) #Linear space spanning 0 to L with N equally spaced points
dx_Sc = x_Sc[1]-x_Sc[0] #space between points 

########## Matrix and Array Initialization ##########

#Note: Twice the size for each matrix and array due to coupling of Ts and Tf (Fluid and Solid combined)
AA_Sc = np.array([[0.]*2*N]*2*N)
RHS_Sc =np.array([[0.]*1]*2*N)
T_Sc = np.array([T0]*2*N) #initialization of T0 everywhere
T_Sc[0] = Tinf #Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
delta_Sc = np.array([[T0]*1]*2*N) #delta is T at time t-1 - initialization 
delta_Sc[0] = Tinf

#Matrix Coefficient Terms at t=0
#Fluid
A_Scf0 = getA_Scf(rhof_0,cpf_0)
B_Scf0 = getB_Scf(cpf_0)
C_Scf0 = getC_Scf(rhof_0,muf_0)
#D_Scf0 = getD_Scf(kf_0,muf_0,cpf_0)
#Solid
A_Scs0 = getA_Scs(rhos,cps) #Constant
B_Scs0 = getB_Scs(C_Scf0)
#Arrays
A_Scf = np.array([A_Scf0]*N) #dTf/dt
A_Scf[0] = getA_Scf(rhof_Sc[0],cpf_Sc[0]) 
B_Scf = np.array([B_Scf0]*N) #dTf/dx
B_Scf[0] = getB_Scf(cpf_Sc[0]) 
C_Scf = np.array([C_Scf0]*N) #Ts, Tf
C_Scf[0] = getC_Scf(rhof_Sc[0],muf_Sc[0])
# D_Scf = np.array([D_Scf0]*N) #Ts, Tf
# D_Scf[0] = getD_Scf(kf_Sc[0],muf_Sc[0],cpf_Sc[0])
A_Scs = np.array([A_Scs0]*N) #dTs/dt #Constant
B_Scs = np.array([B_Scs0]*N) #Ts, Tf 
B_Scs[0] = getB_Scs(C_Scf[0]) 

#Fill in Coefficient Matrix and RHS
#Fluid BC at LE - Temperature is fixed to inlet fluid temperature - Solid temperature is not fixed
AA_Sc[0,0]=1.
RHS_Sc[0]=Tprof(a,b,ps,lamda,dt_Sc)

for j in range(1,N): #Begin for Loop
    if j < N-1:
        #Fluid #Central Differencing Scheme
        AA_Sc[j,j-1] = -B_Scf[j]/(2.*dx_Sc)
        AA_Sc[j,j] = A_Scf[j]/dt_Sc +C_Scf[j]
        AA_Sc[j,j+1] = B_Scf[j]/(2.*dx_Sc)
        AA_Sc[j,j+N] = -C_Scf[j]
        RHS_Sc[j] = delta_Sc[j]*(A_Scf[j]/dt_Sc)
        #Solid
        AA_Sc[j+N,j+N] = A_Scs[j]/dt_Sc +B_Scs[j]
        AA_Sc[j+N,j] = -B_Scs[j]
        RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs[j]/dt_Sc)
    else:
        #Fluid #Backward Differencing Scheme
        AA_Sc[j,j-1] = -B_Scf[j]/dx_Sc
        AA_Sc[j,j] = A_Scf[j]/dt_Sc +B_Scf[j]/dx_Sc +C_Scf[j]
        AA_Sc[j,j+N] = -C_Scf[j]
        RHS_Sc[j] = delta_Sc[j]*(A_Scf[j]/dt_Sc)
        #Solid
        AA_Sc[j+N,j+N] = A_Scs[j]/dt_Sc +B_Scs[j]
        AA_Sc[j+N,j] = -B_Scs[j]
        RHS_Sc[j+N] = delta_Sc[j+N]*(A_Scs[j]/dt_Sc) #End for loop
        
#Solid at LE
AA_Sc[N,N] = A_Scs[0]/dt_Sc +B_Scs[0] #B_Scs[0] is constant
AA_Sc[N,0] = -B_Scs[0]
RHS_Sc[N] = delta_Sc[N]*(A_Scs[0]/dt_Sc)

print('Setup and Initialization Complete')


########## Time Marching ##########

#Must reinitialize before each run
print('Solving')
## First timestep
delta_Sc=np.linalg.lstsq(AA_Sc,RHS_Sc,rcond=-1) #rcond: cut-off ratio for small singular values of AA (rcond=-1 is for machine precision)
T_Sc=np.vstack([T_Sc,delta_Sc[0].T]) #Reshape 1-D arrays of horizontal shape (N,1) to vertical shape (1,N)
t_Sc = np.append(t_Sc,dt_Sc) #Append initial timestep value to the end of an array of each time instance

tic() #Start timekeeping (tictoc) to determine how long computation takes
inc=1
while True: #(abs(T_Sc[inc,2*N-1]-Tinf) > 0.001): 
    #At Inlet
    Tinf = Tprof(a,b,ps,lamda,dt_Sc*inc)
    rhof_in = pamb/Rgas[1]/Tinf #kg/m^3
    cpf_in=specheat(Tinf, p_frc[1], rxC)*1000./MW[1] #J/kg-K
    muf_in=viscosity(Tinf, p_frc[1], rxC) #kg/m-s
    kf_in=conductivity(Tinf, p_frc[1], rxC) #W/m-K
    T_Sc[inc][0]= Tinf
    rhof_Sc[0] = rhof_in #kg/m^3
    cpf_Sc[0] = cpf_in #J/kg-K
    muf_Sc[0] = muf_in #kg/m-s
    kf_Sc[0] = kf_in #W/m-K
    RHS_Sc[0]=Tinf 
    dm_dt = getdm_dt(rhof_Sc[0])
    G = dm_dt/area
    
    for j in range(1,N): #Begin for loop
        
        #Fluid Properties Note: A function of T only (i.e. incompressible ideal gas assumed)
        T_cell_Scf=T_Sc[inc][j] #To compare to original 1D model, set T_cell=243.15 (not computationally efficient to do this)
        rhof_Sc[j]=p_Sc[j]/Rgas[1]/T_cell_Scf #kg/m^3
        cpf_Sc[j]=specheat(T_cell_Scf, p_frc[1], rxC)*1000./MW[1] #J/kg-K
        muf_Sc[j]=viscosity(T_cell_Scf, p_frc[1], rxC) #kg/m-s
        #kf_Sc[j]=conductivity(T_cell_Scf, p_frc[1], rxC) #W/m-K    #Unused
        
        #Get variables Avar_Scf,Bvar_Scf,Cvar_Scf
        A_Scf[j]=getA_Scf(rhof_Sc[j],cpf_Sc[j])
        B_Scf[j]=getB_Scf(cpf_Sc[j])
        C_Scf[j]=getC_Scf(rhof_Sc[j],muf_Sc[j])
        B_Scs[j]=getB_Scs(C_Scf[j])
        
        #Matrix Coefficients and RHS
        if j < N-1: 
            #Fluid #Central Differencing Scheme 
            AA_Sc[j,j-1]=-B_Scf[j]/(2.*dx_Sc) 
            AA_Sc[j,j]=A_Scf[j]/dt_Sc +C_Scf[j]
            AA_Sc[j,j+1]=B_Scf[j]/(2.*dx_Sc)
            AA_Sc[j,j+N] = -C_Scf[j]
            RHS_Sc[j]=delta_Sc[0][j]*(A_Scf[j]/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N]=A_Scs[j]/dt_Sc +B_Scs[j]
            AA_Sc[j+N,j] = -B_Scs[j]
            RHS_Sc[j+N]=delta_Sc[0][j+N]*(A_Scs[j]/dt_Sc)
        else:
            #Fluid #Backward Differencing Scheme
            AA_Sc[j,j-1] = -B_Scf[j]/dx_Sc
            AA_Sc[j,j] = A_Scf[j]/dt_Sc +B_Scf[j]/dx_Sc +C_Scf[j]
            AA_Sc[j,j+N] = -C_Scf[j]
            RHS_Sc[j] = delta_Sc[0][j]*(A_Scf[j]/dt_Sc)
            #Solid
            AA_Sc[j+N,j+N] = A_Scs[j]/dt_Sc +B_Scs[j]
            AA_Sc[j+N,j] = -B_Scs[j]
            RHS_Sc[j+N] = delta_Sc[0][j+N]*(A_Scs[j]/dt_Sc) #End for loop
        
    #Solid at LE - Fluid Properties are Constant
    AA_Sc[N,N] = A_Scs[0]/dt_Sc +B_Scs[0]
    AA_Sc[N,0] = -B_Scs[0]
    RHS_Sc[N] = delta_Sc[0][N]*(A_Scs[0]/dt_Sc)

    delta_Sc=np.linalg.lstsq(AA_Sc,RHS_Sc,rcond=-1) #Returns the least-squares solution to a linear matrix equation
    T_Sc=np.vstack([T_Sc,delta_Sc[0].T]) #Stack transpose of delta array (now horizontal) to T(t) solution matrix
    t_Sc = np.append(t_Sc,dt_Sc*(inc+1)) #Append time values to the end of an array
    if t_Sc[-1] >= max_time:
        print('Solving Complete. Final solution time: %.2f hr' % (dt_Sc*(inc+1)/3600.)) 
        break
    inc+=1

# toc() #End timekeeping #End while loop
# cprint('time = %g' % (t_Sc[-1]),'green') #Print flow time

########## Post-Processing ##########
print('Post Processing')
#Exporting
fileName = 'Schumann-Validation_Abdel-Ghaffar-1980-PhD'

## plotting

markers = ['o','d','s','*','<','>','^','v','1','2','3','4']
linestyles = ['solid','solid','solid','solid','solid','dashdot','dashdot', \
              'dashdot','dashdot','dashdot','dashdot'] #available linestyles: solid, dashed, dashdot, dotted
colors = ['r','b','g','y','c','r','b','g','y','c','m','r','b','g','y','c','m','r','b','g','y','c','m'] 
#colours available: blue (b), green (g), red (r), cyan (c), magenta (m), yellow (y), black (k), white (w)

xval1 = 0
xval2 = int(np.ceil(N/4))
xval3 = int(np.ceil(N/2))
xval4 = int(np.ceil(3*N/4))
xval5 = -1
xval6 = int(np.ceil(N/10))
xval7 = int(np.ceil(N/20))

xValSc = t_Sc/3600.
yValScf1 = T_Sc[:,xval1] -273.15
yValScf2 = T_Sc[:,xval2] -273.15
yValScf3 = T_Sc[:,xval3] -273.15
yValScf4 = T_Sc[:,xval4] -273.15
yValScf5 = T_Sc[:,xval5] -273.15
yValScf6 = T_Sc[:,xval6] -273.15
yValScf7 = T_Sc[:,xval7] -273.15
yValScs1 = T_Sc[:,N+xval1] -273.15
yValScs2 = T_Sc[:,N+xval2] -273.15
yValScs3 = T_Sc[:,N+xval3] -273.15
yValScs4 = T_Sc[:,N+xval4] -273.15
yValScs5 = T_Sc[:,N+xval5] -273.15
yValScs6 = T_Sc[:,N+xval6] -273.15
yValScs7 = T_Sc[:,N+xval7] -273.15

plot_title=fileName+'\nVs = %.3f m/s, Q = %.1f cfm \nCycle = %.2f hours, Total time = %.1f hours' % (V_in, Qcfm_in,lamda/3600.0,t_Sc[-1]/3600)
# plot_title=fileName+'\nDimensions: L = %.2f m, D = %.2f m, Dp = %.3f m \nQ = %.1f cfm = %.5f m3/s \nCycle = %.2f hours, Total time = %.1f hours \ndt = %.1fs, dx = %.2fm' % (L,D,Dp,Qcfm_in,Q_in,lamda/3600.0,t_SP[-1]/3600, dt_SP, dx_SP)
xlabel= 'time, hr' 
ylabel1= 'T, C'

fig, ax1 = plt.subplots()
ax1title=(plot_title)
plt.title(ax1title)

ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel1) 
ax1.tick_params(axis='y') 

m=0
ax1.plot(xValSc,yValScf1,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval1]) #,marker=markers[m])

m=1
ax1.plot(xValSc,yValScf2,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval2]) #,marker=markers[m])

m=2
ax1.plot(xValSc,yValScf3,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval3]) #,marker=markers[m])

m=3
ax1.plot(xValSc,yValScf4,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval4]) #,marker=markers[m])

m=4
ax1.plot(xValSc,yValScf5,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval5]) #,marker=markers[m])

# m=5
# ax1.plot(xValSc,yValScf6,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval6]) #,marker=markers[m])

# m=6
# ax1.plot(xValSc,yValScf7,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval7]) #,marker=markers[m])

m=5
ax1.plot(xValSc,yValScs1,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval1]) #,marker=markers[m])

m=6
ax1.plot(xValSc,yValScs2,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval2]) #,marker=markers[m])

m=7
ax1.plot(xValSc,yValScs3,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval3]) #,marker=markers[m])

m=8
ax1.plot(xValSc,yValScs4,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval4]) #,marker=markers[m])

m=9
ax1.plot(xValSc,yValScs5,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval5]) #,marker=markers[m])

# m=12
# ax1.plot(xValSc,yValScs6,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval6]) #,marker=markers[m])

# m=13
# ax1.plot(xValSc,yValScs7,color=colors[m],linestyle=linestyles[m],label='x=%.2f' % x_Sc[xval7]) #,marker=markers[m])

ax1.xaxis.set_ticks_position('both')
ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

plt.grid(True)

#rootdir = r'C:\Users\pgareau\Source\repos\NHEET-1D-model\NHEET-1D-model\Figures'
workingDir = os.getcwd()
dirName = workingDir + r'\Solutions\Figures'
# fileName = 'test.png'
plt.savefig(dirName+'/'+fileName,bbox_inches = 'tight') #dpi=100


print('Plots Complete')

#dirName = r'C:\Users\pgareau\source\repos\NHEET-1D-model\NHEET-1D-model\Solutions\Data'
import csv
with open(fileName+'.csv', mode='w', newline='') as CVS_file_writer:
    fieldnames = ['t_Sc', 'yValScf1', 'yValScf2', 'yValScf3', 'yValScf4', 'yValScf5', 'yValScs1', 'yValScs2', 'yValScs3', 'yValScs4', 'yValScs5']
    write_file = csv.DictWriter(CVS_file_writer, fieldnames=fieldnames)
    write_file.writeheader()
    i=0
    t_Sc2 = t_Sc/3600.
    while i<len(T_Sc): 
        write_file.writerow({'t_Sc': '%.6f' %t_Sc2[i], 'yValScf1': '%.2f' %yValScf1[i], 'yValScf2': '%.2f' %yValScf2[i], 'yValScf3': '%.2f' %yValScf3[i], 
                                   'yValScf4': '%.2f' %yValScf4[i], 'yValScf5': '%.2f' %yValScf5[i], 'yValScs1': '%.2f' %yValScs1[i], 
                                 'yValScs2': '%.2f' %yValScs2[i], 'yValScs3': '%.2f' %yValScs3[i], 'yValScs4': '%.2f' %yValScs4[i], 
                                 'yValScs5': '%.2f' %yValScs5[i]})
        i += 1   

print('Post Processing Complete')
print('END')

plt.show()
plt.close(fig)