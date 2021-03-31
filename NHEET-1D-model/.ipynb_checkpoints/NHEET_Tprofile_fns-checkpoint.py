##### TITLE BLOCK #####
# Created by D Cerantola, Mar.22 2021
#
#
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy import integrate
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import r2_score
from termcolor import cprint

from fluid_properties12 import chem_rxn,MolWt,specheat,specheatk, \
    enthalpy, enthalpy_Tnew, enthalpy_T, entropy,viscosity,conductivity

from NHEET_other_fns import NusseltNumber

def findminmax(t_Sc,T_Sc,inc,MM,QQ,cell_ID,Iminmax,tminmax,Tminmax): #requires sinusoidal temperature profile
    dT_Sc = T_Sc[1:,cell_ID]-T_Sc[:-1,cell_ID]#difference wrt time
    sign_dT_Sc = np.sign(dT_Sc)

    #finds phase and phaseshift. Assumes sinusoidal inlet temperature profile
    idx=-1
    if inc > 0:
        #i=inc
        for j in range(0,MM):
            if sign_dT_Sc[inc,j] + sign_dT_Sc[inc-1,j] ==0:
                idx = next((i for i, x in enumerate(Iminmax[:,j]) if x==0), None)
                Iminmax[idx,j]=inc #index of maximum or minimum
                tminmax[idx,j]=t_Sc[inc]/3600. #time of maximum or minimum 
                Tminmax[idx,j]=T_Sc[inc,cell_ID[j]] #temperature of maximum or minimum
                if j ==0 and idx%2==0:
                    cprint('\t %s inlet period complete' % str(idx/2.),'yellow')
                if j == MM-1 and j <len(cell_ID)-1:
                    MM+=1
                if j == len(cell_ID)-1 and idx >2:
                    if 0.995*abs(Tminmax[idx-1,j]-Tminmax[idx-2,j]) < (Tminmax[idx,j]-Tminmax[idx-1,j]) \
                        and 1.005*abs(Tminmax[idx-1,j]-Tminmax[idx-2,j]) > (Tminmax[idx,j]-Tminmax[idx-1,j]):
                        cprint('\t periodicity temperature evolution achieved','green')
                        QQ= 1
                    if int(idx/2)==10:
                        cprint('\t periodicity not found','red')
                        QQ=1
                        #break
    return Iminmax,tminmax,Tminmax,MM,QQ
    
def post_process_responses(pamb,const_input_vals,rxn_consts,geom_vals,Tprof_consts,Matrix,Iminmax,tminmax,Tminmax,n,dirResp,dirName):
    Nu_method=const_input_vals[6]
    rhos,cps,ks=const_input_vals[12:]
    MW,Rgas,p_frc,z_st,p_mol,ptot,rxC=rxn_consts
    Dp,L,area,eps,W0,N=geom_vals
    lamda=Tprof_consts[-1]
    #Matrix = np.array([x_Sc,t_Sc,T_Sc,delta_ps,ps,Pt,xM,V])
    x_Sc = Matrix[0]
    t_Sc = Matrix[1]
    T_Sc = Matrix[2]
    delta_ps=Matrix[3]
    ps=Matrix[4]
    Pt=Matrix[5]
    xM=Matrix[6]
    V=Matrix[7]
    #Tf = np.mean(T_Sc[:,:N],axis=1)
    Tf = T_Sc[:,:N]
    Ts = T_Sc[:,N:]
    
    ### Geometry stuff
    Ap = np.pi*Dp**2.
    Vp = np.pi*Dp**3/6.

    Volt = L*area
    Volp = Volt*(1-eps)

    Np = Volp/Vp
    Atp= Ap*Np
    #print(Np)
    #Atp = 6*(1-eps)/(Dp)
    denom=(Atp*(x_Sc[1]-x_Sc[0])/L)
    ###
    
    rhof = np.array([[0.]*(N)]*len(t_Sc))
    cpf = np.array([[0.]*(N)]*len(t_Sc))
    muf = np.array([[0.]*(N)]*len(t_Sc))
    kf = np.array([[0.]*(N)]*len(t_Sc))
    for j in range(0,len(t_Sc)):
        for i in range(0,N):
            rhof[j,i] = pamb/Rgas/Tf[j,i] #kg/m^3 
            cpf[j,i] = specheat(Tf[j,i], p_frc, rxC)*1000./MW #J/kg-K
            muf[j,i] = viscosity(Tf[j,i], p_frc, rxC) #kg/m-s
            kf[j,i] = conductivity(Tf[j,i], p_frc, rxC) #W/m-K    #Unused
    
    Qact = np.array([[0.]*(N)]*len(t_Sc))
    Qf = np.array([[0.]*(N)]*len(t_Sc))
    Qs = np.array([[0.]*(N)]*len(t_Sc))
    hact = np.array([[0.]*(N)]*len(t_Sc)) 
    hf = np.array([[0.]*(N)]*len(t_Sc))
    hs = np.array([[0.]*(N)]*len(t_Sc))
    Nuf = np.array([[0.]*(N)]*len(t_Sc))
    Nus = np.array([[0.]*(N)]*len(t_Sc))
    Nuact = np.array([[0.]*(N)]*len(t_Sc))
    
    for j in range(1,len(t_Sc)):
        iVal=0
        Rep = rhof[j,iVal]*Dp*V[j,iVal]/muf[j,iVal]
        Pr= muf[j,iVal]*cpf[j,iVal]/kf[j,iVal]
        Tfref=Tf[j,0]
        for i in range(0,N):
                #Rep = rhof[j,i]*Dp*V[j,i]/muf[j,i]
                #Pr= muf[j,i]*cpf[j,i]/kf[j,i]
                Nuact[j,i] = NusseltNumber(eps,Rep,Pr,Nu_method) ####################
                hact[j,i] = Nuact[j,i]/Dp*kf[j,i]*(Ts[j,i]-Tf[j,i])/(Ts[j,i]-Tfref)
                Qact[j,i] = hact[j,i]*(Ts[j,i]-Tfref) 
        for i in range(1,N):
            if i == N-1: #fluid values
                Qf[j,i] = (Qf[j,i-2]-Qf[j,i-1])/(x_Sc[i-2]-x_Sc[i-1])*(x_Sc[i]-x_Sc[i-1])+Qf[j,i-1]
                #hf[j,i] = (hf[j,i-2]-hf[j,i-1])/(x_Sc[i-2]-x_Sc[i-1])*(x_Sc[i]-x_Sc[i-1])+hf[j,i-1]
                #Qf[j,i] = rhof[j,i]*area*V[j,i]*cpf[j,i]*0.5*(3*Tf[j,i]-4*Tf[j,i-1]+Tf[j,i-2])/denom
                hf[j,i] = Qf[j,i]/(Ts[j,i]-Tfref)
                
                #Qf[j,0] = (Qf[j,2]-Qf[j,1])/(x_Sc[2]-x_Sc[1])*(x_Sc[0]-x_Sc[1])+Qf[j,1]
                #hf[j,0] = (hf[j,2]-hf[j,1])/(x_Sc[2]-x_Sc[1])*(x_Sc[0]-x_Sc[1])+hf[j,1]
                Qf[j,0] = rhof[j,0]*area*V[j,0]*cpf[j,0]*0.5*(-3*Tf[j,0]+4*Tf[j,1]-Tf[j,2])/denom
                hf[j,0] = Qf[j,0]/(Ts[j,0]-Tfref)
                Nuf[j,0] = hf[j,0]*Dp/kf[j,0]
            else:    
                #Qf[j,i] = rhobar*area*V_in*cpbar*(0.5*(Tf[j,i+1]+Tf[j,i])-0.5*(Tf[j,i]+Tf[j,i-1]))
                Qf[j,i] = rhof[j,i]*area*V[j,i]*cpf[j,i]*(0.5*(Tf[j,i+1]-Tf[j,i-1]))/denom
                #if abs((Ts[j,i]-Tf[j,i]))>0.3:
                if abs((Ts[j,i]-Tfref))>0.03:
                    #hf[j,i] = Qf[j,i]/(Ts[j,i]-Tf[j,i])
                    hf[j,i] = Qf[j,i]/(Ts[j,i]-Tfref)
                    #hf[j,i] = Qf[j,i]/(Atp*1/N*(Ts[j,i]-Tf[j,i]))
                else: # when Ts=Tf, get a 0/0 situation. 
                    hf[j,i]=hf[j,i-1]
            Nuf[j,i] = hf[j,i]*Dp/kf[j,i]

            if j == len(t_Sc)-1: #solid values
                Qs[j-1,i] = Qs[j-1,i]
                hs[j-1,i] = hs[j-1,i]            
            elif i == N-1:
                Qs[j,i]=Volp*(x_Sc[i]-x_Sc[i-1])/L*rhos*cps*(Ts[j+1,i]-Ts[j-1,i])/(t_Sc[j+1]-t_Sc[j-1])/denom
                if abs((Ts[j,i]-Tfref))>0.03:
                    #hs[j,i]= Qs[j,i]/(Atp*(x_Sc[i]-x_Sc[i-1])/L*(Tf[j,i]-Ts[j,i]))
                    #hs[j,i]= Qs[j,i]/(Tf[j,i]-Ts[j,i])
                    hs[j,i]= Qs[j,i]/(Tfref-Ts[j,i])
                else:
                    hs[j,i]=hs[j,i-1]
                Qs[j,0] = Volp*(x_Sc[1]-x_Sc[0])/L*rhos*cps*(Ts[j+1,0]-Ts[j-1,0])/(t_Sc[j+1]-t_Sc[j-1])/denom
                #hs[j,0] = (hs[j,2]-hs[j,1])/(x_Sc[2]-x_Sc[1])*(x_Sc[0]-x_Sc[1])+hs[j,1]
                hs[j,0] = Qs[j,0]/(Tfref-Ts[j,0])
                Nus[j,0] = hs[j,0]*Dp/kf[j,0]
            else:
                Qs[j,i]=Volp*(x_Sc[i+1]-x_Sc[i])/L*rhos*cps*(Ts[j+1,i]-Ts[j-1,i])/(t_Sc[j+1]-t_Sc[j-1])/denom
                #if abs((Ts[j,i]-Tf[j,i]))>0.3:
                if abs((Ts[j,i]-Tfref))>0.03:
                    #hs[j,i]= Qs[j,i]/(Atp*(0.5*(x_Sc[i+1]-x_Sc[i-1]))/L*(Tf[j,i]-Ts[j,i]))
                    #hs[j,i]= Qs[j,i]/((Tf[j,i]-Ts[j,i]))
                    hs[j,i]= Qs[j,i]/((Tfref-Ts[j,i]))
                else:
                    hs[j,i] = hs[j,i-1]
            Nus[j,i] = hs[j,i]*Dp/kf[j,i]
    
    ## 
    jpos = 0 #0=inlet, -1 = outlet
    idx=next((i for i, x in enumerate(Iminmax[:,jpos]) if x==0), None)-1 
    
    #print(Iminmax,idx)
    istart=idx-2
    i0 = Iminmax[istart,jpos]
    #i1a = Iminmax[istart+1,-1]
    i2 = Iminmax[istart+1,jpos]
    i4 = Iminmax[istart+2,jpos]
    i3 = int(0.5*(i2+i4))
    i1 = int(0.5*(i0+i2))
    iall = np.array([i0,i1,i2,i3,i4])
    iall_labels = ['lam0','lam25','lam50','lam75','lam100']
    #print(iall)
    #print(t_Sc[iall]/3600.,Tf[iall,jpos])
    
    Nufvol=integrate.simps(integrate.simps(Nuf[i0:i4+1,:], x=None, dx=1, axis=1, even='avg')/N, x=t_Sc[i0:i4+1], even='avg')/ \
        (t_Sc[i4]-t_Sc[i0])
    Nusvol=integrate.simps(integrate.simps(Nus[i0:i4+1,:], x=None, dx=1, axis=1, even='avg')/N, x=t_Sc[i0:i4+1], even='avg')/ \
        (t_Sc[i4]-t_Sc[i0])
    Nuactvol=integrate.simps(integrate.simps(Nuact[i0:i4+1,:], x=None, dx=1, axis=1, even='avg')/N, x=t_Sc[i0:i4+1], even='avg')/ \
        (t_Sc[i4]-t_Sc[i0])
    
    Nuvol = [Nufvol,Nusvol,Nuactvol]
    
    #print(iall)
    print('Nuvol',Nuvol)
    
    plot_colors = ['r','b','g','y','c','r','b','g','y','c']              
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('length, m')
    ax1.set_ylabel('Temperature, C') 
    ax1.tick_params(axis='y') 

    for i in range(0,len(iall)):
        ax1.plot(x_Sc,T_Sc[iall[i],:N]-273.15,color=plot_colors[i],linestyle='solid',label='t=%.2f h' % (t_Sc[iall[i]]/3600.) )
        ax1.plot(x_Sc,T_Sc[iall[i],N:]-273.15,color=plot_colors[i],linestyle='dashdot')

    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)
    plt.savefig(dirName+'T_vs_x'+n,bbox_inches = 'tight') #dpi=100
    plt.show()
           
    ### dataset    
    indx=-1
    Qinlet=W0*(enthalpy(Tf[iall[indx],0], p_frc, 0,0,0)-enthalpy(298.15, p_frc, 0,0,0))*1000./MW
    Qoutlet=W0*(enthalpy(Tf[iall[indx],-1], p_frc, 0,0,0)-enthalpy(298.15, p_frc, 0,0,0))*1000./MW
    Qwall=sum(Qact[iall[indx],:]*denom)
    
    cprint('\t velocity = %.5f m/s' % (V[iall[indx],0]),'cyan')
    cprint('\t delta_ps = %.5f Pa' % (delta_ps[iall[indx],0]),'cyan')
    #print(Tf[iall[indx],0],rhof[iall[indx],0],muf[iall[indx],0],Dp)
    print('Energy balance',Qinlet,Qoutlet,Qwall)
    cprint('\t mass flow rate = %.5f kg/s' % (rhof[iall[indx],0]*area*V[iall[indx],0]),'blue')
    cprint('\t particle Reynolds number = %.1f' % (rhof[iall[indx],0]*Dp*V[iall[indx],0]/muf[iall[indx],0]),'blue')
    cprint('\t Tf.out = %.5f K, Ts.out= %.5f K' % (Tf[iall[indx],-1],Ts[iall[indx],-1]),'cyan')

    ### Figures ###
    fig, ax1 = plt.subplots()
    plt.title('At t = %.2f h' % (t_Sc[iall[indx]]/3600.))
    ax1.set_xlabel('length, m')
    ax1.set_ylabel('Nusselt Number') 
    ax1.plot(x_Sc,Nuact[iall[indx],:],color=plot_colors[0],linestyle='solid',label='correlation')
    ax1.plot(x_Sc,Nuf[iall[indx],:],color=plot_colors[1],linestyle='solid',label='fluid')
    ax1.plot(x_Sc,Nus[iall[indx],:],color=plot_colors[2],linestyle='solid',label='solid')
    #plt.plot(x_Sc,Nuf[iall[indx],:], 'r', x_Sc,Nus[iall[indx],:], 'b', x_Sc,Nuact[iall[indx],:], 'g')
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))
    plt.grid(True)
    plt.savefig(dirName+'Nu_vs_x'+n,bbox_inches = 'tight') #dpi=100
    plt.show()
     
    fig, ax1 = plt.subplots()
    plt.title('At t = %.2f h' % (t_Sc[iall[indx]]/3600.))
    ax1.set_xlabel('length, m')
    ax1.set_ylabel('Heat flux, W/m2')
    ax1.plot(x_Sc,Qact[iall[indx],:],color=plot_colors[0],linestyle='solid',label='correlation') 
    ax1.plot(x_Sc,Qf[iall[indx],:],color=plot_colors[1],linestyle='dashed',label='fluid')
    ax1.plot(x_Sc,-Qs[iall[indx],:],color=plot_colors[2],linestyle='dashed',label='solid')
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))
    plt.grid(True)
    plt.savefig(dirName+'Q_vs_x'+n,bbox_inches = 'tight') #dpi=100
    plt.show()

    fig, ax1 = plt.subplots()
    plt.title('At t = %.2f h' % (t_Sc[iall[indx]]/3600.))
    ax1.set_xlabel('length, m')
    ax1.set_ylabel('Heat transfer coefficient, W/m2-K')
    ax1.plot(x_Sc,hact[iall[indx],:],color=plot_colors[0],linestyle='solid',label='correlation') 
    ax1.plot(x_Sc,hf[iall[indx],:],color=plot_colors[1],linestyle='solid',label='fluid')
    ax1.plot(x_Sc,hs[iall[indx],:],color=plot_colors[2],linestyle='solid',label='solid')
    #ax1.plot(x_Sc,Qf[iall[indx],:]/(Atp*dx_Sc/L),color=plot_colors[1],linestyle='dashdot',label='fluid')
    #ax1.plot(x_Sc,-Qs[iall[indx],:]/(Atp*dx_Sc/L),color=plot_colors[2],linestyle='dashdot',label='solid')
    #plt.plot(x_Sc,Nuf[iall[indx],:], 'r', x_Sc,Nus[iall[indx],:], 'b', x_Sc,Nuact[iall[indx],:], 'g')
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)
    plt.savefig(dirName+'h_vs_x'+n,bbox_inches = 'tight') #dpi=100
    plt.show()
    
    
    fig, ax1 = plt.subplots()
    plt.title('At t = %.2f h' % (t_Sc[iall[indx]]/3600.))
    ax1.set_xlabel('length, m')
    ax1.set_ylabel('pressure, Pa')
    ax1.plot(x_Sc,delta_ps[iall[indx],:],color=plot_colors[0],linestyle='solid',label='correlation') 
    #ax1.plot(x_Sc,Qf[iall[indx],:]/(Atp*dx_Sc/L),color=plot_colors[1],linestyle='dashdot',label='fluid')
    #ax1.plot(x_Sc,-Qs[iall[indx],:]/(Atp*dx_Sc/L),color=plot_colors[2],linestyle='dashdot',label='solid')
    #plt.plot(x_Sc,Nuf[iall[indx],:], 'r', x_Sc,Nus[iall[indx],:], 'b', x_Sc,Nuact[iall[indx],:], 'g')
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))

    plt.grid(True)
    plt.savefig(dirName+'deltaps_vs_x'+n,bbox_inches = 'tight') #dpi=100
    plt.show()  


    
    ### Fluent data input file for wall heat transfer ###
    def linear_fn_cf(x,a,b,c):
        #return (a*np.exp(x*b)+c)
        return (a*np.power(x,b)+c)
    
    ii=iall[indx]
    x_labels_all = ['y','Tf','Ts','hact','hf','hs','Qact','Qf','Qs','Nuact','Nuf','Nus','del_ps','ps','Pt','xM','V','rhof','cpf','muf','kf']
    juse = [0,1,2,3,6,9]
    x_labels = (x_labels_all[i] for i in juse)

    xvals = np.concatenate((np.array([L-x_Sc]).T,np.array([Tf[ii,:]]).T,np.array([Ts[ii,:]]).T \
        ,np.array([hact[ii,:]]).T,np.array([hf[ii,:]]).T,np.array([hs[ii,:]]).T,np.array([Qact[ii,:]]).T,np.array([Qf[ii,:]]).T,np.array([-Qs[ii,:]]).T \
        ,np.array([Nuact[ii,:]]).T,np.array([Nuf[ii,:]]).T,np.array([Nus[ii,:]]).T\
        ,np.array([delta_ps[ii,:]]).T,np.array([ps[ii,:]]).T,np.array([Pt[ii,:]]).T,np.array([xM[ii,:]]).T,np.array([V[ii,:]]).T\
        ,np.array([rhof[ii,:]]).T,np.array([cpf[ii,:]]).T,np.array([muf[ii,:]]).T,np.array([kf[ii,:]]).T),axis=1)
    
    #[delta_ps,ps,Pt,xM,V,rhof,cpf,muf,kf,*Rep,*Pr]
    
    #print(np.shape(xvals))
    #print(xvals)
    print('saved to %s\n' % dirName) 
    fileName = 'code1d_heat_transfer_%s.csv' % (iall_labels[indx])
    with open(dirName+fileName, "w") as f: #ab
        np.savetxt(f, xvals[:,juse], fmt='%1.7e',delimiter=',',header="[Name]\ncode1d\n\n[Data]\n"+",".join(map(str,x_labels)),comments='')
    f.close()
    
    fileName = 'code1d_output_all_%s.csv' % (iall_labels[indx])    
    with open(dirName+fileName, "w") as f: #ab
        np.savetxt(f, xvals, fmt='%1.7e',delimiter=',',header="[Name]\ncode1d\n\n[Data]\n"+",".join(map(str,x_labels_all)),comments='')
    f.close()    
    ###    
    nPoly=2
    xVal=(L-x_Sc)/L
    #yVal = Qact[ii,:]/(Atp*(x_Sc[1]-x_Sc[0])/L) #[W/m2]
    yVal = Ts[ii,:]/Tf[ii,0]
    
    coefs_p = poly.polyfit(xVal, yVal, nPoly,full=True)
    yres_p = poly.polyval(xVal, coefs_p[0])
    r2_p=r2_score(yVal, yres_p)
    print('Ts poly fit',coefs_p[0],r2_p)
    
    coefs = curve_fit(linear_fn_cf, xVal, yVal)#,bounds=([1.,0.],[1.+1.e-9,5.])
    yres = linear_fn_cf(xVal,coefs[0][0],coefs[0][1],coefs[0][2])
    r2=r2_score(yVal, yres)
    print('Ts myfn fit',coefs[0],r2)
    
    Re0=rhof[ii,0]*V[ii,0]*Dp/muf[ii,0]
    Pr0=muf[ii,0]*cpf[ii,0]/kf[ii,0]
    
    qsave=0
    if qsave==0:
        print("q not saved line335")
    elif qsave==1:
        Qlabels = ['W0','eps','Dp','L','lamda','Tf0','V0','rhof0','cpf0','muf0','kf0','Rep','Pr']+['c%s' % i for i in range(0,len(coefs[0]))]+['r2']
        QMatrix=np.array([np.concatenate((np.array([W0,eps,Dp,L,lamda,Tf[ii,0],V[ii,0],rhof[ii,0],cpf[ii,0],muf[ii,0],kf[ii,0],Re0,Pr0]),coefs[0],np.array([r2])))])
        #print(QMatrix)
        #fileName = 'code1d_QMatrix_%s.csv' % (iall_labels[indx])
        fileName = 'code1d_TwMatrix_%s.csv' % (iall_labels[indx])
        with open(dirResp+fileName, "a") as f: #ab
            if os.stat(dirResp+fileName).st_size==0:
                np.savetxt(f, QMatrix, fmt='%1.7e',delimiter=',',header=",".join(map(str,Qlabels)))
            else:
                np.savetxt(f, QMatrix,fmt='%1.7e',delimiter=',')
        f.close()
    ###
        
    ### phase shift and outlet amplitude ###
    idx0 = next((i for i, x in enumerate(Iminmax[:,0]) if x==0), None)-1
    idx1 = next((i for i, x in enumerate(Iminmax[:,-1]) if x==0), None)-1    
    phase = tminmax[idx1,-1]-tminmax[idx1-2,-1]
#     if (np.shape(tminmax)[0]-2)%2 == idx%2:
#         xm=2
#     else:
#         xm=3
    #phaseshift = tminmax[np.shape(tminmax)[0]-xm,0]-tminmax[idx,-1]
    phaseshift = tminmax[idx1,-1]-tminmax[idx0,0]
    
    #print(Iminmax[idx1-2,-1],Iminmax[idx1,-1]+1)
    Tmean = np.mean(T_Sc[Iminmax[idx1-2,-1]:Iminmax[idx1,-1]+1,:])
    
    TampL = Tminmax[idx1,-1]-Tminmax[idx1-1,-1]
    
    return Nuvol,Tmean,phase,phaseshift,TampL