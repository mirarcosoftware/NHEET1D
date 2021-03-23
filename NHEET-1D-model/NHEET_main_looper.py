##### TITLE BLOCK #####
# Created by D Cerantola, January 15, 2021
#
# Description: 
# -completes a parametric study 1D conservation of energy analysis on a packed bed
#
# Required input functions:
# -NHEET_1D_model_glbl.py
# -NHEET_other_fns.py
# -NHEET_Tprofile_fns.py
# -NHEET_data_posting.py
# -fluid_properties12.py 
# -tictoc.py
#
# Output: parametric study results
#
### Revisions ###
# Feb10, 2021
# - changed inputs to number of timesteps and velocity
# - save responses and heat transfer distribution to file.
# Mar20, 2021
# - 3 inlet bc options (const velocity, flow rate, massflow)
# - moved fns into other_fns.py and Tprofile_fns.py
#
#
# Notes
# import jupytext
#jupytext --to py notebook.ipynb                 # convert notebook.ipynb to a .py file
#jupytext --to notebook notebook.py              # convert notebook.py to an .ipynb file with no outputs
#jupytext --to notebook --execute notebook.py    # convert notebook.py to an .ipynb file and run it 
# in command prompt > jupytext --to py NHEET_main_looper.ipynb 
#
# Copyright 2021
def NHEET_main(const_input_vals, var_inputs):
    # #############################
    # #####     Libraries     #####
    # #############################
    import os
    import numpy as np
    from tictoc import tic, toc
    #import random
    #from termcolor import colored, cprint

    import NHEET_1D_model as nim
    from NHEET_other_fns import reaction_balance,factors_fn
    from NHEET_Tprofile_fns import post_process_responses
    #post-processing
    from NHEET_data_posting import parsed_data_1d, parsed_data_2d, plot_n_y1_y2, plot_x_y1_y2, plot_contour_y1_y2,plot_3dsurf
    # Jupyter setup directory in C:\Users\User\.jupyter\jupyter_notebook_config.py > line265

    # +
    dirRespName = os.getcwd() + '/Solutions/' + 'Responses/' #% mm[0]
    if os.path.exists(dirRespName):
        print('folder already exists')
        #input("Press Enter to continue...")
    else:
        os.makedirs(dirRespName)

#     #### Constant inputs
#     fileName = 'Lab-Design-Test'
#     inlet_bc_type=2 #0=V,1=Qcfm,2=W0 (kg/s)    
#     T_inlet_method = 1 #Choose Between Two Methods: 1 = User-Defined Function, 2 = Input CSV Table
#     T0 = 283.0 #[K] 
#     dTampin = 17. #[K] inlet temperature profile amplitude
#     dlamdain = 1./12. #inlet temperature profile period (x3600s)
#     Nu_method=1 #method 1=, 2=Allen2015
#     CFLin = 500 #approximate CFL number at time=0
#     Ndt=30 #approximate number of timesteps per period
#     first_time_step_size = 10.
#     Enhanced_Schumann = 0 #0=Schumann, 1=Enhanced Schmann
#     time_integrator = 2 #1=Backward Euler, 2=Crank Nicholson
#     rhos = 2700. #kg/m^3
#     cps = 900. #J/kg-K
#     ks = 3. #W/m-K

#     const_input_vals=[fileName,inlet_bc_type,T_inlet_method,T0,dTampin,dlamdain,Nu_method,CFLin,Ndt,first_time_step_size, \
#                       Enhanced_Schumann,time_integrator,rhos,cps,ks]

    inlet_bc_type=const_input_vals[1]
    T_inlet_method=const_input_vals[2]
    
#     # ##Patrick's values
#     # inlet_bc_type=1 #0=V,1=Qcfm,2=W0 (kg/s)
#     # T_inlet_method = 2 #Choose Between Two Methods: 1 = User-Defined Function, 2 = Input CSV Table
#     # const_input_vals=[fileName,inlet_bc_type,T_inlet_method,293.15,10.,24.,1,2000,30,120,0,1,2635,790,2.6]

#     #### Variable inputs (brackets matter)
#     inlet_bcs=[0.009] #V,Qcfm,W0. Set inlet_bc_type 
#     #inlet_bcs=[0.003,0.006,0.009,0.012,0.015,0.018]
#     Ds=[0.1]#container width
#     Lfacs = [1.5] #container cylinder height factor, L=Lfac*D
#     Dps =[0.02] #spherical equivalent particle diameter, m
#     delDps =[0.] #distribution spread, not used
#     #phis = [0.88] #sphericity
#     phis = [1.]
#     Dpcorrs=[1.] #particle diameter correction. ie if want d32 in Ergun's, not used
#     epss = [0.455] #void fraction #uniform spheres
#     #epss = [0.382] #void fraction #spheres distribution
#     #epss = [0.396] #void fraction #rocks distribution
#     #epss=[0.3,0.35,0.4,0.45,0.5]

#     #Patrick's values
#     # inlet_bcs=[525.]
#     # Ds=[14*.3048]
#     # Lfacs=[8./14.]
#     # Dps=[0.02]
#     # epss=[0.3]
#     var_inputs = np.array([inlet_bcs,Ds,Lfacs,Dps,delDps,phis,Dpcorrs,epss])

    if inlet_bc_type==0:
        inlet_bc_label='V_in'
        print('velocity inlet')
    elif inlet_bc_type==1:
        inlet_bc_label='Qcfm'
        print('flow rate inlet')
    elif inlet_bc_type==2:
        inlet_bc_label='W0'
        print('mass flow inlet')

    factor_labels = [inlet_bc_label,'D','L','Dp','delDp','phi','Dpcorr','eps']
    response_labels = ['pamb','dpsErg','dpsK','Tmean','phase','phaseshift','TampL','timex','Nuf','Nus','Nuact']
    track_labels = ['run']+factor_labels+response_labels
    #['A%s' % i for i in range(0,N)]+

    ###########################
    ### Main Function Magic ###
    ###########################
    #tic()
    factors,NN,NNtot=factors_fn(var_inputs)

    responses = np.zeros([NNtot,len(response_labels)]) #initialize array
    rxn_consts=reaction_balance()

    for n in range(0,NNtot): 
        Tprof_consts=nim.inputs_const(factors[n,0],factors[n,1],factors[n,1]*factors[n,2],const_input_vals,rxn_consts)
        nim.inputs_var(factors[n,:])
        pamb,dirItName,x_Sc,T_Sc,t_Sc,Iminmax,tminmax,Tminmax,geom_vals,Matrix,current_time,delta_p,delta_p2 = \
            nim.iterate(str(n),inlet_bc_type,const_input_vals,rxn_consts,Tprof_consts)
        #responses[n,:],x_Sc,T_Sc,t_Sc,Iminmax,tminmax,Tminmax,Matrix = nim.iterate(str(n),inlet_bc_type,dirName,const_input_vals,rxn_consts,Tprof_consts)

        if T_inlet_method == 1:
            Nuvol,Tmean,phase,phaseshift,TampL=post_process_responses(pamb,const_input_vals,rxn_consts,geom_vals, \
                                                    Matrix,Iminmax,tminmax,Tminmax,str(n),dirRespName,dirItName)
        elif T_inlet_method == 2: #not programmed
            Nuvol=[-1,-2,-3]
            Tmean = -100
            phase = -100
            phaseshift = -100
            TampL = -100

        responses[n,:] =np.concatenate((np.array([pamb,delta_p,delta_p2,Tmean,phase,phaseshift,TampL,current_time/3600.]),Nuvol))
        trackvals = [np.concatenate(([n],factors[n],responses[n,:]))]

        fileName = 'responses.txt'
        with open(dirRespName+fileName, "ab") as f: #ab
            if os.stat(dirRespName+fileName).st_size==0:
                np.savetxt(f, trackvals, fmt='%1.7e',delimiter=',',header=",".join(map(str,track_labels)))
            else:
                np.savetxt(f, trackvals,fmt='%1.7e',delimiter=',')

    ###################################################
    ##  Post processing                              ##
    ###################################################
    m=0 #iteration number
    kk=0 #operating condition

    markers = ['o','d','s','*','<','>','^','v','1','2','3','4']*100
    linestyles = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted', \
                  'solid','dashed','dashdot','dotted']*100

    ###########################################
    xInc =factor_labels.index(inlet_bc_label) # force 0 for now
    yInc1 = response_labels.index('phase')
    yInc2= response_labels.index('phaseshift')
    ###########################################\n",
    plot_n_y1_y2(m,kk,markers,linestyles,xInc,yInc1,yInc2,NNtot,NN,factors,responses,factor_labels,response_labels,dirRespName,\
        plot_title='tmp')

    yInc1 = response_labels.index('phase')
    yInc2= response_labels.index('phaseshift')
    ###########################################
    plot_x_y1_y2(m,kk,markers,linestyles,xInc,yInc1,yInc2,NNtot,NN,factors,responses,factor_labels,response_labels,dirRespName,\
        plot_title='tmp',legendonoff=0) #, xlabel='%s' % inlet_bc_label, ylabel1='dp, m', ylabel2='Tmean, C')

    #response_labels = ['pamb','dp1','dp2','Tmean','phase','phaseshift','TampL','timex','Nuf','Nus','Nuact']
    show_2d_fig = 1
    if show_2d_fig ==1 and np.where(NN>1)[0].astype('int32').size>1:
        xcol=factor_labels.index('W0') #factor x-axis
        ycol=factor_labels.index('eps') #factor y-axis
        #ycol=factor_labels.index('Ltotfac') #factor y-axis
    ###############################################################
        iShow=response_labels.index('phaseshift') #response contours
        iShow=response_labels.index('TampL') #response contours
        #iShow2=response_labels.index('dp2') #response lines
        iShow2=[] #no line contours
    ############################################################################
        minmax=0 #0=min, 1=max of response contours

        #plot_overrides = np.array([[1,1]])
        plot_overrides =[]
        plot_contour_y1_y2(m,kk,markers,linestyles,xcol,ycol,iShow,iShow2,minmax,NNtot,NN,var_inputs,factors,responses,\
                           factor_labels,response_labels,dirRespName,\
                           plot_overrides,\
                           plot_title='tmp')
    print('parametric study complete')
    ### END ###
    return()