import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from termcolor import cprint
from scipy.interpolate import griddata
import sys

#xcol=2
def parsed_data_1d(inputs,factors,xcol,NN):
    #print(inputs,factors,xcol,NN)
    
    idx_all = np.array(range(0,np.shape(factors)[1])) #number of factors indices

    idx_rep_all = np.where(NN>1)[0].astype('int32')
    if len(idx_rep_all) < 2:
        idx_rep_other =[xcol,0]
        idx_2d=[xcol]
        idx_other=range(0,np.shape(factors)[0])
        keep_rows=idx_other
    else:
        idx_rep_other_tmp=idx_rep_all[~np.in1d(range(len(idx_rep_all)),xcol)]
        idx_rep_other = idx_all[idx_rep_other_tmp]   
        idx_2d=np.concatenate(([xcol],idx_rep_other)) #plotting indices
        idx_other=idx_all[~np.in1d(range(len(idx_all)),idx_2d)] #factors-plotting
        plot_val = [0]*np.shape(factors)[1] #other plot values
        idx_2d_set = np.array([factors[:,i]==inputs[i][plot_val[i]] for i in idx_other]) #true/false evaluation of other factors
        keep_rows = np.where(np.array([all(idx_2d_set[:,i])==True for i in range(0,np.shape(factors)[0])]))[0].astype('int32')  #parsing to index 0 for other factors

    #icols1 = np.where(NN==1)[0].astype('int32')
    #note that lexsort sorts first from the last row, so sort keys are in reverse order

    xx=factors[keep_rows,:]
    #ix = np.lexsort((data[:, 3][::-1], data[:, 2]))
    #ix = np.lexsort(([xx[:,i] for i in np.flip(idx_2d)]))

    sorted_order = np.lexsort(([factors[:,i] for i in idx_2d])).astype('int32') #sorted row indices
    #sorted_order = np.lexsort(([factors[:,i] for i in np.flip(idx_2d)])).astype('int32') #sorted row indices
    return sorted_order,idx_rep_other

def parsed_data_2d(inputs,factors,xcol,ycol,*args):

    idx_all = np.array(range(0,np.shape(factors)[1])) #number of factors indices
    idx_2d=np.array([xcol,ycol]) #plotting indices
    
    plot_val = [0]*np.shape(factors)[1] #other plot values
    
    #print(args)
    #print(args[0])
    #print(args[0,0])
    if args:
        plot_overrides = args[0]
        for i in range(0,len(plot_overrides)):
            plot_val[plot_overrides[i,0]]=plot_overrides[i,1]
    
    #icols1 = np.where(NN==1)[0].astype('int32')

    idx_other=idx_all[~np.in1d(range(len(idx_all)),idx_2d)] #factors-plotting
    #print(idx_all,idx_2d,idx_other)
####### keep
#     idx_rep_all = np.where(NN>1)[0].astype('int32')
#     idx_rep_other_tmp=idx_rep[~np.in1d(range(len(idx_rep)),idx_2d)]
#     idx_rep_other = idx_rep_all[idx_rep_other_tmp]

    #note that lexsort sorts first from the last row, so sort keys are in reverse order
    #data=factors
    idx_2d_set = np.array([factors[:,i]==inputs[i][plot_val[i]] for i in idx_other]) #true/false evaluation of other factors
    keep_rows = np.where(np.array([all(idx_2d_set[:,i])==True for i in range(0,np.shape(factors)[0])]))[0].astype('int32')  #parsing to index 0 for other factors
    xx=factors[keep_rows,:]
    
    #ix = np.lexsort((data[:, 3][::-1], data[:, 2]))
    #ix = np.lexsort(([xx[:,i] for i in np.flip(idx_2d)]))

    sorted_order = np.lexsort(([xx[:,i] for i in idx_2d])).astype('int32') #sorted row indices
    #sorted_order = np.lexsort(([xx[:,i] for i in np.flip(idx_2d)])).astype('int32') #sorted row indices
    return keep_rows[sorted_order]

def plot_n_y1_y2(mm,kk,markers,linestyles,xInc,yInc1,yInc2,NNtot,NN,factors,responses,factor_labels,response_labels,dirName,**kwargs):
                 #plot_title,xlabel,ylabel1,ylabel2):
    #defaults
    m=0
    plot_title='COND%s' % kk
    xlabel= 'test case' 
    ylabel1= '%s' %response_labels[yInc1]
    ylabel2= '%s' %response_labels[yInc2]
    for key, value in kwargs.items():
        if key is 'plot_title': plot_title=value
        if key is 'xlabel': xlabel=value
        if key is 'ylabel1': ylabel1=value
        if key is 'ylabel2': ylabel2=value

#     Ni = int(NNtot/np.prod(NN[xInc])) #number of sets
#     #Nj = int(np.prod(NN[:xInc])) #skip value
#     #Nk = NN[xInc] #number of values in set
    
#     irows1,idx_rep_other=parsed_data_1d(inputs,factors,xInc)
#     Nj = NN[idx_rep_other[0]]
#     Nk = NN[idx_rep_other[1]]
#     #print(irows1,Nj,Nk)
#     #irows2=parsed_data(inputs,factors,xInc,yInc2)

#     if xInc ==0:
#         N1=NN[0]
#     else:
#         N1 = 1
#     N2 = len(NN)
#     #print(Ni,Nj,Nk)

    #for m in range(0,Ni):
        #xVal = factors[int(m*N1):int(Nj*Nk*(m+1)):Nj,xInc]
    xVal1 = np.array(range(0,np.shape(factors)[0])) 
    #print(xVal1)
    yVal1 = responses[:,yInc1]
    yVal2 = responses[:,yInc2]
        
        #print(xVal)

    fig, ax1 = plt.subplots()
    #ax1title=('C%sH%s-air performance at Cond%s, T0=%sK and xM=%s' % (rfC,rfH,kk,np.round(T[0],2),np.round(xM[0],2)))
    ax1title=(plot_title)
    plt.title(ax1title)#, fontdict=None, loc='center', pad=None)
    colors = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1,color=colors)
        #xVal = factors[int(m*N1):int(Nj*Nk*(m+1)):Nj,xInc]
        #yVal = responses[int(m*N1):int(Nj*Nk*(m+1)):Nj,yInc1,kk]
    ax1.plot(xVal1,yVal1,color=colors,linestyle=linestyles[m],marker=markers[m])
    ax1.tick_params(axis='y', labelcolor=colors)
    ax1.xaxis.set_ticks_position('both')
    #ax1.yaxis.set_ticks_position('both')
    ##ax1.set_label('Label via method')
    #ax1.legend(title='factors',loc='center left', bbox_to_anchor=(1.25, 0.5))

    # # Put a legend below current axis
    # leg=ax1.legend(title='ARdiff',loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #           fancybox=True, shadow=True, ncol=ll2)
    # leg._legend_box.align = "right"
    # #leg.get_title().set_position((-20, 0))
    if yInc1 != yInc2:
        ax2 = ax1.twinx()
        colors = 'tab:blue'
        ax2.set_ylabel(ylabel2, color=colors)  # we already handled the x-label with ax1
            #xVal = factors[int(m*N1):int(Nj*Nk*(m+1)):Nj,xInc]
            #yVal = responses[int(m*N1):int(Nj*Nk*(m+1)):Nj,yInc2,kk]
        ax2.plot(xVal1,yVal2,color=colors,linestyle=linestyles[m],marker=markers[m])#,label='inputs[1][m]'
        ax2.tick_params(axis='y', labelcolor=colors)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    
    #dirName = rootdir+'it%s' % (mm)  
    #fileName = 'COND%s_%s_vs_%s.png' % (kk,response_labels[yInc1],'N')
    fileName = 'responses_%s_vs_%s.png' % (response_labels[yInc1],'N')
    #print(dirName+'/'+fileName)
    plt.savefig(dirName+fileName,bbox_inches = 'tight') #dpi=100
    plt.show()
    plt.close(fig)
    return ()

def plot_x_y1_y2(mm,kk,markers,linestyles,xInc,yInc1,yInc2,NNtot,NN,factors,responses,factor_labels,response_labels,dirName,**kwargs):
                 #plot_title,xlabel,ylabel1,ylabel2):

    #defaults
    plot_title='COND%s' % kk
    xlabel= '%s' % factor_labels[xInc]
    ylabel1= '%s' %response_labels[yInc1]
    ylabel2= '%s' %response_labels[yInc2]
    legendonoff=1
    for key, value in kwargs.items():
        if key is 'plot_title': plot_title=value
        if key is 'xlabel': xlabel=value
        if key is 'ylabel1': ylabel1=value
        if key is 'ylabel2': ylabel2=value
        if key is 'legendonoff': legendonoff=value

    Ni = int(NNtot/np.prod(NN[xInc])) #number of sets
    #Nj = int(np.prod(NN[:xInc])) #skip value
    #Nk = NN[xInc] #number of values in set
    
    inputs = [0]*np.shape(factors)[1]
    for i in range(0,np.shape(factors)[1]):
        inputs[i] = np.unique(factors[:,i])
    
    irows1,idx_rep_other=parsed_data_1d(inputs,factors,xInc,NN)
    Nj = NN[xInc]
    if len(idx_rep_other)>1:
        Nk = NN[idx_rep_other[1]]
    else: Nk=1.
    #print(irows1,Nj,Nk)
    #irows2=parsed_data(inputs,factors,xInc,yInc2)

#     if xInc ==0:
#         N1=NN[0]
#     else:
#         N1 = 1
#     N2 = len(NN)
#     #print(Ni,Nj,Nk)

    #for m in range(0,Ni):
        #xVal = factors[int(m*N1):int(Nj*Nk*(m+1)):Nj,xInc]
    xVal1 = factors[irows1,xInc]
    #print(xVal1)
    yVal1 = responses[irows1,yInc1]
    yVal2 = responses[irows1,yInc2]
        
        #print(xVal)

    fig, ax1 = plt.subplots()
    #ax1title=('C%sH%s-air performance at Cond%s, T0=%sK and xM=%s' % (rfC,rfH,kk,np.round(T[0],2),np.round(xM[0],2)))
    ax1title=(plot_title)
    plt.title(ax1title)#, fontdict=None, loc='center', pad=None)
    colors = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1,color=colors)
    for m in range(0,Ni):
        #xVal = factors[int(m*N1):int(Nj*Nk*(m+1)):Nj,xInc]
        #yVal = responses[int(m*N1):int(Nj*Nk*(m+1)):Nj,yInc1,kk]
        #print(m*Nj)
        ax1.plot(xVal1[Nj*m:Nj*(m+1)],yVal1[Nj*m:Nj*(m+1)],color=colors,linestyle=linestyles[m],marker=markers[m],label=factors[irows1[(m)*Nj],:])
    ax1.tick_params(axis='y', labelcolor=colors)
    ax1.xaxis.set_ticks_position('both')
    #ax1.yaxis.set_ticks_position('both')
    ##ax1.set_label('Label via method')
    if legendonoff==1: ax1.legend(title='factors',loc='center left', bbox_to_anchor=(1.25, 0.5))

    # # Put a legend below current axis
    # leg=ax1.legend(title='ARdiff',loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #           fancybox=True, shadow=True, ncol=ll2)
    # leg._legend_box.align = "right"
    # #leg.get_title().set_position((-20, 0))
    if yInc1 != yInc2:
        ax2 = ax1.twinx()
        colors = 'tab:blue'
        ax2.set_ylabel(ylabel2, color=colors)  # we already handled the x-label with ax1
        for m in range(0,Ni):
            #xVal = factors[int(m*N1):int(Nj*Nk*(m+1)):Nj,xInc]
            #yVal = responses[int(m*N1):int(Nj*Nk*(m+1)):Nj,yInc2,kk]
            ax2.plot(xVal1[Nj*m:Nj*(m+1)],yVal2[Nj*m:Nj*(m+1)],color=colors,linestyle=linestyles[m],marker=markers[m])#,label='inputs[1][m]'
        ax2.tick_params(axis='y', labelcolor=colors)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    
    #dirName = rootdir+'it%s' % (mm)  
    #fileName = 'COND%s_%s_vs_%s.png' % (kk,response_labels[yInc1],factor_labels[xInc])
    fileName = 'responses_%s_vs_%s.png' % (response_labels[yInc1],factor_labels[xInc])
    #print(dirName+'/'+fileName)
    plt.savefig(dirName+fileName,bbox_inches = 'tight')#,dpi=100
    plt.show()
    plt.close(fig)
    return ()

def plot_contour_y1_y2(mm,kk,markers,linestyles,xcol,ycol,iShow,iShow2,minmax,NNtot,NN,inputs,factors,responses,factor_labels,response_labels,dirName, \
                      *args,**kwargs):
    
    #defaults
    plot_title='COND%s' % kk
    xlabel= '%s' % factor_labels[xcol]
    ylabel1= '%s' %factor_labels[ycol]
    contour_label= '%s' % response_labels[iShow]
    if iShow2:
        contour_label2= '%s' % response_labels[iShow2]
    for key, value in kwargs.items():
        if key is 'plot_title': plot_title=value
        if key is 'xlabel': xlabel=value
        if key is 'ylabel1': ylabel1=value
        if key is 'contour_label': contour_label=value
        if key is 'contour_label2': contour_label2=value
    
    inputs = [0]*np.shape(factors)[1]
    for i in range(0,np.shape(factors)[1]):
        inputs[i] = np.unique(factors[:,i])
    
    xx = inputs[xcol]
    yy = inputs[ycol]
    #print(xx)
    #print(yy)
    
    if args:
        plot_overrides= args[0]
        irows=parsed_data_2d(inputs,factors,xcol,ycol,plot_overrides)
    else:
        irows=parsed_data_2d(inputs,factors,xcol,ycol)
    
    x=factors[irows,xcol]
    y=factors[irows,ycol]
    z=responses[irows,iShow]
    
    xi,yi = np.meshgrid(xx,yy)
    zi = griddata((x,y),z,(xi,yi),method='linear')
    if iShow2:
        zi2 = griddata((x,y),responses[irows,iShow2]/min(responses[irows,iShow2]),(xi,yi),method='linear')
    if minmax==0:
        max_value = np.min(zi)
        minmaxtxt = 'min'
        lblsign=-1.
    else:
        max_value = np.max(zi)
        minmaxtxt = 'max'
        lblsign=1.
    
    local_max_index = np.where(zi==max_value)
    ## retrieve position of your
    max_x = xi[local_max_index[0][0], local_max_index[1][0]]
    max_y = yi[local_max_index[0][0], local_max_index[1][0]]
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS=plt.contourf(xi,yi,zi,10,cmap = plt.cm.jet)#,np.arange(0,1.01,0.01))
    if iShow2:
        CS2 = plt.contour(xi,yi,(zi2-np.min(zi2[:]))/(np.max(zi2[:])-np.min(zi2[:])),10,levels=[0.1,0.5,0.9], colors='k')
    plt.plot(x,y,'k.')
    
    # plot one marker on this position
    plt.plot(max_x, max_y, color="green", marker = "o", fillstyle='none',zorder = 10, 
            markeredgewidth=3,markersize=15, clip_on=False)
    xlbl=(max_x+lblsign*0.05*(max(x)-min(x)))
    ylbl=(max_y+lblsign*0.05*(max(y)-min(y)))
    plt.annotate(minmaxtxt,(xlbl,ylbl),color='green')
    if iShow2:
        plt.annotate('lines = %s' % contour_label2,(xx[0]+0.02*(xx[-1]-xx[0]),yy[0]+0.02*(yy[-1]-yy[0])))
    
    plt.xlabel(xlabel ,fontsize=12)
    plt.ylabel(ylabel1,fontsize=12)
    #ax1title=('%s' % plot_title)
    plt.title(plot_title)#, fontdict=None, loc='center', pad=None)
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(contour_label)
    if iShow2:
        plt.clabel(CS2, fmt='%2.2f', colors='k', fontsize=8)
    
    #dirName = rootdir+'it%s' % (mm)  
    #fileName = 'COND%s_%s_2D_%s_%s.png' % (kk,response_labels[iShow],factor_labels[xcol],factor_labels[ycol])
    fileName = 'responses_%s_2D_%s_%s.png' % (response_labels[iShow],factor_labels[xcol],factor_labels[ycol])
    plt.savefig(dirName+fileName,bbox_inches = 'tight') #dpi=100
    #plt.close(fig)
    return()


def plot_3dsurf(mm,kk,markers,linestyles,xcol,ycol,zcol,iShow,iShow2,minmax,NNtot,NN,factors,responses,factor_labels,response_labels,dirName, \
                      *args,**kwargs):

    #import pylab
    import matplotlib
    #from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d
    #from scipy.interpolate import RegularGridInterpolator as rgi
    #from scipy.interpolate import interpn
    from scipy.interpolate import Rbf
    from statistics import mean

    #kk=kks[0]


    plot_title='COND%s' % kk
    xlabel= '%s' % factor_labels[xcol]
    ylabel= '%s' %factor_labels[ycol]
    zlabel= '%s' %factor_labels[zcol]
    contour_label= '%s' % response_labels[iShow]
    contour_label2= '%s' % response_labels[iShow2]
    for key, value in kwargs.items():
        if key is 'plot_title': plot_title=value
        if key is 'xlabel': xlabel=value
        if key is 'ylabel': ylabel=value
        if key is 'zlabel': zlabel=value
        if key is 'contour_label': contour_label=value
        if key is 'contour_label2': contour_label2=value


    #parametric data
    xi1=factors[:,xcol]
    yi1=factors[:,ycol]
    zi1=factors[:,zcol]
    Vi1=responses[:,iShow]
    Vi2=responses[:,iShow2]


    xi3=factors[:,xcol].reshape(NN[xcol],NN[ycol],NN[zcol])
    yi3=factors[:,ycol].reshape(NN[xcol],NN[ycol],NN[zcol])
    zi3=factors[:,zcol].reshape(NN[xcol],NN[ycol],NN[zcol])
    # Vi3=responses[:,iShow,kk].reshape(NN[zcol],NN[ycol],NN[xcol])

    #find z at (x,y) 2D
    xi_pts=xi3[:,:,0]
    yi_pts=yi3[:,:,0]
    #zi1=zi3[0,0,:]

    Nx=50; Ny=50
    xi1g = np.linspace(min(xi1),max(xi1),Nx)
    yi1g = np.linspace(min(yi1),max(yi1),Ny)
    xi,yi=np.meshgrid(xi1g,yi1g)

    # interp_mesh = np.array(np.meshgrid(xi[0,:], yi[:,0],Viconst))
    # interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((NN[xcol],NN[ycol],3))
    #Vival = min(Vi1)
    if minmax==0:
        Vival = np.min(Vi1)
        #minmaxtxt = 'min'
        #lblsign=-1.
    else:
        Vival = np.max(Vi1)
        #minmaxtxt = 'max'
        #lblsign=1.

    ival=np.where(Vi1==Vival)[0].astype('int32')

    Vi_pts=[[Vival]*NN[ycol]]*NN[xcol]

    Viconst=[[Vival]*Ny]*Nx

    #interpolation function
    rbfi = Rbf(xi1, yi1, Vi1, zi1) #function='linear'
    zi = rbfi(xi,yi,Viconst)

    zi_pts = rbfi(xi_pts,yi_pts,Vi_pts)

    #rbfi2 = Rbf(xi1, yi1, zi1, Vi2,function='thin-plate',smooth=0.001,epsilon=1)
    rbfi2 = Rbf(xi1, yi1, zi1, Vi2,function='linear',smooth=0.3)
    VVi2 = rbfi2(xi,yi,zi)



    # fourth dimention - colormap
    # create colormap according to x-value (can use any 50x50 array)
    color_dimension = VVi2 # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    mplt = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    mplt.set_array([])
    fcolors = mplt.to_rgba(color_dimension)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')
    #surf=ax.plot_surface(xi, yi, zi, cmap=cm.jet,linewidth=0, antialiased=False)

    surf=ax.plot_surface(xi,yi,zi, rstride=1, cstride=1, facecolors=fcolors, cmap='jet', clim=(minn,maxx),
                         vmin=minn, vmax=maxx, shade=False,linewidth=0, antialiased=False,zorder=1)

    ##surf=ax.plot_surface(xi,yi,zi,facecolors=fcolors, cmap='jet')
    ## surf=ax.plot_surface(xi,yi,zi, rstride=1, cstride=1, color=norm, cmap='jet', clim=(minn,maxx),
    ##                      vmin=minn, vmax=maxx, shade=False,linewidth=0, antialiased=False)

    minpt=ax.scatter(xi1[ival],yi1[ival],1.0000*zi1[ival],marker='.',s=100,color='g',zorder=6)#,linewidth=3
    ax.scatter(xi_pts, yi_pts, zi_pts, marker='.',color='k',zorder=5)
    #ax.scatter(xi3, yi3, zi3, marker='.',color='purple')

    #ax.text(xi1[ival][0],yi1[ival][0],1.00001*zi1[ival][0],'min',color='green',zorder=1)

    ax.view_init(20,-20)
    #ax.view_init(azim=0, elev=90)

    x2, y2, _ = proj3d.proj_transform(xi1[ival],yi1[ival],zi1[ival], ax.get_proj())
    label=plt.annotate(
        "(%.2f,%.2f,%.2f)" % (xi1[ival],yi1[ival],zi1[ival]) , 
        xy = (x2, y2), xytext = (-5, 120),
        textcoords = 'offset points', ha = 'left', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'),zorder=10)

    def update_position(e):
        #x2, y2, _ = proj3d.proj_transform(1,1,1, ax.get_proj())
        x2, y2, _ = proj3d.proj_transform(xi1[ival],yi1[ival],zi1[ival], ax.get_proj())
        label.xy = x2,y2
        #label.update_positions(fig.canvas.renderer)
        #minpt.update_positions(fig.canvas.get_renderer())
        label.update_positions(fig.canvas.get_renderer())
        fig.canvas.draw()
    fig.canvas.mpl_connect('button_release_event', update_position)

    #ax.text(1.,1.,5.,'hi',color='green',zorder=1)

    # ax.scatter(xi_pts[0,0],yi_pts[0,0],zi_pts[0,0], marker = "o", facecolors='none',edgecolors='g',
    #             linewidth=3,s=15)
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(plot_title)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10


    # Add a color bar which maps values to colors.
    cbar=fig.colorbar(surf, shrink=0.7, aspect=15)
    #cbar = fig.colorbar(surf)
    cbar.ax.set_ylabel(contour_label2)
    #plt.clim(minn,maxx)
    #plt.clabel(CS2, fmt='%2.2f', colors='k', fontsize=8)
    #fig.colorbar(surf)

    ax.dist = 10

    #plt.show()
    #fig.canvas.show()
    #plt.show()
    #fig.canvas.draw()
    #plt.show(block=False)

    #dirName = rootdir+'it%s' % (mm)  
    #fileName = 'COND%s_3dsurf_of_%s_on_%s_%s_%s.png' % (kk,response_labels[iShow],factor_labels[xcol],factor_labels[ycol],factor_labels[zcol])
    fileName = 'responses_3dsurf_of_%s_on_%s_%s_%s.png' % (response_labels[iShow],factor_labels[xcol],factor_labels[ycol],factor_labels[zcol])
    #print(dirName+'/'+fileName)
    plt.savefig(dirName+fileName,bbox_inches = 'tight') #dpi=100
    plt.show()
    plt.close(fig)
    return ()