import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

import lens.sie.model as sie 

def circle(r = 0.01) :
    """
    define a circle of radius r center on 0,0 in cartesian coordinate
    r : radius
    return : [x,y] an numpy array of length [2,20], 20 equidistant points on the circle
    """
    theta =  np.pi*np.arange(0,2,0.1)
    dy1 = r*np.cos(theta)
    dy2 = r*np.sin(theta)
    return np.array([dy1,dy2])

def plotLens(f,ax,color='C0'):
    """
    plot the SIE lens cut and caustic curve defined by f in the axis ax
    f :  lens parameter
    ax : matplotlib axis
    """
    theta =  np.pi*np.arange(0,2,0.001)
    xy = sie.cut(theta,f)
    ax.plot(xy[0],xy[1],color=color,label='cut f=%s'%f)
    xy = sie.caustic(theta,f)
    ax.plot(xy[0],xy[1],'--',color=color,label='caustic f=%s'%f)
    
def plotSourceImage(y1,y2,f,ax = plt.subplot(111,aspect='equal')):
    """
    plot the images of the source y1,y2 through the SIE lens defined by f in the axis ax
    y1,y2 : source position relative to the lens
    f : SIE lens parameter
    ax : matplotlib axis
    """
    xs,phis = sie.solve(f,y1,y2)
    dy =  circle(0.1)
    for phi,x in zip(phis,xs) :
        dx = np.dot(np.linalg.inv(sie.A(x,phi,f)),dy)
        x1 = x*np.cos(phi)
        x2 = x*np.sin(phi)
        ax.scatter(x1,x2,color='C0')#s = np.exp(np.abs(mag(x,phi,f)))
        ax.scatter(x1+dx[0],x2+dx[1],s=1,color='C0')
        ax.plot([x1,x1+dx[0][0]],[x2,x2+dx[1][0]],color='C0')
        ax.plot([x1,x1+dx[0][5]],[x2,x2+dx[1][5]],'--',color='C0')
    ax.scatter(y1,y2,s=np.exp(1),label="source",color='C1')
    ax.scatter(y1+dy[0],y2+dy[1],color='C1',s=1)
    ax.plot([y1,y1+dy[0][0]],[y2,y2+dy[1][0]],color='C1')
    ax.plot([y1,y1+dy[0][5]],[y2,y2+dy[1][5]],'--',color='C1')
    ax.grid()   
    
def plotLensSourceImage(f,y1,y2):
    """
    plot SIE lens and images
    """
    ax = plt.subplot(221,aspect='equal')
    plotLens(f,ax)
    ax.scatter(y1,y2,label="source",color='C1')
    plt.grid()
    plt.legend()
    ax = plt.subplot(222,aspect='equal')
    plotSourceImage(y1,y2,f,ax=ax)