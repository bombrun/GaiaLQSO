import numpy as np
from scipy import optimize

"""
Alex Bombrun 
an implementation of SIS model 
r,phi : polar coordinates of the images, r is scaled by the Einstein's radius

"""


def kappa(r,phi):
    """
    SIS dimensionless surface mass density
    r,phi : angular polar coordinate of the image
    """
    return 1/(2*r)

def magnification(r,phi):
    """
    SIS magnification at
    r, phi : angular polar coordinate of the image 
    """
    return 1/(1-2*kappa(r,phi))

def A(r,phi):
    """
    SIS distortion matrix
    x, phi : angular polar coordinate of the image 
    """
    A11 = 1-2*kappa(r,phi)*np.power(np.sin(phi),2)
    A12 = kappa(r,phi)*np.sin(2*phi)
    A22 = 1-2*kappa(r,phi)*np.power(np.cos(phi),2)
    return np.array([[A11,A12],[A12,A22]])

def alpha(phi):
    """SIS deflection angle"""
    return np.array([np.cos(phi),np.sin(phi)])

def cut(phi) :
    """
    limit of the lens equation when r tends to zero
    """
    return -alpha(phi)

def caustic(phi) :
    """
    the points where the distortion matrix is singular
    """
    return [0,0]

def psiTilde(phi) : 
    return 1

def radius(phi,y1,y2) :
    """ 
    SIS lens equation x as a function of phi
    phi : polar angle of lens image
    y1,y2 : source location
    return : r as a function of phi,y1,y2
    """
    return y1*np.cos(phi)+y2*np.sin(phi)+1

def eq2(phi,y1,y2) :
    """
    SIS lens equation in phi
    phi : polar angle of lens image
    y1,y2 : source location
    """
    eqA = (y1+np.cos(phi))*np.sin(phi)
    eqB =-(y2+np.sin(phi))*np.cos(phi)
    return eqA+eqB

def solve(y1,y2) : 
    """ solve SIS lens equation with
    y1,y2 : relative source position with respect to the lens
    return : phi,x image position in polar coordinate as arrays of length 2 or 4
    """
    eq =  lambda phi : eq2(phi,y1,y2)
    step = 0.1
    phiTest = np.arange(0,2*np.pi+step,step)
    test =  eq(phiTest)>0
    phiI = []
    for phi0 in phiTest[np.where(test[:-1] != test[1:])]:
        root = optimize.brentq(eq,phi0,phi0+step)
        phiI.append(root%(2*np.pi))
    phiI = np.array(phiI)
    rI = radius(phiI,y1,y2)
    return phiI,rI