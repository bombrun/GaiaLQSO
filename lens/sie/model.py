import numpy as np
from scipy import optimize

"""
Alex Bombrun 
an implementation of SIE model as introduced in Kormann, Schneider & Bartelmann (1994)

f :  is the SIE lens axis ratio parameter in [0,1]
r,phi : polar coordinates of the images, r is scaled by the Einstein's radius

"""

def fRatio(f):
    """
    f : SIE parameter
    """
    return np.sqrt(f)/np.sqrt(1-f**2)

def kappa(r,phi,f):
    """
    SIE dimensionless surface mass density
    r,phi : angular polar coordinate of the image
    f : SIE lens parameter
    """
    N = np.sqrt(f)
    D = 2*r*np.sqrt(np.power(np.cos(phi),2)+np.power(f,2)*np.power(np.cos(phi),2))
    return N/D

def magnification(r,phi,f):
    """
    SIE magnification at
    r, phi : angular polar coordinate of the image 
    f : SIE lens parameter
    """
    return 1/(1-2*kappa(r,phi,f))

def A(r,phi,f):
    """
    SIE distortion matrix
    x, phi : angular polar coordinate of the image 
    f : SIE lens parameter
    """
    A11 = 1-2*kappa(r,phi,f)*np.power(np.sin(phi),2)
    A12 = kappa(r,phi,f)*np.sin(2*phi)
    A22 = 1-2*kappa(r,phi,f)*np.power(np.cos(phi),2)
    return np.array([[A11,A12],[A12,A22]])

def alpha(phi,f):
    """SIE deflection angle"""
    return -fRatio(f)*np.array([np.arcsinh(np.sqrt(1-f*f)*np.cos(phi)/f),np.arcsin(np.sqrt(1-f*f)*np.sin(phi))])

def cut(phi,f) :
    """
    limit of the lens equation when x tends to zero
    """
    return -alpha(phi,f)

def caustic(phi,f) :
    """
    the points where the distortion matrix is singular
    """
    DeltaPhi = np.sqrt(np.power(np.cos(phi),2)+np.power(f*np.sin(phi),2))
    v1 =  np.array([np.sqrt(f)*np.cos(phi),np.sqrt(f)*np.sin(phi)])/DeltaPhi
    v2 = alpha(phi,f)
    return v1+v2

def psiTilde(phi,f) : 
    return fRatio(f)*(
    np.cos(phi)*np.arcsinh(np.sqrt(1-f*f)*np.cos(phi)/f)+
    np.sin(phi)*np.arcsin(np.sin(phi)*np.sqrt(1-f*f))   
    )

def radius(phi,f,y1,y2) :
    """ 
    SIE lens equation x as a function of phi
    phi : polar angle of lens image
    f : SIE lens parameter
    y1,y2 : source location
    return : x as a function of phi
    """
    return y1*np.cos(phi)+y2*np.sin(phi)+psiTilde(phi,f)

def eq2(phi,f,y1,y2) :
    """
    SIE lens equation in phi
    phi : polar angle of lens image
    f : SIE lens parameter
    y1,y2 : source location
    """
    eqA = (y1+fRatio(f)*np.arcsinh(np.sqrt(1-f*f)*np.cos(phi)/f))*np.sin(phi)
    eqB =-(y2+fRatio(f)*np.arcsin(np.sin(phi)*np.sqrt(1-f*f)))*np.cos(phi)
    return eqA+eqB

def solve(f,y1,y2) : 
    """ solve SIE lens equation with
    f : axis ratio
    y1,y2 : relative source position with respect to the lens
    return : phi,x image position in polar coordinate as arrays of length 2 or 4
    """
    eq =  lambda phi : eq2(phi,f,y1,y2)
    step = 0.1
    phiTest = np.arange(0,2*np.pi+step,step)
    test =  eq(phiTest)>0
    phiI = []
    for phi0 in phiTest[np.where(test[:-1] != test[1:])]:
        root = optimize.brentq(eq,phi0,phi0+step)
        phiI.append(root%(2*np.pi))
    phiI = np.array(phiI)
    rI = radius(phiI,f,y1,y2)
    return rI,phiI