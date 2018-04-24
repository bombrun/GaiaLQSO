"""
QSO position and magnitude SIE lens inference 
from LQSO images position and magnitude
"""


import lens.sie.model as sie 
import numpy as np
from scipy.stats import beta, uniform, norm, gamma, cauchy, multivariate_normal

def radiusPrior(b):
    """SIE Einstein radius prior in as"""
    rlen=1
    return (1/(2*rlen**3))*(b**2)*np.exp(-b/rlen) if (b>0) else 0
    #return beta.pdf(b/5,2,3)

def ratioPrior(q):
    """SIE axis ratio prior in [0,1]"""
    return uniform.pdf(q,0,1)

def positionPrior(x):
    """SIE source and lens position prior"""
    return norm.pdf(x,0,0.1)

def magnitudePrior(x):
    """SIE source proper motion prior"""
    return gamma.pdf(x,10,5)

def thetaPrior(theta):
    """SIE lens orientation prior in [0,pi]"""
    return uniform.pdf(theta,0,np.pi)

def log_prior(model):
    """Return log10 of the priors"""
    (xs,ys,gs,b,q,xl,yl,theta) = tuple(model)
    res = np.log10(positionPrior(xs)) + np.log10(positionPrior(ys)) + np.log10(magnitudePrior(gs))
    res = res + np.log10(positionPrior(xl)) + np.log10(positionPrior(yl))
    res = res + np.log10(radiusPrior(b)) + np.log10(ratioPrior(q)) + np.log10(thetaPrior(theta))
    return res

def imageLikelyhood(s):
    """likely hood of one SIE image"""
    x,y,g,xe,ye,ge = tuple(s)
    cov = [[xe*xe,0,0],
           [0,ye*ye,0],
           [0,0,ge*ge]]
    return multivariate_normal(mean=[x,y,g], cov=cov) 

def getImages(model):
    (xS,yS,gS,bL,qL,xL,yL,thetaL) = tuple(model)
    rI,phiI = sie.solve(qL,xS,yS)
    magI = gS - 2.5 * np.log10(np.abs(sie.magnification(rI,phiI,qL)))
    res = []
    for r,phi,g in zip(rI,phiI,magI):
        res.append([bL*r*np.cos(phi+thetaL)+xL,bL*r*np.sin(phi+thetaL)+yL,g])
    return res
    
def log_likelihood(model,data) :
    """Return log10 (normalized) likelihood: P(3D astrometry | 3D phase space, Covariance)"""
    functions = []
    for s in data :
        functions.append(imageLikelyhood(s))
    images = getImages(model)
    
    if(len(images)==len(functions)): # probably a very bad idea
        res = []
        for point,f in zip(images,functions):
            res.append(f.logpdf(point)/np.log(10))
        return sum(res)
    else :
        return -np.inf
    
def log_posterior(model,data) :
    logprior = log_prior(model)
    res = logprior + log_likelihood(model,data) if np.isfinite(logprior) else -np.inf
    return np.array(res)
    
