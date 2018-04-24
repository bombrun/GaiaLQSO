"""
QSO position and magnitude SIE lens inference 
from LQSO images position and magnitude
"""


import lens.sis.model as sis 
import numpy as np
from scipy.stats import beta, uniform, norm, gamma, cauchy, multivariate_normal

def radiusPrior(b):
    """SIS Einstein radius prior in as"""
    rlen=1
    return (1/(2*rlen**3))*(b**2)*np.exp(-b/rlen) if (b>0) else 0
    #return beta.pdf(b/5,2,3)

def positionPrior(x):
    """SIS source and lens position prior"""
    return norm.pdf(x,0,0.1)

def magnitudePrior(x):
    """SIS source magnitude prior"""
    return gamma.pdf(x,10,5)


def log_prior(model):
    """Return log10 of the priors"""
    (xS,yS,gS,bL,xL,yL) = tuple(model)
    res = np.log10(positionPrior(xS)) + np.log10(positionPrior(yS)) + np.log10(magnitudePrior(gS))
    res = res + np.log10(positionPrior(xL)) + np.log10(positionPrior(yL))
    res = res + np.log10(radiusPrior(bL))
    return res

def imageLikelyhood(s):
    """likely hood of one SIE image"""
    x,y,g,xe,ye,ge = tuple(s)
    cov = [[xe*xe,0,0],
           [0,ye*ye,0],
           [0,0,ge*ge]]
    return multivariate_normal(mean=[x,y,g], cov=cov) 

def getImages(model):
    (xS,yS,gS,bL,xL,yL) = tuple(model)
    phiI,rI = sis.solve(xS,yS)
    magI = gS - 2.5 * np.log10(np.abs(sis.magnification(rI,phiI)))
    res = []
    for phi,r,g in zip(phiI,rI,magI):
        res.append([bL*r*np.cos(phi)+xL,bL*r*np.sin(phi)+yL,g])
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
    
