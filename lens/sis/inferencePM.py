"""
QSO position, magnitude and proper motion inference and SIS lens inference 
from LQSO images position, magnitude and proper motion

model : xS,yS,dxS,dyS,gS,bL,xL,yL
"""

from lens.sis.inference import *


def pmPrior(x):
    """SIS L-S proper motion prior"""
    return norm.pdf(x,0,0.5)



def log_prior_pm(model):
    """Return log10 of all the priors for model with proper motion
    xS,yS,dxS,dyS,gS,bL,xL,yL : 
    """
    (xS,yS,dxS,dyS,gS,bL,xL,yL) = tuple(model)
    res = np.log10(positionPrior(xS)) + np.log10(positionPrior(yS)) + np.log10(magnitudePrior(gS))
    res = res + np.log10(pmPrior(dxS)) + np.log10(pmPrior(dyS))
    res = res + np.log10(positionPrior(xL)) + np.log10(positionPrior(yL))
    res = res + np.log10(radiusPrior(bL))
    return res

def imageLikelyhood_pm(s):
    """return the likelyhood function of the SIS image defined by data s"""
    x,y,dx,dy,g,xe,ye,dxe,dye,ge = tuple(s)
    cov = [[xe*xe,0,0,0,0],
           [0,ye*ye,0,0,0],
           [0,0,dxe*dxe,0,0],
           [0,0,0,dye*dye,0],
           [0,0,0,0,ge*ge]]
    return multivariate_normal(mean=[x,y,dx,dy,g], cov=cov) 

def getImages_pm(model):
    """return SIS images from model with proper motion"""
    (xS,yS,dxS,dyS,gS,bL,xL,yL) = tuple(model)
    phiI,rI = sis.solve(xS,yS)
    
    # images magnitude    
    magI = gS - 2.5 * np.log10(np.abs(sis.magnification(rI,phiI)))
    
    # images proper motion
    dxI = [] 
    dyI = []
    for phi,r in zip(phiI,rI) :
        dx = np.dot(np.linalg.inv(sis.A(r,phi)),np.array([dxS,dyS]))
        dxI.append(dx[0])
        dyI.append(dx[1])
    
    res = []
    for phi,r,dx,dy,g in zip(phiI,rI,dxI,dyI,magI):
        res.append([bL*r*np.cos(phi)+xL,bL*r*np.sin(phi)+yL,dx,dy,g])
    return res
    
def log_likelihood_pm(model,data) :
    """return log10 (normalized) likelihood for model and data with proper motion"""
    functions = []
    for s in data :
        functions.append(imageLikelyhood_pm(s))
    images = getImages_pm(model)
    
    if(len(images)==len(functions)): # probably a very bad idea
        res = []
        for point,f in zip(images,functions):
            res.append(f.logpdf(point)/np.log(10))
        return sum(res)
    else :
        return -np.inf

def log_posterior_pm(model,data) :
    """return the log 10 posterior prior for model and data with proper motion"""
    logprior = log_prior_pm(model)
    res = logprior + log_likelihood_pm(model,data) if np.isfinite(logprior) else -np.inf
    return np.array(res)