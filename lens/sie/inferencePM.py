"""
QSO position, magnitude and proper motion inference and SIE lens inference 
from LQSO images position, magnitude and proper motion 
"""

from lens.sie.inference import *


def pmPrior(x):
    """SIE L-S proper motion prior"""
    return norm.pdf(x,0,0.5)



def log_prior_pm(model):
    """Return log10 of all the priors for model with proper motion
    xs,ys,dxs,dys,gs,b,q,xl,yl,theta : 
    """
    (xs,ys,dxs,dys,gs,b,q,xl,yl,theta) = tuple(model)
    res = np.log10(positionPrior(xs)) + np.log10(positionPrior(ys)) + np.log10(magnitudePrior(gs))
    res = res + np.log10(pmPrior(dxs)) + np.log10(pmPrior(dys))
    res = res + np.log10(positionPrior(xl)) + np.log10(positionPrior(yl))
    res = res + np.log10(radiusPrior(b)) + np.log10(ratioPrior(q)) + np.log10(thetaPrior(theta))
    return res

def imageLikelyhood_pm(s):
    """return the likelyhood function of the SIE image defined by data s"""
    x,y,dx,dy,g,xe,ye,dxe,dye,ge = tuple(s)
    cov = [[xe*xe,0,0,0,0],
           [0,ye*ye,0,0,0],
           [0,0,dxe*dxe,0,0],
           [0,0,0,dye*dye,0],
           [0,0,0,0,ge*ge]]
    return multivariate_normal(mean=[x,y,dx,dy,g], cov=cov) 

def getImages_pm(model):
    """return SIE images from model with proper motion"""
    (xS,yS,dxS,dyS,gS,bL,qL,xL,yL,thetaL) = tuple(model)
    phiI,rI = sie.solve(qL,xS,yS)
    
    # images magnitude    
    magI = gS - 2.5 * np.log10(np.abs(sie.magnification(rI,phiI,qL)))
    
    # images proper motion
    rot = np.array([[np.cos(thetaL),np.sin(thetaL)],[-np.sin(thetaL),np.cos(thetaL)]])
    dxI = [] 
    dyI = []
    for phi,r in zip(phiI,rI) :
        dx = np.dot(rot,np.dot(np.linalg.inv(sie.A(r,phi,qL)),np.array([dxS,dyS])))
        dxI.append(dx[0])
        dyI.append(dx[1])
    
    res = []
    for phi,r,dx,dy,g in zip(phiI,rI,dxI,dyI,magI):
        res.append([bL*r*np.cos(phi+thetaL)+xL,bL*r*np.sin(phi+thetaL)+yL,dx,dy,g])
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