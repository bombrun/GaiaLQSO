import numpy as np
import pandas as pd
import astropy.units as u
import healpy as hp

from lens.sie.plot import *

def angle2pixel(ra_deg,dec_deg):
    """ return healpix index 12"""
    phi = ra_deg * np.pi / 180
    theta = np.pi/2 - (dec_deg * np.pi/180)
    return hp.ang2pix(4096,theta,phi,nest=True)


def lensedQSO(f,scale,w,y,dy,gy):
    """ to generate a lensed QSO
    f : SIE lens eliptisicity parameter
    scale : a scale parameter (TODO link it to some physical parameter of the lens)
    w : lens orientation
    y : source position relative to the lens
    dy : source proper motion relative to the lens
    gy : source magnitude (assume that the magnitude is defined as 2.5 log10(flux))
    """
    # locations of lens images in the source plane
    xs,phis = sie.solve(f,y[0],y[1])
    
    # compute images position proper motion and magnitude 
    ra = []
    dec = []
    pmra = []
    pmdec = []
    g = []
    R = np.array([[np.cos(w),np.sin(w)],[-np.sin(w),np.cos(w)]])
    for phi,x in zip(phis,xs) :
        dx = np.dot(R,np.dot(np.linalg.inv(sie.A(x,phi,f)),dy))
        ra.append(x*np.cos(phi+w)*scale)
        dec.append(x*np.sin(phi+w)*scale)
        pmra.append(dx[0]*scale)
        pmdec.append(dx[1]*scale)
        g.append(gy-2.5*np.log10(np.abs(sie.magnification(x,phi,f))))
    
    # set a pandas data frame to store the result
    res = pd.DataFrame()
    res['ra'] = ra
    res['dec'] = dec
    res['pmra'] = pmra
    res['pmdec'] = pmdec
    res['phot_g_mean_mag'] = g
    return res

def getSourceId(ra_rad,dec_rad):
    x = np.asarray(ra_rad)
    y = np.asarray(dec_rad)
    s=34359738368
    sourceid = angle2pixel(x*u.rad.to(u.deg),y*u.rad.to(u.deg))*s
    if x.size==1 :
        return sourceid + np.int64(np.random.uniform(0,s))
    else :
        return sourceid + np.int64(np.random.uniform(0,s,x.size))

def randomLQSO(verbose=False):
    """ a dummy random lensed QSO generator """
    
    #scale 
    scale = np.random.uniform(1,2)
    
    # lens parameter
    f = np.random.uniform()
    
    # relative source-lens position
    y = np.random.uniform(-0.5,0.5,2)
    
    # relative source-lens proper motion
    dy =  np.random.normal(0,0.1,2)
   
    # source magnitude
    gy =  np.random.uniform(18,20)
    
    # random lens orientation
    w =  np.random.uniform(0,2*np.pi)
   
    # wrap the data
    data = f,scale,w,y,dy,gy
    
    # to visualise the lens
    if verbose :
        print(data)
        plotLensSourceImage(f,y[0],y[1])
    
    res = lensedQSO(*data)
    
    # sky location
    ra =  np.random.uniform(0,2*np.pi)
    dec = np.random.uniform(-np.pi/2+0.1,np.pi/2-0.1) # a bit wrong as we exclude the pole
    while(np.abs(dec) < 10*u.deg.to(u.rad)) :
        dec = np.random.uniform(-np.pi/2+0.1,np.pi/2-0.1) # a bit wrong as we exclude the pole
    res['ra'] = ra + res.ra*u.arcsecond.to(u.rad)
    res['dec'] = dec + res.dec*u.arcsecond.to(u.rad)
    res['source_id'] = getSourceId(res.ra,res.dec)
    res.index=res.source_id
    res['qsoid'] = res.phot_g_mean_mag.idxmin()
    return res

def generateLQSO(n):
    """return n random QSO in a pandas DataFrame"""
    res = [randomLQSO() for i in range(0,n)]
    return pd.concat(res)