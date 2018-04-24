"""
to plot healpix maps using data from a pandas DataFrame with at leat one column source_id that follows Gaia data model specification
"""

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp

class gaiapix :
    nnn=34359738368
    
    def __init__(self,i) :
        healpix_expression='source_id/34359738368'
        healpix_max_level=12
        self.healpix_level=i
        reduce_level = healpix_max_level - self.healpix_level
        self.NSIDE = 2**self.healpix_level
        self.scaling = 4**reduce_level
        self.s = 34359738368*self.scaling 
        self.expr = "%s/%s" % (healpix_expression, self.scaling)
        self.shape = hp.nside2npix(self.NSIDE)
        self.values= np.zeros(self.shape)
        
        
    def setValues(self,p_df,sourceId='source_id',keyValue='val',mode='median'):
        """
        computes and set the healpix value to the median per pixel
        p_df : a pandas data frame
        sourceId : source index encoding healpix index
        keyValue : values column name
        """
        if mode=='median' :
            g = p_df.groupby(np.int32(p_df[sourceId]/self.s))[keyValue].median()
        else :
            g = p_df.groupby(np.int32(p_df[sourceId]/self.s))[keyValue].median()
        values= np.zeros(self.shape)
        for i,v in zip(g.index,g.values):
            values[i]=v
        
        self.values = hp.ma(values,badval=0)
        
            
    def setHpValues(self,p_df,hp='hp',keyValue='val',grp=True):
        """
        computes and set the healpix value to the median per pixel
        p_df : a pandas data frame
        sourceId : source index encoding healpix index
        keyValue : values column name
        """
        if grp :
            g = p_df.groupby(np.int32(p_df[hp]))[keyValue].median()
            self.values= np.zeros(self.shape)
            for i,v in zip(g.index,g.values):
                self.values[i]=v
        else :
            self.values= np.zeros(self.shape)
            for i,v in zip(p_df[hp].values,p_df[keyValue].values) :
                self.values[i]=v
          
            
    def setCount(self,p_df,sourceId='source_id'):
        """
        computes and set the healpix value to the median per pixel
        p_df : a pandas data frame
        sourceId : source index encoding healpix index
        keyValue : values column name
        """
        g = p_df.groupby(np.int32(p_df[sourceId]/self.s))[sourceId].count()
        
        self.values= np.zeros(self.shape)
        for i,v in zip(g.index,g.values):
            self.values[i]=v
            
    def setHpCount(self,p_df,hp='hp'):
        """
        p_df : a pandas data frame
        hp : healpix index (nested) 
        """
        g = p_df.groupby(np.int32(p_df[hp]))[hp].count()
        
        self.values= np.zeros(self.shape)
        for i,v in zip(g.index,g.values):
            self.values[i]=v
            
    def plot(self,title='',unit='',coord='C', sub=None,vmin=-100,vmax=100,cmap=plt.cm.bwr,norm=None):
        """
        moll view plot
        """
        m2 = hp.reorder(self.values, inp="NEST", out="RING")
        cmap.set_under("w")
        hp.mollview(m2,coord=['C', coord],
                    fig=1,
                    title=title,
                    unit=unit+" [hp%s]"%self.healpix_level,
                    cmap=cmap,
                    sub=sub,
                    min=vmin,max=vmax,
                   norm=norm) 
        
    def gethpNeighbours(self,i):
        theta, phi = hp.pix2ang(self.NSIDE,i)
        return hp.pixelfunc.get_all_neighbours(self.NSIDE,theta,phi,nest=True)
    
    def getRot(self,i):
        """
        get the rotation to the healpix center
        i : the heapix index that defines the center of the rotation
        """
        theta, phi = hp.pix2ang(self.NSIDE,i,nest=True)
        ra_deg = phi / np.pi *180
        dec_deg = (np.pi/2 - theta) / np.pi *180
        return [ra_deg,dec_deg,0.0]
    
    def query_disc(self,ra,dec,r):
        phi = ra
        theta = np.pi/2-dec
        iL = hp.query_disc(self.NSIDE,hp.ang2vec(theta,phi),r,nest=True)
        thetaL, phiL = hp.pix2ang(hpE.NSIDE,iL,nest=True)
        x = iL,phiL-ra,np.pi/2-thetaL-dec,self.values[iL]
        return pd.DataFrame((np.array(x)).transpose()
             ,columns=['hp','r','d','n'])
    
    def zoom(self,rot,f,
             extent = (1,10,1,10),
             xsize=1000,ysize=1000,
             vmin=0,vmax=10,
             cmap=mp.cm.gnuplot):
        m2=hp.reorder(self.values, inp="NEST", out="RING")
        g_ax=hp.zoomtool.PA.HpxGnomonicAxes(f,extent)
        f.add_axes(g_ax)
        g_ax.projmap(m2,rot=rot,xsize=xsize,ysize=ysize,vmin=vmin,vmax=vmax,cmap=cmap)
        g_ax.graticule()
        
    def pixel2angle(self,i):
        theta, phi = hp.pix2ang(self.NSIDE,i,nest=True)
        ra_deg = phi / np.pi *180
        dec_deg = (np.pi/2 - theta) / np.pi *180
        return ra_deg,dec_deg
    
    def angle2pixel(self,ra_deg,dec_deg):
        phi = ra_deg * np.pi / 180
        theta = np.pi/2 - (dec_deg * np.pi/180)
        return hp.ang2pix(self.NSIDE,theta,phi,nest=True)