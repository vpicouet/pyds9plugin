#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:42:02 2022

@author: Vincent
"""
from matplotlib import colors as mcolors, path
from matplotlib.collections import RegularPolyCollection
import matplotlib.pyplot as plt
from matplotlib.widgets import Lasso
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

class Datum:
    colorin = mcolors.to_rgba("red")
    colorout = mcolors.to_rgba("blue")

    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include:
            self.color = self.colorin
        else:
            self.color = self.colorout


class LassoManager:
    def __init__(self, data):
        self.image=data
        self.image_masked = data
        self.y = np.median(data[0:100,:],axis=0)
        self.x = np.arange(len(self.y))
        self.axes =plt.axes(xlim=(0, 1000), ylim=(ds9[0:100,:].mean(axis=0).min(), ds9[0:100,:].mean(axis=0).max()), autoscale_on=False)# ax
        self.canvas = self.axes.figure.canvas
        self.data = [Datum(x,y) for x,y in enumerate(self.y)]

        self.Nxy = len(data)

        facecolors = [d.color for d in self.data]
        self.xys = [(d.x, d.y) for d in self.data]
        self.collection = RegularPolyCollection(
            6, sizes=(10,),
            facecolors=facecolors,
            offsets=self.xys,
            transOffset=self.axes.transData)
        self.axes.plot(np.mean(data[0:100,:],axis=0),'.')
        self.axes.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.on_press)


    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        self.ind = p.contains_points(self.xys)
        print(self.ind)
        for i in range(len(self.xys)):
            if self.ind[i]:
                facecolors[i] = Datum.colorin
            else:
                facecolors[i] = Datum.colorout

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        self.change_mask()
        del self.lasso

    def change_mask(self):
        #1 definir le haut le bas et le milieu
        #2 associé à chaque i un valeur inf et superieur
        #3 l'appliquer
        f1 = interp1d(self.x[self.ind],self.y[self.ind], kind='nearest', fill_value="extrapolate")
        self.middle_values = f1(self.x)
        self.value_max = []
        self.value_min = []
        value_max = np.max(self.image)
        value_min = np.min(self.image)
        for i in range(len(self.xys)):
            if ~self.ind[i]:
                if self.y[i]>self.middle_values[i]:
                    value_max = self.y[i]
                else:
                    value_min = self.y[i]
            self.value_max.append(value_max)
            self.value_min.append(value_min)


        self.image_masked_min = self.image#
        self.image_masked_max = self.image#[self.image<self.value_max]
        self.image_masked_both = self.image#[(self.image<self.value_max)&(self.image>self.value_min)]
        self.image_masked_min[self.image<self.value_min] = np.nan
        self.image_masked_max[self.image>self.value_max] = np.nan
        self.image_masked_both[(self.image>self.value_max)|(self.image<self.value_min)]=np.nan
        val = np.random.randint(10)
        fits.HDUList([fits.PrimaryHDU(self.image_masked_min )]).writeto('/tmp/min%s.fits'%(val),overwrite=True)
        fits.HDUList([fits.PrimaryHDU(self.image_masked_max )]).writeto('/tmp/max%s.fits'%(val),overwrite=True)
        fits.HDUList([fits.PrimaryHDU(self.image_masked_both )]).writeto('/tmp/both%s.fits'%(val),overwrite=True)

        return


    def on_press(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)


# if __name__ == '__main__':
    # from pyds9plugin.DS9Utils import *
    # d=DS9n()
    # ds9=d.get_pyfits()[0].data
    # np.random.seed(19680801)
    # data = [Datum(*xy) for xy in np.random.rand(100, 2)]
    # ax.set_title('Lasso points using left mouse button')
lman = LassoManager(ds9)
plt.title('Choose data you want to keep. Unselected data will be used to clip out image')
plt.show()
print(1,2,4)
ds9 = lman.image_masked_both
