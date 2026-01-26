# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:56 2026

@author: chris
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.visualization as vis
from astropy.nddata import Cutout2D

    
def imshow_on_ax(ax, data, interval=None, pos=None, size=None, title=None):
    '''
    displays image    

    Parameters
    ----------
    ax : axes._axes.Axes
        ax to plot on
    data : np.ndarray
        data to plot.
    interval : tuple, optional
        (min_val, max_val) of colorbar. The default is None.
    pos : tuple, optional
        (x_center, y_center). The default is None.
    size : tuple, optional
        (nx, ny). Has to be set if pos is set. The default is None.
    title : str, optional
        title of plot. The default is None.

    Returns
    -------
    None.

    '''
    if interval == None:
        interval = vis.MinMaxInterval()
    else:
        interval = vis.ManualInterval(interval[0], interval[1])
    norm = vis.ImageNormalize(data, interval=interval,
                              stretch=vis.SqrtStretch())

    if title is not None:
        ax.set_title(title)
    if pos is not None and size is not None:
        cut = Cutout2D(data, position=pos, size=(size[1], size[0]))
        bbox = cut.bbox_original
        extent = (bbox[1][0], bbox[1][1], bbox[0][0], bbox[0][1])
        im = ax.imshow(cut.data, origin="lower", cmap="gray", norm=norm, extent=extent)
    else:
        im = ax.imshow(data, origin="lower", cmap="gray", norm=norm, )
    return im
    
def imshow(data, **kwargs):
    '''
    displays image data

    Parameters
    ----------
    data : np.ndarray
        image.
    interval : tuple, optional
        (min_val, max_val) of colorbar. The default is None.
    pos : tuple, optional
        (x_center, y_center). The default is None.
    size : tuple, optional
        (nx, ny). Has to be set if pos is set. The default is None.
    title : str, optional
        title of plot. The default is None.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots()
    im = imshow_on_ax(ax, data, **kwargs)
    im_ratio = data.shape[0]/data.shape[1]
    fig.colorbar(im, fraction=0.047*im_ratio)
    plt.show()

def hist_on_ax(ax, data, bins=1000, survival=True, title=None):
    '''
    Plots survival function or histogram on given axis

    Parameters
    ----------
    ax : axes._axes.Axes
        ax to plot on.
    data : np.ndarray
        data to plot.
    bins : int, str, optional
        number of bins if int or one of ("blocks", "knuth", "scott", "freedman"). 
        rules via strings can be extremely slow, maybe for final plots
        The default is 1000.
    survival : bool, optional
        Don't plot hist but survival function. Better imo. The default is True.
    title : str, optional
        title of plot. The default is None.

    Returns
    -------
    None.

    '''
    data1d = data.flatten()
    data1d = data1d[np.isfinite(data1d)]
    if survival is True:
        vals = np.sort(data1d)
        ccdf = 1 - np.arange(1, len(vals)+1)/len(vals)
        ax.plot(vals, ccdf)
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Fraction of pixels > Pixel value")
    else:
        vis.hist(data1d, bins=bins, ax=ax)
    ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)

def hist(data, **kwargs):
    '''
    Plots survival function or histogram on given axis

    Parameters
    ----------
    ax : axes._axes.Axes
        ax to plot on.
    data : np.ndarray
        data to plot.
    bins : int, str, optional
        number of bins if int or one of ("blocks", "knuth", "scott", "freedman"). 
        rules via strings can be extremely slow, maybe for final plots
        The default is 1000.
    survival : bool, optional
        Don't plot hist but survival function. Better imo. The default is True.
    title : str, optional
        title of plot. The default is None.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots()
    hist_on_ax(ax, data, **kwargs)
    plt.show()
    
def subplots(nx, ny, funcs, plots, title=None, figsize=(4, 3), resolution=2, 
             add_colorbar=True, fname=None):
    '''
    Plots nx*ny subplots, filling row after row with given func/plot pairs. 

    Parameters
    ----------
    nx : int
        number of subplots in x-direction.
    ny : int
        number of subplots in x-direction.
    funcs : list of functions
        functions that are used to plot.
    plots : list of dicts
        parameters of funcs. Has to be in same order.
    title : str, optional
        title of plot. The default is None.
    figsize : tuple, optional
        size of individual plots. The default is (4, 3).
    resolution : int, optional
        resulution of saved image. The default is 2.
    add_colorbar : bool, optional
        if True, adds colorbar to first subplot returning an image. 
        Dangerous, maybe someone has a better idea? The default is True.
    fname : str, optional
        path to saving location. Dont save if None. The default is None.

    Returns
    -------
    None.

    '''
    fig, axes = plt.subplots(ny, nx, figsize=(figsize[0]*nx, figsize[1]*ny),
                             dpi=100, squeeze=False, constrained_layout=True)
    ims = []
    for i, ax in enumerate(axes.flat):
        if i < len(plots):
            im = funcs[i](ax, **plots[i])
            ims.append(im)
        else:
            ax.axis("off")
    ims = [im for im in ims if im is not None]
    if add_colorbar is True:
        im_shape = ims[0].get_shape()
        im_ratio = im_shape[0]/im_shape[1]
        fig.colorbar(ims[0], fraction=0.047*im_ratio)
    if title is not None:
        plt.suptitle(title)
    if fname is not None:
        fig.savefig(fname, dpi=100*resolution, bbox_inches="tight")
    # plt.tight_layout()
    plt.show()

def reduction(frame, reduced, color, positions, ysize, figdir):
    '''
    Plot cutout of original and reduced frame as well as respective histograms.
    Save plots in figdir

    Parameters
    ----------
    frame : ScienceFrame
        The science frame that is reduced.
    reduced : np.ndarray
        reduced image.
    color : str
        filter used.
    positions : dict of tuple
        centers of cutouts that are plotted, ie stellar position.
    ysize : int
        width of cutout in y direction
    figdir : str
        directory where figs are stored.
        {ID}_{color}_reduction.png
        
    Returns
    -------
    None.

    '''
    orig = frame.load()
    #reduced = reduce(frame, mbias, mdark_rate, mflat)
    interval = (reduced.min(), reduced.max())
    pos = positions[color]
    ysize = int(ysize)
    size = (ysize*4//3, ysize)
    funcs = [imshow_on_ax, imshow_on_ax,
             hist_on_ax, hist_on_ax]
    plots = [
        dict(data=orig, interval=interval, pos=pos, size=size, title="original"), 
        dict(data=reduced, interval=interval, pos=pos, size=size, title="reduced"),
        dict(data=orig, survival=True), dict(data=reduced, survival=True)]
    
    ID = frame.header.get("OBJECT", "UNKNOWN")
    os.makedirs(figdir, exist_ok=True)
    fname = f"{figdir}/{ID}_{color}_reduction.png"
    subplots(2, 2, funcs, plots, title=f"Data reduction for {color}-band",
             fname=fname)