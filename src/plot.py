# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:56 2026

@author: chris
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.visualization as vis
from astropy.nddata import Cutout2D
from typing import Sequence, Any
    
def imshow_on_ax(ax, data, interval=None, pos=None, size=None, title=None,
                 stretch=vis.AsinhStretch(a=0.05)):
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
                              stretch=stretch)

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
    
def imshow(data, **imshow_on_ax_kwargs):
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
    im = imshow_on_ax(ax, data, **imshow_on_ax_kwargs)
    im_ratio = data.shape[0]/data.shape[1]
    fig.colorbar(im, fraction=0.047*im_ratio)
    plt.show()

def imshow_coords(data, wcs_object, stars=None, **imshow_on_ax_kwargs):
    _, ax = plt.subplots(subplot_kw=dict(projection=wcs_object))
    ax.grid(color="white", ls="dashed", alpha=0.3)
    ax.set(xlabel='Longitude', ylabel='Latitude')
    imshow_on_ax(ax, data)
    if stars is not None:
        ax.scatter(stars['xcentroid'], stars['ycentroid'], s=50, edgecolors='r', 
                   facecolors='none', label='Detected')
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
    _, ax = plt.subplots()
    hist_on_ax(ax, data, **kwargs)
    plt.show()
    
def subplots(nx, ny, funcs, plots, title=None, figsize=(4, 3), resolution=2, 
             add_colorbar=True, fname=None, showPlot = False):
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
        print(f"Saved plot to: {fname}")
    if showPlot:
        plt.show()
    else:
        plt.close(fig)

def plot_on_ax(ax, 
               data: list[Sequence[float]], 
               title: str = None,
               xlabel: str = None,
               ylabel: str = None,
               legend: list[str | None] = [None] * 5, 
               fname: str = None,
               # --- Advanced options ---
               # text
               text: Sequence[Any] | Sequence[Sequence[Any]] = None,
               # Formatting
               marker: list[str] = ['None'],
               linestyle: list[str] = ['-'],
               ncolors: int = None,
               # Ticks and Grid
               xdate: dict = None,
               xticks: int | Sequence[float] = None,
               yticks: int | Sequence[float] = None,
               xgrid: bool = False,
               ygrid: bool = False,
               # Sizes
               titleSize: int = 12,
               labelSize: int = 10,
               scale: Sequence[str] = ('linear', 'linear'),
               legendLoc: int = 0,
               # errors
               xerr: Sequence[Sequence[float] | None] = (None,),
               yerr: Sequence[Sequence[float] | None] = (None,),
               fillErr: Sequence[bool] = (False,),
               errlabel: Sequence[str | None] = (None,)
               ) -> None:
    '''
    Plot one or multiple data series.
    
    Core Parameters
    ---------------
    data : list of sequence of float
        List of data arrays. The first element is used as the x-axis values.
        All subsequent elements are plotted against the first element.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str, optional
        Label of the y-axis.
    legend : list of str or None, optional
        Labels for the plotted data series. Length should match ``len(data) - 1``.
        Entries may be ``None`` to omit individual legend items.
    fname : str, optional
        If provided, the plot is saved to this file name.
    
    Advanced Parameters
    -------------------
    text : sequence or sequence of sequences, optional
        Text annotations. Either a single entry
        ``[x_pos, y_pos, text]`` or a list of such entries.
    
    Formatting
    ----------
    marker : list of str, optional
        Marker styles for the plotted data series.
    linestyle : list of str, optional
        Line styles for the plotted data series.
    ncolors : int, optional
        Number of colors to cycle through before repeating.
    
    Ticks and Grid
    --------------
    xdate : dict, optional
        Dictionary defining date formatting for the x-axis.
        Example: ``{"format": "%m-%d", "type": "day", "locator": {"interval": 4}}``.
    xticks : int or sequence of float, optional
        Tick positions for the x-axis. If a sequence is provided, the values are
        used directly. If an integer is provided, it specifies the maximum number
        of major ticks.
    yticks : int or sequence of float, optional
        Tick positions for the y-axis. If a sequence is provided, the values are
        used directly. If an integer is provided, it specifies the maximum number
        of major ticks.
    xgrid : bool, optional
        Enable grid lines along the x-axis.
    ygrid : bool, optional
        Enable grid lines along the y-axis.
    
    Sizes and Scaling
    -----------------
    titleSize : int, optional
        Font size of the plot title.
    labelSize : int, optional
        Font size of the axis labels.
    scale : sequence of str, optional
        Axis scaling for x and y axes, e.g. ``('linear', 'log')``.
    figsize : tuple of float, optional
        Size of the figure in inches.
    dpi : int, optional
        Dots per inch of the figure.
    legendLoc : int, optional
        Location code for the legend (matplotlib convention).
    resolution : int, optional
        Resolution multiplier applied when saving the figure.

    Returns
    -------
    None
        The plot is displayed and optionally saved to disk.
    '''
    
    plt.xscale(scale[0])
    plt.yscale(scale[1])
    ax.set_title(
        title, fontsize=titleSize, fontname='serif')  # 'Messung 1'
    ax.set_xlabel(
        xlabel, fontsize=labelSize, fontname='serif')
    ax.set_ylabel(
        ylabel, fontsize=labelSize, fontname='serif')
    if xticks is not None:
        ax.xticks(xticks)  # plt.
    if yticks is not None:
        ax.yticks(yticks)
    if len(data)-1 != len(legend):
        i = 0
        while i < len(data)-2:
            legend.append(None)
            i += 1
        i = None
    if len(marker) != len(legend):
        marker = marker*int(len(legend)/len(marker))
    if len(linestyle) != len(legend):
        linestyle = linestyle * \
            int(len(legend)/len(linestyle))

    for i in range(len(data)-1):
        ax.plot(data[0], data[i+1], marker=marker[i],  # plt.
                 linestyle=linestyle[i], label=legend[i])  # plotting data

    if xdate is not None:
        if "format" in xdate:
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(xdate["format"]))
        if "locator" in xdate:
            loc_kwargs = xdate["locator"]
            if xdate["type"] == "day":
                ax.xaxis.set_major_locator(mpl.dates.DayLocator(**loc_kwargs))
            if xdate["type"] == "month":
                ax.xaxis.set_major_locator(mpl.dates.MonthLocator(**loc_kwargs))
            if xdate["type"] == "year":
                loc_kwargs = {"base": xdate["locator"]["interval"]}
                ax.xaxis.set_major_locator(mpl.mdates.YearLocator(**loc_kwargs))
        plt.gcf().autofmt_xdate()
    if text != None:
        try:
            ax.figtext(float(text[0]), float(text[1]), text[2])  # plt.
        except TypeError:  # more than one text
            for line in text:
                ax.figtext(line[0], line[1], line[2])  # plt.
    if xgrid is True and ygrid is True:
        ax.grid()  # plt.
    elif xgrid is True:
        ax.grid(axis='x')  # plt.
    elif ygrid is True:
        ax.grid(axis='y')  # plt.
    # legend contains only None
    if not legend.count(None) == len(legend):
        ax.legend(loc=legendLoc)  # plt.
    
def plot(figsize: tuple[float, float] = (8, 6),  # (4,3) for large labels, (8,4.5) instead of (16,9)
         dpi: int = 100, 
         resolution: int = 2,
         fname: str = None, 
         **plot_kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plot_on_ax(ax, **plot_kwargs)
    if fname != None:
        fig.savefig(fname, dpi=dpi*resolution, bbox_inches='tight')
    plt.show()
    
# def plot(data: list[Sequence[float]], 
#          title: str = None,
#          xlabel: str = None,
#          ylabel: str = None,
#          legend: list[str | None] = [None] * 5, 
#          fname: str = None,
#          # --- Advanced options ---
#          # text
#          text: Sequence[Any] | Sequence[Sequence[Any]] = None,
#          # Formatting
#          marker: list[str] = ['None'],
#          linestyle: list[str] = ['-'],
#          ncolors: int = None,
#          # Ticks and Grid
#          xdate: dict = None,
#          xticks: int | Sequence[float] = None,
#          yticks: int | Sequence[float] = None,
#          xgrid: bool = False,
#          ygrid: bool = False,
#          # Sizes
#          titleSize: int = 12,
#          labelSize: int = 10,
#          scale: Sequence[str] = ('linear', 'linear'),
#          figsize: tuple[float, float] = (8, 6),  # (4,3) for large labels
#                                                  # (8,4.5) instead of (16,9)
#          dpi: int = 100,
#          legendLoc: int = 0,
#          resolution: int = 2,
#          # errors
#          xerr: Sequence[Sequence[float] | None] = (None,),
#          yerr: Sequence[Sequence[float] | None] = (None,),
#          fillErr: Sequence[bool] = (False,),
#          errlabel: Sequence[str | None] = (None,)
#          ) -> None:
#     '''
#     Plot one or multiple data series.
    
#     Core Parameters
#     ---------------
#     data : list of sequence of float
#         List of data arrays. The first element is used as the x-axis values.
#         All subsequent elements are plotted against the first element.
#     title : str, optional
#         Title of the plot.
#     xlabel : str, optional
#         Label of the x-axis.
#     ylabel : str, optional
#         Label of the y-axis.
#     legend : list of str or None, optional
#         Labels for the plotted data series. Length should match ``len(data) - 1``.
#         Entries may be ``None`` to omit individual legend items.
#     fname : str, optional
#         If provided, the plot is saved to this file name.
    
#     Advanced Parameters
#     -------------------
#     text : sequence or sequence of sequences, optional
#         Text annotations. Either a single entry
#         ``[x_pos, y_pos, text]`` or a list of such entries.
    
#     Formatting
#     ----------
#     marker : list of str, optional
#         Marker styles for the plotted data series.
#     linestyle : list of str, optional
#         Line styles for the plotted data series.
#     ncolors : int, optional
#         Number of colors to cycle through before repeating.
    
#     Ticks and Grid
#     --------------
#     xdate : dict, optional
#         Dictionary defining date formatting for the x-axis.
#         Example: ``{"format": "%m-%d", "type": "day", "locator": {"interval": 4}}``.
#     xticks : int or sequence of float, optional
#         Tick positions for the x-axis. If a sequence is provided, the values are
#         used directly. If an integer is provided, it specifies the maximum number
#         of major ticks.
#     yticks : int or sequence of float, optional
#         Tick positions for the y-axis. If a sequence is provided, the values are
#         used directly. If an integer is provided, it specifies the maximum number
#         of major ticks.
#     xgrid : bool, optional
#         Enable grid lines along the x-axis.
#     ygrid : bool, optional
#         Enable grid lines along the y-axis.
    
#     Sizes and Scaling
#     -----------------
#     titleSize : int, optional
#         Font size of the plot title.
#     labelSize : int, optional
#         Font size of the axis labels.
#     scale : sequence of str, optional
#         Axis scaling for x and y axes, e.g. ``('linear', 'log')``.
#     figsize : tuple of float, optional
#         Size of the figure in inches.
#     dpi : int, optional
#         Dots per inch of the figure.
#     legendLoc : int, optional
#         Location code for the legend (matplotlib convention).
#     resolution : int, optional
#         Resolution multiplier applied when saving the figure.
    
#     Errors
#     ------
#     xerr : sequence of sequence of float or None, optional
#         x-error values for error bars. One entry per data series.
#     yerr : sequence of sequence of float or None, optional
#         y-error values for error bars. One entry per data series.
#     fillErr : sequence of bool, optional
#         If True, error regions are drawn using ``fill_between`` instead of error bars.
#     errlabel : sequence of str or None, optional
#         Labels for error bars or filled error regions.
    
#     Returns
#     -------
#     None
#         The plot is displayed and optionally saved to disk.
#     '''
    
#     _, ax = plt.subplots(figsize=figsize, dpi=dpi)  # figsize=(4, 3), dpi=100
#     plt.xscale(scale[0])
#     plt.yscale(scale[1])
#     ax.set_title(
#         title, fontsize=titleSize, fontname='serif')  # 'Messung 1'
#     ax.set_xlabel(
#         xlabel, fontsize=labelSize, fontname='serif')
#     ax.set_ylabel(
#         ylabel, fontsize=labelSize, fontname='serif')
#     if xticks is not None:
#         plt.xticks(xticks)
#     if yticks is not None:
#         plt.yticks(yticks)
#     if len(data)-1 != len(legend):
#         i = 0
#         while i < len(data)-2:
#             legend.append(None)
#             i += 1
#         i = None
#     if len(marker) != len(legend):
#         marker = marker*int(len(legend)/len(marker))
#     if len(linestyle) != len(legend):
#         linestyle = linestyle * \
#             int(len(legend)/len(linestyle))
#     if len(fillErr) != len(legend):
#         fillErr = fillErr*int(len(legend)/len(fillErr))
#     for i in range(len(data)-1):
#         plt.plot(data[0], data[i+1], marker=marker[i],
#                  linestyle=linestyle[i], label=legend[i])  # plotting data
#         if fillErr[i]:
#             try:
#                 plt.fill_between(
#                     data[0], data[i+1]-yerr[i], data[i+1]+yerr[i], label=errlabel[i])
#             except:
#                 pass
#         else:
#             try:
#                 plt.errorbar(data[0], data[i+1], yerr=yerr[i],
#                              xerr=xerr[i], label=errlabel[i], ls='none')  # 1
#             except:
#                 try:
#                     plt.errorbar(
#                         data[0], data[i+1], yerr=yerr[i], label=errlabel[i], ls='None')
#                 except:
#                     try:
#                         plt.errorbar(
#                             data[0], data[i+1], xerr=xerr[i], label=errlabel[i], ls='None')
#                     except:
#                         pass
#     if xdate is not None:
#         if "format" in xdate:
#             ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(xdate["format"]))
#         if "locator" in xdate:
#             loc_kwargs = xdate["locator"]
#             if xdate["type"] == "day":
#                 ax.xaxis.set_major_locator(mpl.dates.DayLocator(**loc_kwargs))
#             if xdate["type"] == "month":
#                 ax.xaxis.set_major_locator(mpl.dates.MonthLocator(**loc_kwargs))
#             if xdate["type"] == "year":
#                 loc_kwargs = {"base": xdate["locator"]["interval"]}
#                 ax.xaxis.set_major_locator(mpl.mdates.YearLocator(**loc_kwargs))
#         plt.gcf().autofmt_xdate()
#     if text != None:
#         try:
#             plt.figtext(
#                 float(text[0]), float(text[1]), text[2])
#         except TypeError:  # more than one text
#             for line in text:
#                 plt.figtext(line[0], line[1], line[2])
#     if xgrid is True and ygrid is True:
#         plt.grid()
#     elif xgrid is True:
#         plt.grid(axis='x')
#     elif ygrid is True:
#         plt.grid(axis='y')
#     # legend contains only None
#     if not legend.count(None) == len(legend):
#         # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#         #            fancybox=True, shadow=True, ncol=2, prop={'size': 8})
#         plt.legend(loc=legendLoc)
#         # loc=0-best, loc=1-rechts oben, loc=2-links oben, loc=3-links unten,loc=4-rechts unten,loc=5-rechts,loc=6-mitte links,loc=7-mitte rechts,loc=8-unten mitte,loc=9-oben mitte,loc=10-mitte
#     if fname != None:
#         plt.savefig(fname, dpi=dpi*resolution, bbox_inches='tight')
#     plt.show()


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
    
def plot_stars(im, stars, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(stars['xcentroid'], stars['ycentroid'], s=50, edgecolors='r', 
               facecolors='none', label='Detected')
    imshow_on_ax(ax, im)
    fig.legend()
    ax.set_title(title)
    plt.show()