"""This thing is taken from my other personal library"""
from typing import Any, Optional
import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, AutoLocator

import torch
import numpy as np

from IPython import display

# from ..torch_tools import seeded_randperm
from ._common import _dict_update, _axplot, _axplot_update, _aximshow, _aximshow_update, _axpath10d, _axpath10d_update, _axpath2d, _axpath2d_update, _axscatter10d, _axscatter10d_update


class LiveFigure:
    def __init__(self):
        self.loc_label: dict[tuple[int,int], list[Any]] = {}
        self.label_data: dict[Any,dict [str, Any]] = {}
        self.label_ax: dict[Any, Axes] = {}
        self.needs_init = set()
        self.fig = None
        self.ax = None

    def _add_loc(self, loc, label):
        if loc not in self.loc_label:
            self.loc_label[loc] = [label]
        else: self.loc_label[loc].append(label)
        self.needs_init.add(label)

    def add_plot(self, label, loc, linewidth=1, xlim=None,ylim=None,xlabel=None,ylabel=None,title=None, **kwargs):
        kwargs = _dict_update(kwargs, linewidth=linewidth,xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,title=title)
        self.label_data[label] = {"draw func":_axplot, "update func": _axplot_update, "label":label, "kwargs": kwargs}
        self._add_loc(loc, label)

    def add_image(self, label, loc, cmap='gray', vmin=None,vmax=None,alpha=None,xlabel=None,ylabel=None,title=None, allow_alpha=False,mode="auto",**kwargs):
        kwargs = _dict_update(kwargs,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha, xlabel=xlabel, ylabel=ylabel,title=title,allow_alpha=allow_alpha,mode=mode)
        self.label_data[label] = {"draw func":_aximshow, "update func": _aximshow_update, "label":label,  "kwargs": kwargs}
        self._add_loc(loc, label)

    def add_scatter10d(self, label, loc, alpha = 0.5,cmap='gnuplot2', xlim=None,ylim=None,xlabel=None,ylabel=None,title=None, **kwargs):
        kwargs = _dict_update(kwargs, alpha=alpha,xlim=xlim,cmap=cmap, ylim=ylim, xlabel=xlabel, ylabel=ylabel,title=title)
        self.label_data[label] = {"draw func":_axscatter10d, "update func": _axscatter10d_update, "label":label,  "kwargs": kwargs}
        self._add_loc(loc, label)

    def add_path2d(self, label, loc, linewidth=1, linecolor=None,markercolor=None,xlim=None,ylim=None,xlabel=None,ylabel=None,title=None,det=True, **kwargs):
        kwargs = _dict_update(kwargs, linewidth=linewidth,linecolor=linecolor,markercolor=markercolor,xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,title=title,det=det)
        self.label_data[label] = {"draw func":_axpath2d, "update func": _axpath2d_update, "label":label,"kwargs": kwargs}
        self._add_loc(loc, label)

    def add_path10d(self, label, loc, linewidth=1, linecolor='black',xlim=None,ylim=None,xlabel=None,ylabel=None,title=None, lastn=None, **kwargs):
        kwargs = _dict_update(kwargs, linewidth=linewidth,linecolor=linecolor,xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,title=title,lastn=lastn)
        self.label_data[label] = {"draw func":_axpath10d, "update func": _axpath10d_update, "label":label,"kwargs": kwargs}
        self._add_loc(loc, label)


    def draw(self, force_redraw = False, figsize = None, layout='tight', update=True, **kwargs):
        # create figure for the first time
        if force_redraw or (self.fig is None):

            # make sure ncols nrows are not lower than sizes
            sizes = list(self.loc_label.keys())
            colmax = max([i[0] for i in sizes])
            rowmax = max([i[1] for i in sizes])
            nrow = rowmax + 1
            ncol = colmax + 1

            if isinstance(figsize, int): figsize = (nrow*figsize, ncol*figsize) # type:ignore

            # create figure and display
            self.fig, self.ax = plt.subplots(nrow, ncol, figsize=figsize, layout=layout, **kwargs) # type:ignore
            if not isinstance(self.ax, np.ndarray): self.ax = np.array([self.ax])
            self.display = display.display(self.fig, display_id=True)

            # fill in empty places
            for col in range(ncol):
                for row in range(nrow):
                    if (col, row) not in self.loc_label:
                        self.loc_label[(row, col)] = []

            # sort locations
            self.sorted_data = [v for k,v in sorted(self.loc_label.items(), key=lambda x: x[0])]

            # plot
            for i,ax in enumerate(self.ax.ravel()):
                if i<len(self.sorted_data):
                    labels = self.sorted_data[i]
                    if len(labels) > 0:
                        for label in labels:
                            #data = self.label_data[label]
                            #data["draw func"](ax, data["data"], label=label, **data["kwargs"])
                            self.label_ax[label] = ax
                    else:
                        ax.set_axis_off()
                        ax.set_frame_on(False)
                else:
                    ax.set_axis_off()
                    ax.set_frame_on(False)

            # self.draw() # this causes it to do stuff twice which may or may not fix not drawing bug, Dont Work

        # self.fig.canvas.draw() # type:ignore
        #display.clear_output(wait=True)
        if update: self.display.update(self.fig) # type:ignore


    def update(self, label, data, draw=False):
        if label in self.needs_init:
            self.label_data[label]["draw func"](self.label_ax[label], data, label, **self.label_data[label]["kwargs"])
            self.needs_init.remove(label)
        self.label_data[label]["update func"](self.label_ax[label], data, label, **self.label_data[label]["kwargs"])
        if draw: self.display.update(self.fig) # type:ignore

    def update_from_dict(self, d, draw=False):
        for label, data in d.items():
            self.update(self.label_ax[label], data, draw)

    def close(self): plt.close(self.fig)
    def __enter__(self):
        return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        self.close()

