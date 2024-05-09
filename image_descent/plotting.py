import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, AutoLocator

def ax_plot(ax:Axes, *data, title=None, ylim=None, xlabel=None, ylabel=None):
    ax.plot(*data)
    if title: ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator('auto')) # type:ignore
    ax.yaxis.set_minor_locator(AutoMinorLocator('auto')) # type:ignore
    ax.grid(which="major", color='black', alpha=0.09)
    ax.grid(which="minor", color='black', alpha=0.04)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax
