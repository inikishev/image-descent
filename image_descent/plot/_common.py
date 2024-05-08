"""This thing is taken from my other personal library"""
from typing import Any, Optional
import math
import random
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, AutoLocator

import numpy as np
import torch
#from ..random import randperm

@contextmanager
def seeded_rng(seed:Optional[Any]=0):
    """Context manager, sets seed to torch,numpy and random. If seed is None, does nothing."""
    if seed is None:
        yield
        return
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    yield
    torch.random.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)

def randperm(n,
    *,
    out = None,
    dtype= None,
    layout = None,
    device= None,
    pin_memory = False,
    requires_grad = False,
    seed=None,
    ):
    with seeded_rng(seed):
        return torch.randperm(n, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)

_IGNORE_KWARGS = set(("xlim", "ylim", "xlabel", "ylabel", "title", "axlabelsize","allow_alpha", "mode", "linecolor", "det", "lastn"))

_scalartype = (int,float,np.ScalarType)
def _parse_plotdata(data):
    # to numpy if possible
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)) and isinstance(data[0], torch.Tensor): data = [(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t) for t in data]

    # numpy
    # determine data format and convert to sequence of (2, N) or (1, N), so (B, 2, N) / (B, 1, N)
    if isinstance(data, np.ndarray):
        # list of values
        if data.ndim == 1: return [[data]] # returns (1, 1, N)

        # either list of x,y pairs, will have shape=(2,N) or (N,2), or list of linecharts, will have a shape of (N, B)
        if data.ndim == 2:
            if data.shape[0] == 2: return [data] # data already in (2, N) format, e.g. ([1,2,3,4,5], [3,2,5,6,2]), return (1,2,N)
            if data.shape[1] == 2: return [data.T] # data in (N, 2) format, e.g. ([1, 3], [2, 4], [3, 5]), transpose and return (1, 2, N)

            # multiple lines assumed to be in (N, B), e.g. ([3,5,6,3,5,7,9,1], [1,3,6,1,1,5,7,8,5], [1,2,1,1,1,7,6,5,2])
            return [[line] for line in data] # return (B, 1 N)

        # list of lists of x,y pairs, will have shape=(B, 2, N) or (B, N, 2)
        if data.ndim == 3:
            if data.shape[1] == 2: return [line for line in data] # data in (B, 2, N), return it as list
            if data.shape[2] == 2: return [line.T for line in data] # data in (B, N, 2), return transpose into (B, 2, N)

        else: raise ValueError(f"Invalid data shape for plotting: {data.shape}")

    # dicts
    if isinstance(data, dict):
        items = list(data.items())
        # first value is a number, so its a dictionary of (key: scalar)
        if isinstance(items[0][1], _scalartype):
            # first key is a number, so its a dictionary of (scalar: scalar)
            if isinstance(items[0][0], _scalartype): return [list(zip(items))] # returns (1, 2, N)
            # otherwise we don't know how to use keys, we use only values
            else: return [[[i[1] for i in items]]] # returns only values, (1, 1, N)

        # else it is a dictionary of separate linecharts
        # dictionary of lists of scalars, return list of lists of scalars
        elif isinstance(items[0][1][0], _scalartype): return [[i[1]] for i in items] # returns (B, 1, N)
        # dictionary of lists of x-y pairs
        elif len(items[0][1][0]) == 2: return [list(zip(i[1])) for i in items] # returns (B, 2, N)
        # dictionary of lists of lists of x-y pairs
        else: return [i[1] for i in items] # returns (B, 2, N)

    # other sequences
    # first value is a scalars, so its a list of scalars
    if isinstance(data[0], _scalartype): return [data] # returns (1, 1, N)

    # first value is a list of scalars, so either list of linecharts or x-y pairs
    if isinstance(data[0][0], _scalartype):
        # list of linecharts, e.g. [[1,2,3,6,4], [5,3,1,5,4,2,1], [1,2,5]]
        if len(data[0]) == 2: return [[i] for i in data] # returns (B, 1, N)
        # list of x-y pairs, e.g. [[1,5], [1,8], [3,3], [7,4]]
        else: return [[i] for i in zip(data)] # returns (1, 2, N)

    # else its a list of lists of whatevers
    else:
        # list of lists of x-y pairs, e.g. [[[1,5], [1,8], [3,3], [7,4]], [[1,2], [3,4], [5,6]]]
        if len(data[0][0]) == 2: return [list(zip(i)) for i in data] # returns (B, 2, N)
        # list of lists of x/y, e.g. [[[1,2,3,4],[1,2,2,3]], [[1,2,3,4],[4,5,6,7]]], which is already (B, 2, N)
        else: return data


def _img_norm(x:np.ndarray | torch.Tensor, vmin=None,vmax=None) -> np.ndarray:
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    x = x.astype(np.float32)#  type:ignore
    if vmin or vmax: x = np.clip(x, vmin, vmax)
    x -= x.min()
    max_ = x.max()
    if max_ > 0: x /= max_
    return x # type:ignore

def _img_norm_tensor(x:np.ndarray | torch.Tensor, vmin=None,vmax=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor): x = x.detach().cpu()
    else: x = torch.from_numpy(x)
    x = x.to(torch.float32)#  type:ignore
    if vmin or vmax: x = np.clip(x, vmin, vmax)
    x -= x.min()
    max_ = x.max()
    if max_ > 0: x /= max_
    return x # type:ignore


def _ax_match_target(artists:list[Artist], target:Any, types = (Line2D,PathCollection,AxesImage)) -> list[Artist]:
    artists = [a for a in artists if isinstance(a,types)]
    if target is None: return artists
    if isinstance(target, int): return [artists[target]]
    if isinstance(target, str): return [a for a in artists if a.get_label() == target]
    if isinstance(target, (list, tuple)): return [a for a in artists if a.get_label() in target]
    if isinstance(target, slice): return artists[target]
    if callable(target): return [a for a in artists if target(a)]
    raise ValueError(f"invalid target type {type(target)}")

def _ax_get_array(obj:Any) -> np.ndarray:
    if isinstance(obj, Line2D): return np.asanyarray(obj.get_data())
    if isinstance(obj, PathCollection): return np.asanyarray(obj.get_offsets().data).T # type:ignore
    if isinstance(obj, AxesImage): return np.asanyarray(obj.get_array().data) # type:ignore
    raise ValueError(f"invalid object type {type(obj)}")

def _norm_torange(x, range=(0, 1)): #pylint:disable=W0622
    x -= x.min()
    if x.max()!=0:
        x /= x.max()
    else: return x
    x *= (range[1] - range[0])
    x += range[0]
    return x

def _prepare_linechart_data(data, maxlength=10000, lastn=None):
    if isinstance(data, dict):
        if "x" in data: return data["x"], data["y"]
        else: return list(data.keys()), list(data.values())

    if data is None: return [], []
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        try: data = np.asanyarray(data)
        except ValueError: return [],[]

    if data.ndim == 0:
        return [], []

    elif data.ndim == 1:
        x = list(range(len(data)))
        y = data

    elif data.ndim == 2:
        if data.shape[0] == 2:
            x,y = data
        else: x, y = data.T

    else: raise ValueError(f"Invalid shape {data.shape}, data must be 1D or 2D")

    if len(x) > maxlength:
        x = x[::int(math.ceil(maxlength//len(x)))]
        y = y[::int(math.ceil(maxlength//len(x)))]

    if lastn is not None:
        x = x[-lastn:]
        y = y[-lastn:]
    return x,y

def _prepare_image_data(x, mode='auto', allow_alpha=False):
    # convert to numpy
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        try: x = np.asanyarray(x)
        except ValueError: return [[2,1],[2,1]]

    # determine mode
    if mode == "auto":
        if x.ndim > 3: mode = "*c*"
        if x.ndim == 3 and x.shape[2] > 4 and x.shape[0] < x.shape[2]: mode = 'c*'

    # if batched, take central element
    if mode == "*c*":
        while x.ndim > 3: x = x[int(x.shape[-1]//2)]
        mode = "c*"

    # if channel first, transpose
    if mode == "c*":
        x = x.transpose(1,2,0)

    # fix invalid ch count
    if x.ndim == 3:
        if x.shape[2] == 2:
            x = np.concatenate([x, x[:, :, 0:1]], axis=2)
        elif x.shape[2] > (4 if allow_alpha else 3):
            x = x[:,:,:3]

    return x

def _prepare_scatter10d_data(data, lastn=None):
    """Data must be in B*(1-10) shape!"""
    if isinstance(data, dict):
        if "x" in data: return data
        else: return {"x":list(data.keys()), "y":list(data.values()), "c":None, "s":None, "linewidths":None, "edgecolors":None}
    if data is None: return {"x":[], "y":[], "c":None, "s":None, "linewidths":None, "edgecolors":None}
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        try: data = np.asanyarray(data)
        except ValueError: return {"x":[], "y":[], "c":None, "s":None, "linewidths":None, "edgecolors":None}

    if lastn is not None: data = data[-lastn:]
    if data.ndim == 0: return {"x":[], "y":[], "c":None, "s":None, "linewidths":None, "edgecolors":None}
    if data.ndim == 1: return {"x":list(range(len(data))), "y":data, "c":None, "s":None, "linewidths":None, "edgecolors":None}
    if data.ndim > 2: data = np.asanyarray([i.ravel() for i in data])
    if data.ndim == 2:
        if data.shape[1] == 1: return {"x":list(range(data.shape[0])), "y":data[0], "c":None, "s":None, "linewidths":None, "edgecolors":None}
        if data.shape[1] == 2: return {"x":data[:,0], "y":data[:,1], "c":None, "s":None, "linewidths":None, "edgecolors":None}
        if data.shape[1] == 3: return {"x":data[:,0], "y":data[:,1], "c":_norm_torange(data[:,2], (0,1)), "s":None, "linewidths":None, "edgecolors":None}
        if data.shape[1] == 4: return {"x":data[:,0], "y":data[:,1], "c":_norm_torange(data[:,2], (0,1)), "s":_norm_torange(data[:,3], (20,50)), "linewidths":None, "edgecolors":None}
        if data.shape[1] == 5: return {"x":data[:,0], "y":data[:,1], "c":_norm_torange(data[:,2], (0,1)), "s":_norm_torange(data[:,3], (20,50)), "linewidths":_norm_torange(data[:,4],(2,6)), "edgecolors":None}
        if data.shape[1] == 7: return {"x":data[:,0], "y":data[:,1], "c":_norm_torange(data[:,2:5], (0,1)), "s":_norm_torange(data[:,5], (20,50)), "linewidths":_norm_torange(data[:,6],(2,6)), "edgecolors":None}
        if data.shape[1] == 8: return {"x":data[:,0], "y":data[:,1], "c":_norm_torange(data[:,2], (0,1)), "s":_norm_torange(data[:,3], (20,50)), "linewidths":_norm_torange(data[:,4],(2,6)),  "edgecolors":_norm_torange(data[:,5:],(0,1)),}
        if data.shape[1] == 10: return {"x":data[:,0], "y":data[:,1], "c":_norm_torange(data[:,2:5], (0,1)), "s":_norm_torange(data[:,5], (20,50)), "linewidths":_norm_torange(data[:,6],(2,6)),  "edgecolors":_norm_torange(data[:,7:],(0,1)),}
        elif data.shape[0] <= 10: return _prepare_scatter10d_data(data.T)
        else: raise ValueError(f"Invalid scatter shape {data.shape}")
    else: raise ValueError(f"Invalid scatter shape {data.shape}")

def _prepare_scatter2d_data(data, det=True, lastn=None):
    """Data must be in B* shape!"""
    if isinstance(data, dict):
        if "x" in data: return data
        else: return {"x":list(data.keys()), "y":list(data.values()),}
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        try: data = np.asanyarray(data)
        except ValueError: return {"x":[], "y":[]}

    if lastn is not None: data = data[-lastn:]
    if data.ndim == 0: return {"x":[], "y":[]}
    if data.ndim == 1: return {"x":list(range(len(data))), "y":data}
    if data.ndim > 2: data = np.asanyarray([i.ravel() for i in data])
    if data.ndim == 2:
        if data.shape[1] == 1: return {"x":list(range(data.shape[0])), "y":data[0]}
        if data.shape[1] == 2: return {"x":data[:,0], "y":data[:,1]}
        else:
            if det:
                groups = randperm(data.shape[1], seed=0)
            else: groups = torch.randperm(data.shape[1])
        group1 = groups[:int(groups.shape[0]/2)]
        group2 = groups[int(groups.shape[0]/2):]
        data = np.asanyarray([[x[group1].mean(), x[group2].mean()] for x in data])
        return {"x":data[:,0], "y":data[:,1]}
    else: raise ValueError(f"Invalid scatter shape {data.shape}")

def _match_target(artists:list[Artist], target:Any, types = (Line2D,PathCollection,AxesImage)) -> list[Any]:
    artists = [a for a in artists if isinstance(a,types)]
    if target is None: return artists
    if isinstance(target, int): return [artists[target]]
    if isinstance(target, str): return [a for a in artists if a.get_label() == target]
    if isinstance(target, (list, tuple)): return [a for a in artists if a.get_label() in target]
    if isinstance(target, slice): return artists[target]
    if callable(target): return [a for a in artists if target(a)]
    raise ValueError(f"invalid target type {type(target)}")

def _dict_update(d:dict, **kwargs):
    for k,v in kwargs.items():
        if v is not None:
            d[k] = v
    return d

def _ax_settings(ax:Axes, kwargs):
    prop = {"size":6}

    leg = ax.legend(prop=prop, edgecolor=None)
    leg.get_frame().set_alpha(0.3)

    if "xlim" in kwargs: ax.set_xlim(*kwargs["xlim"])
    if "ylim" in kwargs: ax.set_ylim(*kwargs["ylim"])

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator('auto')) # type:ignore
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator('auto')) # type:ignore

    ax.grid(which="major", color='black', alpha=0.08)
    ax.grid(which="minor", color='black', alpha=0.03)

    if 'axlabelsize' in kwargs: labelsize = kwargs["axlabelsize"]
    else: labelsize = 7
    ax.tick_params(axis="x", labelsize=labelsize, rotation=45)
    ax.tick_params(axis="y", labelsize=labelsize, rotation=0)

    if "xlabel" in kwargs: ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs: ax.set_ylabel(kwargs["ylabel"])
    if "title" in kwargs: ax.set_title(kwargs["title"])
    return ax

def _axplot(ax:Axes, data, label, **kwargs):
    x, y = _prepare_linechart_data(data, lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    plot_kwargs = {k:v for k,v in kwargs.items() if k not in _IGNORE_KWARGS}
    ax.plot(x,y, label=label, **plot_kwargs)
    _ax_settings(ax, kwargs)

def _axplot_update(ax:Axes, data, label, **kwargs):
    x, y = _prepare_linechart_data(data, lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    if ax.lines:
        artists:list[Line2D] = _match_target(ax.get_children(), label, Line2D)
        if len(artists) > 1: raise ValueError(f"Found multiple artisis with {label} for some reason, {[i.get_label() for i in artists]}")
        if len(artists) == 0: raise ValueError(f"No artists with {label} found, there are {[i.get_label() for i in artists]}")
        for l in artists:
            l.set_xdata(x) # type:ignore
            l.set_ydata(y) # type:ignore

    # recompute the ax.dataLim
    ax.relim()
    if "xlim" in kwargs: ax.set_xlim(*kwargs["xlim"])
    if "ylim" in kwargs: ax.set_ylim(*kwargs["ylim"])
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()

def _aximshow(ax:Axes, data, label, **kwargs):
    img = _prepare_image_data(data, kwargs["mode"], kwargs["allow_alpha"])
    plot_kwargs = {k:v for k,v in kwargs.items() if k not in _IGNORE_KWARGS}
    ax.imshow(img, label=label, **plot_kwargs)
    if "xlabel" in kwargs: ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs: ax.set_ylabel(kwargs["ylabel"])
    if "title" in kwargs: ax.set_title(kwargs["title"])

def _aximshow_update(ax:Axes, data, label, **kwargs):
    img = _prepare_image_data(data, kwargs["mode"], kwargs["allow_alpha"])
    artists:list[AxesImage]  = _match_target(ax.get_children(), label, AxesImage)
    if len(artists) > 1: raise ValueError(f"Found multiple artisis with {label} for some reason, {[i.get_label() for i in artists]}")
    if len(artists) == 0: raise ValueError(f"No artists with {label} found, there are {[i.get_label() for i in artists]}")
    for l in artists:
        l.set_data(img) # type:ignore
        l.set_norm(None) # type:ignore

def _axscatter10d(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter10d_data(data, lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    x, y, c, s, linewidths, edgecolors = d["x"], d["y"], d["c"], d["s"], d["linewidths"], d["edgecolors"]
    plot_kwargs = {k:v for k,v in kwargs.items() if k not in (_IGNORE_KWARGS | set(("linewidth",)))}
    ax.scatter(x, y, s=s, c=c, label=label, linewidths=linewidths, edgecolors=edgecolors, **plot_kwargs) # type:ignore
    _ax_settings(ax, kwargs)

def _axscatter10d_update(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter10d_data(data, lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    x, y, c, s, linewidths, edgecolors = d["x"], d["y"], d["c"], d["s"], d["linewidths"], d["edgecolors"]
    artists:list[PathCollection]   = _match_target(ax.get_children(), label, PathCollection)
    if len(artists) > 1: raise ValueError(f"Found multiple artisis with {label} for some reason, {[i.get_label() for i in artists]}")
    if len(artists) == 0: raise ValueError(f"No artists with {label} found, there are {[i.get_label() for i in artists]}")
    for l in artists:
        l.set_offsets(np.stack([x, y]).T) # type:ignore
        if s is not None: l.set_sizes(s) # type:ignore
        if c is not None: l.set_facecolor(c) # type:ignore
        #l.set_color(c)
        l.set_norm(None) # type:ignore
        if linewidths is not None: l.set_linewidth(linewidths) # type:ignore
        if edgecolors is not None: l.set_edgecolor(edgecolors) # type:ignore
    # recompute the ax.dataLim
    ax.relim()
    if "xlim" in kwargs: ax.set_xlim(*kwargs["xlim"])
    if "ylim" in kwargs: ax.set_ylim(*kwargs["ylim"])
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()


def _axscatter2d(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter2d_data(data, det=kwargs["det"], lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    x, y = d["x"], d["y"]
    plot_kwargs = {k:v for k,v in kwargs.items() if k not in _IGNORE_KWARGS}
    ax.scatter(x, y, label=label, **plot_kwargs) # type:ignore
    _ax_settings(ax, kwargs)

def _axscatter2d_update(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter2d_data(data, det=kwargs["det"], lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    x, y = d["x"], d["y"]
    artists:list[PathCollection]   = _match_target(ax.get_children(), label, PathCollection)
    if len(artists) > 1: raise ValueError(f"Found multiple artisis with {label} for some reason, {[i.get_label() for i in artists]}")
    if len(artists) == 0: raise ValueError(f"No artists with {label} found, there are {[i.get_label() for i in artists]}")
    for l in artists:
        l.set_offsets(np.stack([x, y]).T) # type:ignore
    # recompute the ax.dataLim
    ax.relim()
    if "xlim" in kwargs: ax.set_xlim(*kwargs["xlim"])
    if "ylim" in kwargs: ax.set_ylim(*kwargs["ylim"])
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()

def _axpath2d(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter2d_data(data, det=kwargs["det"], lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    d = {k:v for k,v in d.items() if k in ["x", "y"]}
    kwargs["alpha"] = 0.5
    kwargs1 = kwargs.copy()
    kwargs2 = kwargs.copy()
    kwargs1["linewidth"] = 1
    kwargs1["color"] = kwargs["linecolor"]
    kwargs2["c"] = kwargs["markercolor"]
    del kwargs["linecolor"]
    del kwargs["markercolor"]
    _axplot(ax, d, label, **kwargs1)
    _axscatter2d(ax, d, label, **kwargs2)

def _axpath2d_update(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter2d_data(data, det=kwargs["det"], lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    _axplot_update(ax, d, label, **kwargs)
    _axscatter2d_update(ax, d, label, **kwargs)

def _axpath10d(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter10d_data(data, lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    kwargs["alpha"] = 0.5
    kwargs1 = kwargs.copy()
    kwargs1["linewidth"] = 0.66
    kwargs1["color"] = kwargs["linecolor"]
    del kwargs["linecolor"]
    # if "lastn" in kwargs and kwargs["lastn"] is not None:
    #     lastn = kwargs["lastn"]
    #     d = {k:v[-lastn:] for k,v in d.items()}
    _axplot(ax, d, label, **kwargs1)
    _axscatter10d(ax, d, label, **kwargs)

def _axpath10d_update(ax:Axes, data, label, **kwargs):
    d = _prepare_scatter10d_data(data, lastn = kwargs["lastn"] if 'lastn' in kwargs else None)
    # if "lastn" in kwargs and kwargs["lastn"] is not None:
    #     lastn = kwargs["lastn"]
    #     d = {k:v[-lastn:] for k,v in d.items() if v is not None}
    _axplot_update(ax, d, label, **kwargs)
    _axscatter10d_update(ax, d, label, **kwargs)
