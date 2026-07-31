"""
renderers.py

Renderer objects handle a plotting functionality for
a single neuron and are designed to be used in conjunction
with a NeuronViewer() object.

Author: Stellina X. Ao
Created: 2026-02-26
Last Modified: 2026-03-19
Python Version: >= 3.10.4
"""

import numpy as np
from damn.alignment import construct_timebins
from spks.viz import plot_event_based_raster_fast
from scipy.stats import sem
import pandas as pd

__all__ = [
    "PETHRasterRenderer",
    "RasterRenderer",
    "PETHRenderer",
    "FitRenderer",
    "KernelRenderer",
]


class PETHRasterRenderer:
    def __init__(
        self,
        event_times: dict | list | pd.Series = None,
        spike_times: list = None,
        peths: dict = None,
        key: str = None,
        pres: float = 1,
        posts: float = 2,
        binwidth_s: float = 0.1,
        tbin_centers: list=None,
        s: float = 1,
        linewidths: float = 0.5,
        colors: list = [
            "#29723E",
            "#9F5DBC",
            "#A33434",
            "#C49B2C",
            "#245AA0",
            "#E67418",
        ],
        do_sem: bool = True,
        relim: bool = True,
        ymax=None,
        save_subdir="peth_raster",
    ):
        self.raster_renderer = RasterRenderer(
            event_times, spike_times, key, pres, posts, s, linewidths, save_subdir
        )
        self.peth_renderer = PETHRenderer(
            peths, pres, posts, binwidth_s, tbin_centers, colors, do_sem, relim, ymax, save_subdir=save_subdir
        )

        self.ncols = self.raster_renderer.ncols + 1
        self.nrows = self.raster_renderer.nrows
        self.sharey = self.raster_renderer.sharey
        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        self.raster_renderer(idx, fig, axes[:, :-1])
        self.peth_renderer(idx, fig, axes[:, -1])


class RasterRenderer:
    def __init__(
        self,
        event_times: dict | list | pd.Series = None,
        spike_times: list = None,
        key: str = None,
        pres: float = 1,
        posts: float = 2,
        s: float = 0.5,
        linewidths: float = 0.5,
        save_subdir="raster",
    ):
        self.event_times_type = type(event_times)
        if self.event_times_type is list and not np.ndim(event_times) == 1:
            raise ValueError(
                "there are more than one event time conditions, which is incompatible with the list format. try a dict instead."
            )
        self.event_times = event_times
        self.spike_times = spike_times

        self.keys = self.event_times.keys() if self.event_times_type is dict else [key]
        self.pres = pres
        self.posts = posts

        self.s = s
        self.linewidths = linewidths

        self.ncols = len(self.event_times) if self.event_times_type is dict else 1
        self.nrows = 1
        self.sharey = False

        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        for i, (ax, key) in enumerate(zip(axes.flat, self.keys)):
            ax.clear()
            event_times = (
                self.event_times[key]
                if self.event_times_type is dict
                else self.event_times
            )
            plot_event_based_raster_fast(
                event_times,
                self.spike_times[idx],
                pre_seconds=self.pres,
                post_seconds=self.posts,
                s=self.s,
                linewidths=self.linewidths,
                ax=ax,
            )
            ax.set_ylabel("Trials")
            ax.set_xlim([-self.pres, self.posts])
            ax.set_ylim([0, len(self.event_times[key])])
            if key is None:
                ax.set_title(f"Unit {idx}")
            else:
                ax.set_title(f"{key}, Unit {idx}")

class PETHWeightCompRenderer:
    def __init__(
            self,
            weights: dict = None,
            weight_names: list = None,
            robs: dict = None,
            robs_ylabel: str = "Spike Counts",
            sc_tavgs: dict = None,
            event_times: dict = None,
            spike_times: list = None,
            peths: dict = None,
            key: str = None,
            pres: float = 0.5,
            posts: float = 1,
            binwidth_s: float = 25/1000,
            s: float = 1,
            linewidths: float = 0.5,
            colors: list = [
                "#29723E",
                "#9F5DBC",
                "#A33434",
                "#C49B2C",
                "#245AA0",
                "#E67418",
            ],
            same_ylim: bool = True,
            do_sem: bool = True,
            relim: bool = True,
            save_subdir="peth_weights_comp",
    ):
        self.peth_weight_renderers = []

        # get ylims
        if same_ylim:
            all_means = {k_: {
                k: ((1 / binwidth_s) * v).mean(axis=1) for k, v in peths[k_].items()
            } for k_ in peths}
            all_stds = {k_: {
                k: sem((1 / binwidth_s) * v, axis=1)
                if do_sem
                else ((1 / binwidth_s) * v).std(axis=1)
                for k, v in peths[k_].items()
            } for k_ in peths}

            ymax = np.max(np.concatenate([[np.max(all_means[strategy][k] + all_stds[strategy][k], axis=1) for k in peths[strategy].keys()] for strategy in ['mb', 'mf']]), axis=0)
        else:
            ymax=None

        for key in weights.keys():
            self.peth_weight_renderers.append(
                PETHWeightRenderer(
                    weights=weights[key],
                    weight_names=weight_names,
                    robs=robs[key],
                    robs_ylabel=robs_ylabel,
                    sc_tavg=sc_tavgs[key],
                    event_times=event_times[key],
                    spike_times=spike_times, 
                    peths=peths[key],
                    key=key,
                    pres=pres,
                    posts=posts,
                    binwidth_s=binwidth_s,
                    s=s,
                    linewidths=linewidths,
                    colors=colors,
                    do_sem=do_sem,
                    ymax=ymax,
                    relim=relim,
                    save_subdir=save_subdir

                )
            )
        
        self.nrows=len(self.peth_weight_renderers)
        self.ncols=self.peth_weight_renderers[0].ncols
        self.sharey = False
        self.save_subdir = save_subdir
    def __call__(self, idx, fig, axes):
        for i, r in enumerate(self.peth_weight_renderers):
            r(idx, fig, axes[i].reshape(1,-1))
        
class PETHWeightRenderer:
    def __init__(
        self,
        weights,
        weight_names,
        robs,
        sc_tavg,
        robs_ylabel=None,
        event_times: dict | list | pd.Series = None,
        spike_times: list = None,
        peths: dict = None,
        key: str = None,
        pres: float = 1,
        posts: float = 2,
        binwidth_s: float = 0.1,
        tbin_edges: list = None,
        s: float = 1,
        linewidths: float = 0.5,
        colors: list = [
            "#29723E",
            "#9F5DBC",
            "#A33434",
            "#C49B2C",
            "#245AA0",
            "#E67418",
        ],
        do_sem: bool = True,
        ymax: list = None,
        relim: bool = True,
        save_subdir="peth_weights",
    ):
        self.weight_renderer = WeightRenderer(
            weights, weight_names, save_subdir
        )

        self.robs_renderer = ROBSRenderer(
            robs, ylabel=robs_ylabel, save_subdir=save_subdir
        )

        self.sctavg_renderer = SCTAVGRenderer(
            sc_tavg
        )

        self.peth_renderer = PETHRenderer(
            peths, pres, posts, binwidth_s, tbin_edges, colors, do_sem=do_sem, ymax=ymax, relim=relim, save_subdir=save_subdir
        )

        self.raster_renderer = RasterRenderer(
            event_times, spike_times, key, pres, posts, s, linewidths, save_subdir
        )

        self.ncols = 6
        self.nrows = 1
        self.sharey = False
        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        self.weight_renderer(idx, fig, axes[:,0])
        self.robs_renderer(idx, fig, axes[:,2])
        self.sctavg_renderer(idx, fig, axes[:,1])
        self.peth_renderer(idx, fig, axes[:,3])
        self.raster_renderer(idx, fig, axes[:,-2:])
    
class PETHWeightRendererTime:
    def __init__(
        self,
        weights,
        weight_names=None,
        weight_idxs=None,
        tv=None,
        tv_vals=None,
        mode="trace",
        peths: dict = None,
        pres: float = 1,
        posts: float = 2,
        binwidth_s: float = 1,
        tbin_centers: list = None,
        colors: list = [
            "#29723E",
            "#9F5DBC",
            "#A33434",
            "#C49B2C",
            "#245AA0",
            "#E67418",
        ],
        do_sem: bool = True,
        ymax: list = None,
        relim: bool = True,
        save_subdir="peth_weights",
    ):
        if mode=="matrix":
            self.weight_renderer = WeightRenderer(
                weights, weight_names, tbin_centers, save_subdir
            )
        elif mode=="trace":
            self.weight_renderer = WeightRendererTime(
                weights=weights, tv=tv, weight_idxs=weight_idxs, tv_vals=tv_vals, tbin_centers=tbin_centers, save_subdir=save_subdir
            )
        else:
            raise ValueError("mode must be 'matrix' or 'trace'")

        self.peth_renderer = PETHRenderer(
            peths, pres, posts, binwidth_s, tbin_centers, colors, do_sem=do_sem, ymax=ymax, relim=relim, save_subdir=save_subdir
        )

        self.ncols = 2
        self.nrows = 1
        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        self.weight_renderer(idx, fig, axes[:,0])
        self.peth_renderer(idx, fig, axes[:,1])

class WeightRendererTimeAll:  
    def __init__(
            self,
            weights,
            tv_keys,
            weight_idxs,
            tv_vals,
            tbin_centers,
            colors: list = [
                "#29723E",
                "#9F5DBC",
                "#A33434",
                "#C49B2C",
                "#245AA0",
                "#E67418",
            ],
            save_subdir="weights_time",
    ):
        self.renderers = {}
        for tv in tv_keys:
            self.renderers[tv] = WeightRendererTime(
                weights=weights,
                tv=tv,
                weight_idxs=weight_idxs,
                tv_vals=tv_vals,
                tbin_centers=tbin_centers,
                colors=colors,
                save_subdir=save_subdir,
            )
        self.nrows = len(tv_keys)
        self.ncols = 1
        
        self.fig_h = 0.75
        self.fig_w = 2.5 

        self.sharex = True
        self.save_subdir = save_subdir
    
    def __call__(self, idx, fig, axes):
        for i, (r, ax) in enumerate(zip(self.renderers.values(), axes.flat)):
            r(idx, fig, ax)

            legend = ax.get_legend()
            for text in legend.get_texts():
                text.set_fontsize(3)
            
class WeightRendererTime:
    def __init__(
            self,
            weights,
            tv,
            weight_idxs,
            tv_vals,
            tbin_centers,
            colors: list = [
                "#29723E",
                "#9F5DBC",
                "#A33434",
                "#C49B2C",
                "#245AA0",
                "#E67418",
            ],
            save_subdir="weights_time",
    ):
        self.weights = weights
        self.tv = tv
        self.weight_idxs = weight_idxs
        self.tv_vals = tv_vals

        self.regressors = [f"{tv}_{val}" for val in tv_vals[tv]]
        self.regr_idxs = [self.weight_idxs[regr] for regr in self.regressors]

        self.tbin_centers = tbin_centers
        self.colors = colors
        self.save_subdir = save_subdir
    
    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        weights_ = self.weights[:,idx,self.regr_idxs].T

        for i, regr in enumerate(self.regressors): 
            ax.plot(weights_[i], color=self.colors[i], label=regr)
    
        ax.legend()

class WeightRenderer:
    def __init__(
            self,
            weights,
            weight_names,
            tbin_centers=None,
            save_subdir="weights",
    ):
        self.weights = weights
        self.weight_names = weight_names
        self.save_subdir = save_subdir

        self.tbin_centers=tbin_centers
    
    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        if self.weights.ndim == 2:
            weights_ = self.weights[idx].reshape(-1,1)
        elif self.weights.ndim == 3:
            weights_ = self.weights[:,idx,:].T

        im = ax.imshow(weights_, vmin=-1, vmax=1, cmap='coolwarm', )
        if self.weights.ndim==2:
            ax.set_xticks([])
            ax.set_yticks(np.arange(self.weights.shape[1]), self.weight_names)
        elif self.weights.ndim==3:
            # ax.set_xticks(range(self.weights.shape[0]), self.tbin_centers)
            # print(ax.get_xticks())
            # # ax.set_xticks(ax.get_xticks(), [f"{t:.2f}" for t in self.tbin_centers[ax.get_xticks()]])
            ax.set_xticks([])
            ax.set_xlabel("Trial Time")
            ax.set_yticks(np.arange(self.weights.shape[2]), self.weight_names)
        # fig.colorbar(im, label=r'$\beta$ weight')
    
class SCTAVGRenderer:
    def __init__(
            self,
            sc_tavg,
            save_subdir="sctavg"
    ):
        self.sc_tavg = sc_tavg
        self.keys = sc_tavg.keys()
        self.save_subdir = save_subdir
    
        self.ymin = min([min(sc_tavg[k]) for k in self.keys])
        self.ymax = max([max(sc_tavg[k]) for k in self.keys])

    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        ax.bar(self.keys, [self.sc_tavg[k][idx] for k in self.keys])
        ax.set_ylim([self.ymin, self.ymax])
        ax.set_ylabel(r"avg spike counts $^*$")

class ROBSRenderer:
    def __init__(
            self,
            robs,
            ylabel="Firing Rate (Hz)",
            save_subdir="robs",
    ):
        self.robs = robs
        self.ylabel = ylabel
        self.save_subdir=save_subdir

    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        ax.plot(self.robs[:,idx])
        ax.set_xlabel("Trials")
        ax.set_ylabel(self.ylabel)

class PETHRenderer:
    def __init__(
        self,
        peths: dict = None,
        pres: float = 1,
        posts: float = 2,
        binwidth_s: float = 0.1,
        tbin_centers: list=None,
        ylabel = "Firing Rate (Hz)",
        colors: list = [
            "#29723E",
            "#9F5DBC",
            "#A33434",
            "#C49B2C",
            "#245AA0",
            "#E67418",
        ],
        do_sem: bool = True,
        relim: bool = True,
        ymax: list = None,
        save_subdir="peth",
    ):
        """
        Parameters
        ----------
        peths = {'cond1': peths_cond1 (shape=(n_units, n_trials, n_bins)),
                    ...,
                 'condN': peths_condN (shape=identical to cond1)
                 }

        Example
        ----------
        >> psths_cond = {
            "left_corr": psths[left_corr_mask],
            "right_corr": psths[right_corr_mask],
            "left_incorr": psths[left_incorr_mask],
            "right_incorr": psths[right_incorr_mask],
        }

        >> renderer = PETHRenderer(
            peths=psths_cond,
            pres=tpre,
            posts=tpost,
            binwidth_s=binwidth_ms/1000,
        )

        >> nv = NeuronViewer(num_units=psths['ACC'].shape[0], render_func=renderer)
        """

        self.peths = peths

        # ensure that the same number of cells are present for each condition
        assert len(np.unique([v.shape[0] for v in self.peths.values()])) == 1, (
            "number of cells in each condition should be the same, but are not"
        )

        # ensure that there are enough colors
        assert len(self.peths) <= len(colors), (
            "not enough colors to support number of conditions"
        )
        print(binwidth_s)
        self.all_means = {
            k: ((1 / binwidth_s) * v).mean(axis=1) for k, v in peths.items()
        }
        self.all_stds = {
            k: sem((1 / binwidth_s) * v, axis=1)
            if do_sem
            else ((1 / binwidth_s) * v).std(axis=1)
            for k, v in peths.items()
        }

        self.ymin = np.min(
            [
                np.min(self.all_means[k] - self.all_stds[k], axis=1)
                for k in peths.keys()
            ],
            axis=0,
        )
        
        self.ymax = np.max(
            [
                np.max(self.all_means[k] + self.all_stds[k], axis=1)
                for k in peths.keys()
            ],
            axis=0,
        ) if ymax is None else ymax

        self.relim = relim
        if not self.relim:
            self.ymin_g = np.min(self.ymin)
            self.ymax_g = np.max(self.ymax)
            padding = 0.05 * (self.ymax_g - self.ymin_g)
            self.ymin_g -= padding
            self.ymax_g += padding

        self.colors = colors
        self.ylabel = ylabel
        if tbin_centers is None:
            self.times, _, _ = construct_timebins(pres, posts, binwidth_s)
        else:
            self.times = tbin_centers

        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        for i, k in enumerate(self.peths.keys()):
            mean = self.all_means[k][idx]
            std = self.all_stds[k][idx]
            ax.plot(self.times, mean, color=self.colors[i], label=k)
            ax.fill_between(
                self.times,
                mean - std,
                mean + std,
                alpha=0.3,
                color=self.colors[i],
            )

        ax.axvline(x=0, color="#666666", linewidth=0.5, linestyle="--")
        ax.legend()

        if self.relim:
            padding = 0.05 * (self.ymax[idx] - self.ymin[idx])
            ax.set_ylim(self.ymin[idx] - padding, self.ymax[idx] + padding)
        else:
            ax.set_ylim(self.ymin_g, self.ymax_g)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.ylabel)
        ax.set_title(f"Unit {idx}")

class FitRenderer:
    def __init__(self, model=None, x=None, y=None, yhat=None, ylabel="Firing Rate (Hz)", rsquared=None, add_r2=True, dfs=None, mode='lite', color=None, save_subdir="model_fits"):
        from scipy.stats import pearsonr as r

        if mode=='liska':
            self.model = model
            self.x = x
            self.y = y
            self.yhat = self.model(self.x).detach().numpy()

        elif mode=='lite':
            self.y = y
            self.yhat = yhat

        self.color = "#5C2392" if color is None else color
        self.ylabel = ylabel
        
        # dfs supports masking (mostly specific to stellina's lvm but could be adapted)
        self.dfs = np.ones(y.shape) if dfs is None else dfs
        self.add_r2 = add_r2

        if self.add_r2:
            self.rsquared = self.get_r2(self.y, self.yhat, self.dfs) if rsquared is None else rsquared

        self.save_subdir = save_subdir

    def get_r2(self, y, yhat, dfs, eps=1e-10):
        ybar = (y * dfs).sum(axis=0) / dfs.sum(axis=0)  # the average y value
        resids = y - yhat  # the difference between observed and predicted
        residnull = y - ybar  # the difference between observed and observed avg
        sstot = np.sum(residnull**2 * dfs, axis=0) + eps  # denom
        ssres = np.sum(resids**2 * dfs, axis=0)  # num
        r2 = 1 - ssres / sstot

        return r2

    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        ax.plot(self.y[:, idx], color="#666666", alpha=0.5, label="observed")
        ax.plot(self.yhat[:, idx], color=self.color, alpha=0.5, label="predicted")

        # ax.legend()
        ax.set_xlabel("Trials")
        ax.set_ylabel(self.ylabel)
        
        if self.add_r2:
            ax.set_title(f"$r^2$={self.rsquared[idx]:.3f}")

class FitRendererTime:
    def __init__(self, x=None, y=None, yhat=None, ylabel="Firing Rate (Hz)", rsquared=None, color=None, save_subdir="model_fits"):
        from scipy.stats import pearsonr as r

        self.x = x
        self.y = y
        self.yhat = yhat

        self.color = "#5C2392" if color is None else color
        self.ylabel = ylabel
        
        self.rsquared = rsquared

        self.ncols = 6
        self.nrows = 6

        self.fig_h = 0.6
        self.fig_w = 0.8

        self.sharex = True
        self.sharey = True

        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        idxs = np.sort(np.random.choice(self.y.shape[1], self.ncols*self.nrows))

        for i, ax in enumerate(axes.flat):
            ax.clear()
            ax.plot(self.x, self.y[:,idxs[i],idx].T, alpha=0.5, linewidth=0.5, color="#666666", label="observed")
            ax.plot(self.x, self.yhat[:,idxs[i],idx].T, alpha=0.5, linewidth=0.5, color="#8F64DB", label="predicted")

            if i % self.ncols == 0:
                ax.set_ylabel(self.ylabel, fontsize=3)
            if i // self.nrows == self.nrows-1:
                ax.set_xlabel("Trial Time (s)", fontsize=4)

            # if i==0:
            #     ax.legend(loc="upper left")
            
        fig.suptitle(fr"$r^2$={self.rsquared[idx]:.3f}")


class FitRendererCompare:
    def __init__(self, y, yhat1, yhat2, label1, label2, ylabel="Firing Rate (Hz)", dfs1=None, dfs2=None, rsquared1=None, rsquared2=None, save_subdir="fit_comp"):
        from scipy.stats import pearsonr as r

        self.y = y
        self.yhat1 = yhat1
        self.yhat2 = yhat2

        self.ylabel = ylabel

        self.label1 = label1
        self.label2 = label2

        self.color1 = "#5C2392"
        self.color2 = "#23926F"

        # dfs supports masking (mostly specific to stellina's lvm but could be adapted)
        self.dfs1 = np.ones(y.shape) if dfs1 is None else dfs1
        self.dfs2 = np.ones(y.shape) if dfs2 is None else dfs2

        self.rsquared1 = self.get_r2(self.y, self.yhat1, self.dfs1) if rsquared1 is None else rsquared1
        self.rsquared2 = self.get_r2(self.y, self.yhat2, self.dfs2) if rsquared2 is None else rsquared2
        
        self.save_subdir = save_subdir

    def get_r2(self, y, yhat, dfs, eps=1e-10):
        ybar = (y * dfs).sum(axis=0) / dfs.sum(axis=0)  # the average y value
        resids = y - yhat  # the difference between observed and predicted
        residnull = y - ybar  # the difference between observed and observed avg
        sstot = np.sum(residnull**2 * dfs, axis=0) + eps  # denom
        ssres = np.sum(resids**2 * dfs, axis=0)  # num
        r2 = 1 - ssres / sstot

        return r2

    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        ax.plot(self.y[:, idx], color="#666666", alpha=0.5, label="observed")
        ax.plot(self.yhat1[:, idx], color=self.color1, alpha=0.5, label=f"{self.label1}")
        ax.plot(self.yhat2[:, idx], color=self.color2, alpha=0.5, label=f"{self.label2}")

        ax.legend(loc="upper left")
        ax.set_xlabel("Trials")
        ax.set_ylabel(self.ylabel)
        ax.set_title(f"$r^2_a$={self.rsquared1[idx]:.3f}, $r^2_b$={self.rsquared2[idx]:.3f}")

class KernelRenderer:
    def __init__(self, model=None, dmat=None, bias=None, subdir="kernel"):
        """
        Parameters
        ----------
        mode : "grand" or "cond"
            grand -> single mean/std
            cond  -> separate a/b condition mean/std

        Example
        ----------
        >> renderer_grand = PETHRenderer(peth, pres, posts, binwidth_s, mode="grand")
        >> viewer1 = NeuronViewer(num_units=peth.shape[0], render_func=renderer_grand, ymin=renderer_grand.ymin, ymax=renderer_grand.ymax)


        >> renderer_cond = PETHRenderer(
            peth_a=peth_l,
            peth_b=peth_r,
            mode="cond",
            label_a="left",
            label_b="right"
        )
        >> viewer2 = NeuronViewer(num_units=peth.shape[0], render_func=renderer_cond, ymin=renderer_cond.ymin, ymax=renderer_cond.ymax)
        """
        self.linkfunc = model.estimators_[0]._base_loss.link.inverse


        for _, reg in dmat.regressors.items():
            self.all_tags.extend(reg.tags)
        self.all_tags = np.unique(self.all_tags)
        self.all_tags = [
            t
            for t in self.all_tags
            if t not in ["task", "interaction", "hmm", "behavior"]
        ]

        self.model = model
        self.dmat = dmat
        self.bias = bias

        self.cache = {}
        ymin = np.inf
        ymax = -np.inf

        for tag in self.all_tags:
            self.cache[tag] = {}
            regs = self.dmat.select(tag=tag)

            for r, reg in regs.items():
                k_all, t = reg.reconstruct_kernel()
                self.cache[tag][f"{reg}_t"] = t
                self.cache[tag][f"{reg}_k"] = np.zeros((len(bias), t.shape[0]))

                for idx in range(len(bias)):
                    k = k_all[:, idx]
                    k = self.linkfunc(k + bias[idx])

                    max_curr = np.max(k)
                    min_curr = np.min(k)

                    if max_curr > ymax:
                        ymax = max_curr
                    if min_curr < ymin:
                        ymin = min_curr

                    self.cache[tag][f"{reg}_k"][idx] = k
        self.ymin = ymin
        self.ymax = ymax

        self.sharey = True

        self.subdir = subdir

    def __call__(self, idx, fig, axes):
        ax = (
            axes[0][0]
            if np.ndim(axes) > 1
            else (axes[0] if np.ndim(axes) > 0 else axes)
        )
        ax.clear()

        for i, tag in enumerate(self.all_tags):
            regs = self.dmat.select(tag=tag)
            for r, reg in regs.items():
                axes[i].plot(
                    self.cache[tag][f"{reg}_t"],
                    self.cache[tag][f"{reg}_k"][idx],
                    label=reg.name,
                )
            axes[i].axvline(x=0, linewidth=0.5, linestyle="--", color="#333333")
            axes[i].set_title(tag)
            if tag not in ["history", "dlc", "video"]:
                axes[i].legend()
            axes[i].set_xlabel("Time (s)")

        axes[0].set_ylabel("Weight")
        fig.suptitle(f"Unit {idx}")
