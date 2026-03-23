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
        binwidth_s: float = 1,
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
        save_subdir="peth_raster",
    ):
        self.raster_renderer = RasterRenderer(
            event_times, spike_times, key, pres, posts, s, linewidths, save_subdir
        )
        self.peth_renderer = PETHRenderer(
            peths, pres, posts, binwidth_s, colors, do_sem, relim, save_subdir
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
            if key is None:
                ax.set_title(f"Unit {idx}")
            else:
                ax.set_title(f"{key}, Unit {idx}")


class PETHRenderer:
    def __init__(
        self,
        peths: dict = None,
        pres: float = 1,
        posts: float = 2,
        binwidth_s: float = 0.1,
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
        self.all_means = {k: v.mean(axis=1) for k, v in peths.items()}
        self.all_stds = {
            k: sem(v, axis=1) if do_sem else v.std(axis=1) for k, v in peths.items()
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
        )

        self.relim = relim
        if not self.relim:
            self.ymin_g = np.min(self.ymin)
            self.ymax_g = np.max(self.ymax)
            padding = 0.05 * (self.ymax_g - self.ymin_g)
            self.ymin_g -= padding
            self.ymax_g += padding

        self.colors = colors
        self.times, _, _ = construct_timebins(pres, posts, binwidth_s)

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
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title(f"Unit {idx}")


class FitRenderer:
    def __init__(self, model=None, x=None, y=None, save_subdir="model_fits"):
        from scipy.stats import pearsonr as r

        self.model = model
        self.x = x
        self.y = y
        self.yhat = self.model(self.x).detach().numpy()
        self.rsquared = r(self.y, self.yhat, axis=0).statistic ** 2

        self.save_subdir = save_subdir

    def __call__(self, idx, fig, axes):
        for ax in axes:
            ax.clear()

        ax = axes[0]
        ax.plot(self.y[:, idx], color="#666666", alpha=0.5, label="observed")
        ax.plot(self.yhat[:, idx], color="#5C2392", alpha=0.5, label="predicted")

        # ax.legend()
        ax.set_xlabel("Trials")
        ax.set_ylabel("Spike Counts")
        ax.set_title(f"$r^2$={self.rsquared[idx]:.3f}")


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

        # get the unique tags from dmat
        self.all_tags = []
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
        for ax in axes:
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
