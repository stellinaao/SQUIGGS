import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection

""" TRAJECTORIES """
# plot a 2d trajectory with line color encoding time (LineCollection)
def plot_trajectory(
    x,
    y,
    xlabel="",
    ylabel="",
    title="",
    cmap="plasma",
    s=2,
    start_marker="*",
    end_marker="s",
    mid_marker=".",
    start_size_mult=10,
    end_size_mult=2,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2), tight_layout=True)

    t = np.arange(len(x))
    vmin, vmax = t.min(), t.max()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, array=t[:-1], linewidth=1)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)

    # middle points
    if len(x) > 2:
        ax.scatter(
            x[1:-1], y[1:-1], c=t[1:-1], cmap=cmap, vmin=vmin, vmax=vmax,
            s=s, marker=mid_marker, zorder=3,
        )

    # start and end, drawn on top with distinct markers
    ax.scatter(
        x[0], y[0], c="#ffffff", 
        s=s * start_size_mult, marker=start_marker, zorder=4,
        edgecolors="k", linewidths=0.5,
    )
    ax.scatter(
        x[-1], y[-1], c="#ffffff", 
        s=s * end_size_mult, marker=end_marker, zorder=4,
        edgecolors="k", linewidths=0.5,
    )

    # axes
    ax.axvline(x=0, color="k", linewidth=0.5, zorder=-2)
    ax.axhline(y=0, color="k", linewidth=0.5, zorder=-2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax