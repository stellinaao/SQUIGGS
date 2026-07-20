"""
neuron_viewer.py

The neuron viewer handles the logic for generating
sliding plots across units.

Author: Stellina X. Ao
Created: 2026-02-26
Last Modified: 2026-07-20
Python Version: >= 3.10.4
"""

import numpy as np
from damn.alignment import construct_timebins
from squiggs.utils.paths import FIGURES_DIR
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
# from PyQt6.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout,
#     QSlider, QLineEdit
# )
# from PyQt6.QtCore import Qt
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class NeuronViewer:
    def __init__(
        self,
        num_units,
        render_func,
        ymin=None,
        ymax=None,
        ncols=1,
        nrows=1,
        fig_h=2.5,
        fig_w=2.5,
        title="Neuron Viewer",
        fig_dir=FIGURES_DIR,
    ):
        plt.close("all")

        self.num_units = num_units
        self.render_func = render_func

        self.save_dir = fig_dir / self.render_func.save_subdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        mpl.rcParams["keymap.save"] = []

        if hasattr(self.render_func, "ncols"):
            ncols = self.render_func.ncols
        if hasattr(self.render_func, "nrows"):
            nrows = self.render_func.nrows
        if hasattr(self.render_func, "fig_h"):
            fig_h = self.render_func.fig_h
        if hasattr(self.render_func, "fig_w"):
            fig_w = self.render_func.fig_w
        sharey = (
            self.render_func.sharey if hasattr(self.render_func, "sharey") else False
        )
        sharex = (
            self.render_func.sharex if hasattr(self.render_func, "sharex") else False
        )

        self.fig, self.axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(fig_w * ncols, fig_h * nrows),
            sharey=sharey,
            sharex=sharex,
            squeeze=False,  # make logic same for 1 subfig too
        )

        self.fig.subplots_adjust(
            left=0.2,
            right=0.9,
            top=0.8,
            bottom=0.2,  # leave space for slider
            hspace=0.4,  # vertical spacing between rows
            wspace=0.3,  # horizontal spacing between columns
        )

        plt.subplots_adjust(bottom=0.3)

        self.current_idx = 0
        self.render_func(self.current_idx, self.fig, self.axes)

        # slider axis
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(
            slider_ax, "Unit", 0, self.num_units - 1, valinit=0, valstep=1
        )

        self.slider.on_changed(self.update)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)

        self._scroll_dir = 0

        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self._scroll_step)

        # button
        button_ax = plt.axes([0.85, 0.05, 0.1, 0.04])
        self.save_button = Button(button_ax, "Save")
        self.save_button.on_clicked(self.save_fig)

    def update(self, val):
        idx = int(self.slider.val)
        self.render_func(idx, self.fig, self.axes)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "right" or event.key == "l":
            self._scroll_dir = 1
            self.timer.start()

        elif event.key == "left" or event.key == "h":
            self._scroll_dir = -1
            self.timer.start()

        elif event.key == "s":
            self.save_fig(event)

    def on_key_release(self, event):
        if event.key in ["left", "right", "l", "h"]:
            self._scroll_dir = 0
            self.timer.stop()

    def _scroll_step(self):
        if self._scroll_dir == 0:
            return

        idx = int(self.slider.val) + self._scroll_dir

        if 0 <= idx < self.num_units:
            self.slider.set_val(idx)

    def save_fig(self, event):
        idx = int(self.slider.val)
        filename = self.save_dir / f"unit_{idx:03d}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")
        filename = self.save_dir / f"unit_{idx:03d}.svg"
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")
