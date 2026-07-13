# main_viewer.py
# -*- coding: utf-8 -*-
"""
Interactive STM/AFM Data Viewer
================================
Provides the graphical interface for exploring Createc STM/AFM data.
All file I/O and calibration logic lives in read_files.py; this module
is responsible solely for the Matplotlib / Tkinter GUI.

``STMAFMEntity`` inherits from ``STMAFMReader`` (read_files.py) and adds
interactive plotting on top of the loaded data.

Key Features:
    - Interactive image display with zooming and panning.
    - Right-click context menu for selecting data channels.
    - Visualization of spectra positions on the image, rotated and offset
      according to scan metadata of the image.
    - Left-click selection of regions-of-interest with configurable radius.
    - On-the-fly extraction and display of nearby spectra in a separate window.

Author: Ricardo Ruvalcaba, August 17, 2025
Potentially to be published in:
    - https://joss.theoj.org/
    - https://openresearchsoftware.metajnl.com/
    - https://link.springer.com/journal/41664

Related repositories:
    - https://github.com/abekipnis/Kondo-Aalto
    - https://github.com/spectrafox/spectrafox
    - https://github.com/cahlikales/Kondo_fit
"""

import os
import sys
import tkinter as tk
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

matplotlib.use("TkAgg")

from spectra_window import open_spectra_window  # noqa: E402  (path-inserted import)
from utilities import create_colormap_menu, pan_factory, zoom_factory  # noqa: E402

from read_files import (  # noqa: E402
    CHANNEL_LABELS,
    CLIM_PERCENTILE_HIGH,
    CLIM_PERCENTILE_LOW,
    PLOT_EXTENT_PADDING,
    SELECTION_RADIUS_DEFAULT,
    SLIDER_LOG_MAX,
    SLIDER_LOG_MIN,
    SLIDER_STEP,
    STMAFMReader,
)

# make it so that it takes an argument for folder, if not let's use current folder, Thalis
dpath = False
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder  = dpath

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': False,
    'svg.fonttype': 'none',
})


# ---------------------------------------------------------------------------
# Viewer class
# ---------------------------------------------------------------------------

class STMAFMEntity(STMAFMReader):
    """Interactive STM/AFM viewer.

    Extends :class:`STMAFMReader` with a full Matplotlib / Tkinter GUI.
    All data loading, calibration, and coordinate handling is provided by the
    parent class; this class is responsible only for rendering and interaction.

    Parameters
    ----------
    folder_path:
        Absolute path to the folder containing exactly one .DAT file and
        any number of .VERT / .Vert spectra files.
    """

    def __init__(self, folder_path: str) -> None:
        # Interaction state — set before super().__init__ so all attributes
        # exist if any parent method were to reference them during init.
        self.selection_radius: float = SELECTION_RADIUS_DEFAULT
        self.selection_circle: Optional[Circle] = None
        self.slider_active: bool = False
        self.last_click: Optional[tuple[float, float]] = None

        # Colormap state (tk.StringVar requires an existing Tk root,
        # which is created in __main__ before this constructor is called)
        self.current_cmap: tk.StringVar = tk.StringVar(value="gray")

        # Delegate all file I/O to the parent
        super().__init__(folder_path)

    # ------------------------------------------------------------------
    # GUI - slider
    # ------------------------------------------------------------------

    def _add_slider(self) -> None:
        """Add a logarithmic selection-radius slider below the main axes.

        The slider controls ``self.selection_radius`` (in angstrom) on a
        log10 scale between 1 A and 200 A.  While the slider is being
        dragged, a preview circle is shown centred on the current view centre.
        """
        ax_slider = self.fig.add_axes([0.25, 0.02, 0.5, 0.03])
        self.slider = Slider(
            ax_slider,
            'Selection Radius (Å)',
            SLIDER_LOG_MIN,
            SLIDER_LOG_MAX,
            valinit=np.log10(self.selection_radius),
            valstep=SLIDER_STEP,
        )

        def _update_valtext(log_val: float) -> None:
            self.slider.valtext.set_text(f"{10**log_val:.2f}")

        def _update_radius(log_val: float) -> None:
            self.selection_radius = 10 ** log_val
            _update_valtext(log_val)
            if self.slider_active:
                center = self._get_plot_center()
                self._update_selection_circle(center)

        def _on_slider_press(event: MouseEvent) -> None:
            if event.inaxes == self.slider.ax:
                self.slider_active = True
                center = self._get_plot_center()
                self._update_selection_circle(center)
                self.im.figure.canvas.draw_idle()

        def _on_slider_release(event: MouseEvent) -> None:
            if self.slider_active:
                self.slider_active = False
                if self.selection_circle is not None:
                    self.selection_circle.remove()
                    self.selection_circle = None
                    self.im.figure.canvas.draw_idle()

        self.slider.on_changed(_update_radius)
        _update_valtext(np.log10(self.selection_radius))

        self.fig.canvas.mpl_connect("button_press_event", _on_slider_press)
        self.fig.canvas.mpl_connect("button_release_event", _on_slider_release)

    # ------------------------------------------------------------------
    # GUI - interaction callbacks
    # ------------------------------------------------------------------

    def _get_plot_center(self) -> tuple[float, float]:
        """Return the (x, y) centre of the current axes view in data units."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        return (
            (xlim[0] + xlim[1]) / 2.0,
            (ylim[0] + ylim[1]) / 2.0,
        )

    def _show_context_menu(self, event: MouseEvent) -> None:
        """Display a right-click Tk context menu for channel selection.

        Parameters
        ----------
        event:
            Matplotlib mouse event; only processed for button 3 (right-click).
        """
        if event.button != 3:
            return

        menu = tk.Menu(None, tearoff=0)
        for name, idx in self.available_channels:
            menu.add_command(
                label=name,
                command=lambda n=name, i=idx: self._update_channel(n, i),
            )
        menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)

    def _handle_click(self, event: MouseEvent) -> None:
        """Handle left-click: draw selection circle and open nearby spectra.

        Iterates over all loaded spectra and calls :func:`open_spectra_window`
        with those whose rotated position falls within ``self.selection_radius``
        of the click location.

        Parameters
        ----------
        event:
            Matplotlib mouse event; only processed for button 1 (left-click).
        """
        if event.button != 1:
            return

        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        self.last_click = (x_click, y_click)
        self._update_selection_circle((x_click, y_click))

        spectra_nearby = [
            spec for spec in self.spectra
            if np.hypot(
                spec['rotated'][0] - x_click,
                spec['rotated'][1] - y_click,
            ) <= self.selection_radius
        ]

        if spectra_nearby:
            open_spectra_window(spectra_nearby, (x_click, y_click))

        self.im.figure.canvas.draw_idle()

    def _update_selection_circle(
        self,
        center: Optional[tuple[float, float]],
    ) -> None:
        """Replace the current selection circle with a new one at *center*.

        Parameters
        ----------
        center:
            (x, y) position in data coordinates, or None to remove the circle.
        """
        if self.selection_circle is not None:
            self.selection_circle.remove()
            self.selection_circle = None

        if center is None:
            return

        x, y = center
        self.selection_circle = Circle(
            (x, y),
            radius=self.selection_radius,
            color='red',
            fill=False,
            linestyle='--',
            linewidth=1.5,
        )
        self.ax.add_patch(self.selection_circle)

    def _update_channel(self, channel_name: str, channel_index: int) -> None:
        """Switch the displayed image channel and refresh the figure.

        Parameters
        ----------
        channel_name:
            Human-readable channel name (used for colour-bar label lookup).
        channel_index:
            Index into ``self.scan.imgs`` for the new channel.
        """
        self.current_channel = channel_name
        self._load_image_channel(channel_index)

        vmin, vmax = np.percentile(
            self.image,
            [CLIM_PERCENTILE_LOW, CLIM_PERCENTILE_HIGH],
        )
        self.im.set_data(self.image)
        self.im.set_clim(vmin, vmax)

        label = CHANNEL_LABELS.get(channel_name, channel_name)
        self.cbar.set_label(label)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Main plotting entry point
    # ------------------------------------------------------------------

    def plot(self) -> None:
        """Create and display the interactive STM/AFM viewer figure.

        Sets up:
        * The main image axes with imshow, colour bar, and axis labels.
        * Scatter markers for the scan offset and all spectra positions.
        * A square auto-computed view extent that encompasses both the image
          and all spectrum positions, with a small padding margin.
        * Zoom, pan, and colormap-menu widgets.
        * The selection-radius slider.
        * Mouse event callbacks for context menu and spectra selection.
        * A close-event handler that terminates the Tk main loop cleanly.
        """
        print("[INFO] Preparing main figure...")

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.get_current_fig_manager().window.wm_geometry("+0+0")

        self._load_image_channel(self.first_valid_index)

        extent = [
            self.X0.min(), self.X0.max(),
            self.Y0.min(), self.Y0.max(),
        ]
        self.im = self.ax.imshow(
            self.image,
            cmap=self.current_cmap.get(),
            origin='lower',
            extent=extent,
            aspect='equal',
        )

        label = CHANNEL_LABELS.get(self.current_channel, self.current_channel)
        self.cbar = plt.colorbar(self.im, ax=self.ax, label=label)
        create_colormap_menu(self.fig, self.im, self.current_cmap)

        # Scan offset marker
        self.ax.scatter(
            self.offset_A_x, self.offset_A_y,
            facecolors='white', color='black', label='Offset',
        )

        # Spectra position markers
        for spec in self.spectra:
            x_rot, y_rot = spec['rotated']
            self.ax.scatter(x_rot, y_rot, facecolor='None', edgecolor=spec['color'], linewidths=1.5)

        # --- Compute square view extent ---
        spec_x = [spec['rotated'][0] for spec in self.spectra]
        spec_y = [spec['rotated'][1] for spec in self.spectra]

        img_xmin, img_xmax = float(self.X0.min()), float(self.X0.max())
        img_ymin, img_ymax = float(self.Y0.min()), float(self.Y0.max())

        if spec_x and spec_y:
            xmin = min(img_xmin, min(spec_x))
            xmax = max(img_xmax, max(spec_x))
            ymin = min(img_ymin, min(spec_y))
            ymax = max(img_ymax, max(spec_y))
        else:
            xmin, xmax = img_xmin, img_xmax
            ymin, ymax = img_ymin, img_ymax

        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)
        half_size = 0.5 * max(xmax - xmin, ymax - ymin) * PLOT_EXTENT_PADDING

        # Note: y-axis is intentionally inverted (ymax first) to match imshow
        # origin='lower' combined with Createc's coordinate convention.
        self.ax.set_xlim(x_center - half_size, x_center + half_size)
        self.ax.set_ylim(y_center + half_size, y_center - half_size)

        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_title("Interactive Spectra Processor")

        zoom_factory(self.ax)
        pan_factory(self.ax, button=2)

        self.fig.canvas.mpl_connect('button_press_event', self._show_context_menu)
        self.fig.canvas.mpl_connect('button_press_event', self._handle_click)

        self._add_slider()

        def _on_close(event) -> None:  # noqa: ANN001 (Matplotlib event type)
            plt.close('all')
            root.quit()
            root.destroy()
            sys.exit(0)

        self.fig.canvas.mpl_connect('close_event', _on_close)

        print("[INFO] Displaying interactive viewer...")
        plt.show(block=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    if folder:
        script_dir = folder
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    viewer = STMAFMEntity(script_dir)
    viewer.plot()

    root.mainloop()