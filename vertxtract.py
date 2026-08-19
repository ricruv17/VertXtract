# vertxtract.py
# -*- coding: utf-8 -*-
"""
Interactive STM/AFM Data Viewer
================================
Provides the graphical interface for exploring Createc STM/AFM data.
All file I/O and calibration logic lives in read_files.py; this module
is responsible solely for the Matplotlib / Tkinter GUI.

``STMAFMEntity`` inherits from ``STMAFMReader`` (read_files.py) and adds
interactive plotting on top of the loaded data.

On startup the viewer opens a blank figure with a prompt to choose a folder.
Data is loaded only after the user selects a directory via the
'Open directory' menu entry, so the window is always ready before any I/O
begins.  A second call to 'Open directory' reloads a new folder into the
same window without reopening it.

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

sys.path.insert(
    0,
    r'C:\Users\ruvalcrm\OneDrive - KAUST\Desktop\PC\scripts\STMAFM\VERTXtract',
)
from spectra_window import open_spectra_window  # noqa: E402
from utilities import (  # noqa: E402
    BANNER,
    create_colormap_menu,
    create_load_directory_menu,
    pan_factory,
    print_welcoming_message,
    zoom_factory,
)
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

# Accept an optional folder path from the command line.
dpath = False
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = dpath

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

    The viewer opens immediately with a blank canvas.  Data is loaded (and the
    canvas populated) only when :meth:`_reload` is called, either from the
    'Open directory' menu or from a command-line path supplied at startup.

    Parameters
    ----------
    folder_path:
        Optional absolute path to pre-load on startup.  Pass ``None`` (the
        default) to open a blank figure and wait for the user to choose a
        directory via the menu.
    """

    def __init__(self, folder_path: Optional[str] = None) -> None:
        # Interaction state
        self.selection_radius: float = SELECTION_RADIUS_DEFAULT
        self.selection_circle: Optional[Circle] = None
        self.slider_active: bool = False
        self.last_click: Optional[tuple[float, float]] = None

        # Colormap state (tk.StringVar requires an existing Tk root)
        self.current_cmap: tk.StringVar = tk.StringVar(value="gray")

        # Data-loaded flag — gates _draw_data() so it's safe to call anytime
        self._data_loaded: bool = False

        # Initialise the parent without loading any files.
        # STMAFMReader.__init__ with folder_path=None skips all I/O.
        super().__init__(folder_path=None)

        # Build the figure immediately (blank canvas).
        self._build_figure()

        # If a path was supplied on the command line, load it right away.
        if folder_path:
            self._reload(folder_path)

    # ------------------------------------------------------------------
    # Figure construction (called once at startup)
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        """Create the Matplotlib figure and attach all permanent UI elements.

        This is called once during ``__init__``.  The axes start blank; data
        is drawn later by :meth:`_draw_data` once a directory is loaded.
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.get_current_fig_manager().window.wm_geometry("+0+0")

        # Placeholder: axes are empty until a directory is loaded.
        self.ax.set_title("VertXtract: Open a directory to begin")
        self.ax.set_xlabel("X (Å)")
        self.ax.set_ylabel("Y (Å)")
        self._placeholder_text = self.ax.text(
            0.03, 0.5,
            BANNER,
            transform=self.ax.transAxes,
            ha='left', va='center',
            fontsize=7, color='gray',
            family='monospace',
        )

        def _fit_banner_fontsize(event=None) -> None:
            """Resize the banner font so it fills the axes on every window resize.

            Compares the text bounding box to the axes bounding box in display
            pixels and scales the font size by the smaller of the two ratios
            (width-fit vs height-fit) so the text always fits without clipping.
            """
            if self._data_loaded or self._placeholder_text is None:
                return
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()
            text_bbox = self._placeholder_text.get_window_extent(renderer=renderer)
            axes_bbox = self.ax.get_window_extent(renderer=renderer)
            if text_bbox.width == 0 or text_bbox.height == 0:
                return
            margin = 0.92
            scale = min(
                (axes_bbox.width  * margin) / text_bbox.width,
                (axes_bbox.height * margin) / text_bbox.height,
            )*1.12
            new_size = max(4.0, min(self._placeholder_text.get_fontsize() * scale, 48.0))
            self._placeholder_text.set_fontsize(new_size)
            self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('resize_event', _fit_banner_fontsize)
        self._fit_banner_fontsize = _fit_banner_fontsize

        # Dummy 1×1 grey image so imshow / colorbar can be initialised now.
        # It is replaced by real data in _draw_data().
        dummy = np.zeros((1, 1))
        self.im = self.ax.imshow(
            dummy,
            cmap=self.current_cmap.get(),
            origin='lower',
            aspect='equal',
        )
        self.im.set_visible(False)
        self.cbar = plt.colorbar(self.im, ax=self.ax, label="")
        self.cbar.ax.set_visible(False)

        # Menu bar — 'Open directory' left, 'Colormap' right.
        # create_load_directory_menu must be called first so its entry
        # appears to the left of the colormap entry.
        create_load_directory_menu(self.fig, self._reload)
        create_colormap_menu(self.fig, self.im, self.current_cmap)

        # Zoom / pan always active (even on blank canvas)
        zoom_factory(self.ax)
        pan_factory(self.ax, button=2)

        # Selection-radius slider (always present)
        self._add_slider()

        # Mouse callbacks
        self.fig.canvas.mpl_connect('button_press_event', self._show_context_menu)
        self.fig.canvas.mpl_connect('button_press_event', self._handle_click)

        def _on_close(event) -> None:  # noqa: ANN001
            plt.close('all')
            root.quit()
            root.destroy()
            sys.exit(0)

        self.fig.canvas.mpl_connect('close_event', _on_close)

    # ------------------------------------------------------------------
    # Data loading / reloading
    # ------------------------------------------------------------------

    def _reload(self, folder_path: str) -> None:
        """Load (or reload) data from *folder_path* and redraw the figure.

        Safe to call multiple times — each call replaces the previous dataset
        without closing or reopening the window.

        Parameters
        ----------
        folder_path:
            Absolute path to the experiment folder to load.
        """
        print(f"[INFO] Loading folder: {folder_path}")
        try:
            # Delegate all file I/O to the parent's load() method.
            self.load(folder_path)
        except Exception as e:
            print(f"[ERROR] Could not load '{folder_path}': {e}")
            return

        self._data_loaded = True
        self._draw_data()

    def _draw_data(self) -> None:
        """Populate the axes with the currently loaded image and spectra.

        Called by :meth:`_reload` after a successful data load.  Clears any
        previous image/scatter artists before drawing the new ones.
        """
        # --- Remove placeholder text ---
        if self._placeholder_text is not None:
            try:
                self._placeholder_text.remove()
            except ValueError:
                pass
            self._placeholder_text = None

        # --- Clear previous scatter artists (spectra markers + offset dot) ---
        # Keep only the imshow artist; remove everything else.
        for artist in self.ax.collections:
            artist.remove()

        # --- Load and display the image channel ---
        self._load_image_channel(self.first_valid_index)

        extent = [
            self.X0.min(), self.X0.max(),
            self.Y0.min(), self.Y0.max(),
        ]
        self.im.set_data(self.image)
        self.im.set_extent(extent)
        self.im.set_visible(True)

        vmin, vmax = np.percentile(
            self.image,
            [CLIM_PERCENTILE_LOW, CLIM_PERCENTILE_HIGH],
        )
        self.im.set_clim(vmin, vmax)

        channel_label = CHANNEL_LABELS.get(self.current_channel, self.current_channel)
        self.cbar.update_normal(self.im)
        self.cbar.set_label(channel_label)
        self.cbar.ax.set_visible(True)

        # --- Scan offset marker ---
        self.ax.scatter(
            self.offset_A_x, self.offset_A_y,
            facecolors='white', color='black', label='Offset',
        )

        # --- Spectra position markers ---
        for spec in self.spectra:
            x_rot, y_rot = spec['rotated']
            self.ax.scatter(
                x_rot, y_rot,
                facecolor='None',
                edgecolor=spec['color'],
                linewidths=1.5,
            )

        # --- Auto-computed square view extent ---
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

        # y-axis intentionally inverted to match imshow origin='lower'
        # combined with Createc's coordinate convention.
        self.ax.set_xlim(x_center - half_size, x_center + half_size)
        self.ax.set_ylim(y_center + half_size, y_center - half_size)

        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_title(f"Interactive Spectra Processor\n{os.path.basename(self.folder_path)}")

        self.fig.canvas.draw_idle()
        print("[INFO] Figure updated.")

    # ------------------------------------------------------------------
    # GUI — slider
    # ------------------------------------------------------------------

    def _add_slider(self) -> None:
        """Add a logarithmic selection-radius slider below the main axes."""
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
    # GUI — interaction callbacks
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
        """Display a right-click Tk context menu for channel selection."""
        if event.button != 3 or not self._data_loaded:
            return

        menu = tk.Menu(None, tearoff=0)
        for name, idx in self.available_channels:
            menu.add_command(
                label=name,
                command=lambda n=name, i=idx: self._update_channel(n, i),
            )
        menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)

    def _handle_click(self, event: MouseEvent) -> None:
        """Handle left-click: draw selection circle and open nearby spectra."""
        if event.button != 1 or not self._data_loaded:
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
        """Replace the current selection circle with a new one at *center*."""
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
        """Switch the displayed image channel and refresh the figure."""
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

    def plot(self) -> None:
        """Show the figure and start the event loop.

        The figure was already constructed in ``__init__`` via
        :meth:`_build_figure`.  This method makes it visible and triggers an
        initial font-size fit for the banner text.
        """
        print("[INFO] Displaying interactive viewer...")
        plt.show(block=False)
        # Fit the banner to the initial window size now that the renderer
        # has a real pixel extent to measure against.
        self._fit_banner_fontsize()


# ---------------------------------------------------------------------------
# Entry point — STMAFMReader must expose a load() method
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    print_welcoming_message()

    # Pass a pre-selected folder if given on the command line; otherwise None
    # so the viewer opens blank and waits for the user to pick a directory.
    initial_folder = folder if folder else None
    viewer = STMAFMEntity(folder_path=initial_folder)
    viewer.plot()

    root.mainloop()