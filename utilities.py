# utilities.py
# -*- coding: utf-8 -*-
"""
Shared utility functions for the STM/AFM interactive viewer suite.

Provides:
    - Matplotlib zoom and pan helpers (``zoom_factory``, ``pan_factory``).
    - Tkinter-based colourmap selector (``create_colormap_menu``).
    - Directory-picker menu entry (``create_load_directory_menu``).
    - CSV export helpers (``export_to_csv``, ``export_grid_csv``).
    - Grid-building routines for 2-D map plots
      (``extract_data_RvsZplot``, ``extract_data_RvsZ_Vstar_plot``,
      ``extract_data_XvsYplot``).
    - Smoothing filters (``low_pass``, ``svgol_low_pass``).
"""

import ast
import re
from tkinter import filedialog
from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                                                                              ║
║               _   _           _  __   ___                  _                 ║
║              | | | |         | | \ \ / / |                | |                ║    
║              | | | | ___ _ __| |_ \ V /| |_ _ __ __ _  ___| |_               ║
║              | | | |/ _ \ '__| __|/   \| __| '__/ _` |/ __| __|              ║
║              \ \_/ /  __/ |  | |_/ /^\ \ |_| | | (_| | (__| |_               ║
║               \___/ \___|_|   \__\/   \/\__|_|  \__,_|\___|\__|              ║
║                                                                              ║
║                                                                              ║
║            Interactive STM/AFM Spectra Extraction & Visualization            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Authors │ Ricardo Ruvalcaba, Thalis Stavridis, Shaoxian Li, Shadi Fatayer   ║
║  Group   │ Manipulation Of NAnosystems (MONA) group                          ║
║  Inst.   │ King Abdullah University of Science and Technology (KAUST)        ║
║          │ Thuwal, Saudi Arabia                                              ║
║  Contact │ ricardo.ruvalcaba.briones@gmail.com                               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  If VertXtract contributes to your research, please cite:                    ║
║                                                                              ║
║    R. Ruvalcaba, T. Stavridis, S. Li, S. Fatayer, (2026). VertXtract:        ║
║    Interactive STM/AFM Data Viewer. Journal_Placeholder.                     ║
║    DOI: DOI_PLACEHOLDER                                                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Keyboard shortcuts                                                          ║
║      Scroll → zoom                   │     Middle-click drag → pan           ║
║      Right-click → switch channel    │     Left-click → open spectra window  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# ---------------------------------------------------------------------------
# Zoom / pan
# ---------------------------------------------------------------------------

def zoom_factory(ax, base_scale: float = 1.05) -> None:
    """Attach a scroll-wheel zoom handler to *ax*.

    Zooming is centred on the current mouse position.

    Parameters
    ----------
    ax:
        The Matplotlib axes to make zoomable.
    base_scale:
        Zoom factor per scroll tick (default 1.05 = 5 % per tick).
    """
    def _zoom(event) -> None:
        if event.inaxes != ax:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        scale_factor = base_scale if event.button == 'down' else 1 / base_scale
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
        ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', _zoom)


def pan_factory(ax, button: int = 2) -> None:
    """Attach a middle-button (or *button*-button) pan handler to *ax*.

    Parameters
    ----------
    ax:
        The Matplotlib axes to make pannable.
    button:
        Mouse button number that activates panning (default 2 = middle).
    """
    pan_active = False
    pan_start = None
    pan_xlim = None
    pan_ylim = None

    def _on_press(event) -> None:
        nonlocal pan_active, pan_start, pan_xlim, pan_ylim
        if event.button != button or event.inaxes != ax:
            return
        pan_active = True
        pan_start = (event.x, event.y)
        pan_xlim = ax.get_xlim()
        pan_ylim = ax.get_ylim()

    def _on_motion(event) -> None:
        nonlocal pan_active
        if not pan_active:
            return
        dx = event.x - pan_start[0]
        dy = event.y - pan_start[1]
        inv = ax.transData.inverted()
        p0 = inv.transform((0, 0))
        p1 = inv.transform((dx, dy))
        dx_data = p1[0] - p0[0]
        dy_data = p1[1] - p0[1]
        ax.set_xlim(pan_xlim[0] - dx_data, pan_xlim[1] - dx_data)
        ax.set_ylim(pan_ylim[0] - dy_data, pan_ylim[1] - dy_data)
        ax.figure.canvas.draw_idle()

    def _on_release(event) -> None:
        nonlocal pan_active
        pan_active = False

    ax.figure.canvas.mpl_connect("button_press_event", _on_press)
    ax.figure.canvas.mpl_connect("motion_notify_event", _on_motion)
    ax.figure.canvas.mpl_connect("button_release_event", _on_release)


# ---------------------------------------------------------------------------
# Directory picker menu entry
# ---------------------------------------------------------------------------

def create_load_directory_menu(
    fig,
    on_directory_selected: Callable[[str], None],
) -> None:
    """Add an 'Open directory' entry to the figure's window menu bar.

    Opens the system file-explorer directory picker when clicked.  The chosen
    path is passed to *on_directory_selected*, which the caller uses to load
    data into the viewer.

    Call this **before** :func:`create_colormap_menu` so the 'Open directory'
    entry appears to the left of 'Colormap' in the menu bar.

    Parameters
    ----------
    fig:
        Matplotlib figure whose Tk window will receive the menu entry.
    on_directory_selected:
        Callable that receives the absolute path of the chosen directory as a
        ``str``.  Called only when the user picks a directory (not on cancel).
    """
    tk_window = fig.canvas.manager.window

    def _pick_directory() -> None:
        folder = filedialog.askdirectory(
            title="Select experiment folder",
            mustexist=True,
        )
        if folder:
            on_directory_selected(folder)

    # Retrieve or create the shared menu bar so both menu functions can add
    # entries to the same bar without overwriting each other.
    menu_bar = tk_window.cget("menu") or ""
    if menu_bar:
        # A menu bar already exists (created by create_colormap_menu first).
        # This path is unlikely given call-order convention, but handled safely.
        menu_bar = tk_window.nametowidget(menu_bar)
    else:
        menu_bar = tk.Menu(tk_window)
        tk_window.config(menu=menu_bar)

    menu_bar.add_command(label="Open directory", command=_pick_directory)


# ---------------------------------------------------------------------------
# Colourmap selector
# ---------------------------------------------------------------------------

def create_colormap_menu(fig, pcm, current_cmap: tk.StringVar) -> None:
    """Add a 'Colormap (loads slowly...)' entry to the figure's window menu bar.

    Opens a scrollable Tk window showing a preview strip for every registered
    Matplotlib colourmap.  Clicking a strip applies that colourmap to *pcm*
    and redraws *fig*.

    Call this **after** :func:`create_load_directory_menu` so the colormap
    entry appears to the right of 'Open directory' in the menu bar.

    Parameters
    ----------
    fig:
        Matplotlib figure whose Tk window will receive the menu entry.
    pcm:
        Any Matplotlib mappable (e.g. ``AxesImage``, ``QuadMesh``) whose
        colourmap will be updated when the user selects one.
    current_cmap:
        ``tk.StringVar`` that tracks the currently active colourmap name.
    """
    gradient = np.linspace(0, 1, 64).reshape(1, -1)
    tk_window = fig.canvas.manager.window

    def _open_selector() -> None:
        win = tk.Toplevel(tk_window)
        win.title("Select colormap")
        win.geometry("320x500")

        container = tk.Frame(win)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        scroll_frame = tk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        preview_images = {}

        def _select_cmap(name: str) -> None:
            current_cmap.set(name)
            pcm.set_cmap(name)
            fig.canvas.draw_idle()

        cmaps = sorted(matplotlib.colormaps(), key=str.lower)
        for cmap_name in cmaps:
            row = tk.Frame(scroll_frame)
            row.pack(fill="x", padx=4, pady=2)

            preview_fig = Figure(figsize=(1.8, 0.18), dpi=100)
            preview_ax = preview_fig.add_axes([0, 0, 1, 1])
            preview_ax.imshow(gradient, aspect="auto", cmap=cmap_name)
            preview_ax.axis("off")

            canvas_preview = FigureCanvasTkAgg(preview_fig, master=row)
            widget = canvas_preview.get_tk_widget()
            widget.pack(side="left")

            lbl = tk.Label(row, text=cmap_name, anchor="w", width=18)
            lbl.pack(side="left", padx=5)

            for w in (row, widget, lbl):
                w.bind("<Button-1>", lambda e, name=cmap_name: _select_cmap(name))
            preview_images[cmap_name] = canvas_preview

        def _on_mousewheel(event) -> None:
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-event.delta / 120), "units")

        def _bind_mousewheel_recursive(widget) -> None:
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
            for child in widget.winfo_children():
                _bind_mousewheel_recursive(child)

        _bind_mousewheel_recursive(win)

    # Retrieve or create the shared menu bar.
    menu_bar = tk_window.cget("menu") or ""
    if menu_bar:
        menu_bar = tk_window.nametowidget(menu_bar)
    else:
        menu_bar = tk.Menu(tk_window)
        tk_window.config(menu=menu_bar)

    menu_bar.add_command(label="Colormap (loads slowly...)", command=_open_selector)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_to_csv(ax, x_label: str, y_label: str) -> None:
    """Export all lines currently plotted on *ax* to a user-chosen CSV file.

    Each line contributes two columns named ``<label>-<x_label>`` and
    ``<label>-<y_label>``.

    Parameters
    ----------
    ax:
        Matplotlib axes whose lines will be exported.
    x_label:
        Column-name suffix for x data.
    y_label:
        Column-name suffix for y data.
    """
    lines = ax.get_lines()
    if not lines:
        print("No lines to export.")
        return

    export_data = {}
    for line in lines:
        label = line.get_label()
        export_data[f"{label}-{x_label}"] = pd.Series(line.get_xdata())
        export_data[f"{label}-{y_label}"] = pd.Series(line.get_ydata())

    df = pd.concat(export_data, axis=1)
    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if filepath:
        df.to_csv(filepath, index=False)
        print(f"Exported to {filepath}")


def export_grid_csv(
    grid: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
) -> None:
    """Export a 2-D data grid to a user-chosen CSV file.

    Parameters
    ----------
    grid:
        2-D array of values (rows = y, columns = x).
    x_centers, y_centers:
        Coordinate tick values used as column / row labels.
    x_label, y_label:
        Axis name strings written as index/column name in the CSV.
    """
    df = pd.DataFrame(grid, index=y_centers, columns=x_centers).fillna("")
    df.index.name = y_label
    df.columns.name = x_label

    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if filepath:
        df.to_csv(filepath)
        print(f"Grid exported to {filepath}")


# ---------------------------------------------------------------------------
# Grid-building helpers (shared by kpfs_window and related plots)
# ---------------------------------------------------------------------------

def _edges_from_centers(vals: np.ndarray) -> np.ndarray:
    """Return bin edges computed from an array of bin centres.

    For a single-element array a ±0.5 interval is used.

    Parameters
    ----------
    vals:
        1-D sorted array of bin centre values.

    Returns
    -------
    np.ndarray
        Array of length ``len(vals) + 1``.
    """
    if len(vals) > 1:
        mids = (vals[:-1] + vals[1:]) / 2
        return np.concatenate((
            [vals[0] - (mids[0] - vals[0])],
            mids,
            [vals[-1] + (vals[-1] - mids[-1])],
        ))
    d = 0.5
    return np.array([vals[0] - d, vals[0] + d])


def extract_data_RvsZplot(
    plots_info: list[dict],
    get_termination_number,
    separation: float = 0.25,
) -> tuple:
    """Build a 2-D LCPD grid indexed by lateral displacement and tip height.

    Used by the KPFS window to render the 'LCPD X vs Z' map.

    Parameters
    ----------
    plots_info:
        List of plot-info dicts, each containing ``'label'``, ``'line'``,
        ``'x_rot'``, and ``'y_rot'`` keys.
    get_termination_number:
        Callable that extracts an integer termination index from a label string.
    separation:
        Unused spacing parameter (kept for API compatibility).

    Returns
    -------
    tuple
        ``(x_edges, y_edges, grid, displacement_centers, height_centers)``
    """
    term_data = []

    for plot in plots_info:
        term_num = get_termination_number(plot['label'])
        if term_num is None:
            continue

        lcpd_match = re.search(r"LCPD=(-?\d+\.?\d*)", plot['line'].get_label())
        if not lcpd_match:
            continue
        lcpd_value = float(lcpd_match.group(1))

        location_match = re.search(r'Location=(\[[^\]]+\])', plot['label'])
        if not location_match:
            continue
        location = ast.literal_eval(location_match.group(1))
        height = float(location[2])

        term_data.append((
            term_num,
            height,
            lcpd_value,
            plot.get('x_rot', np.nan),
            plot.get('y_rot', np.nan),
        ))

    if not term_data:
        raise RuntimeError("No valid LCPD data found")

    term_data = np.array(term_data, dtype=float)

    unique_terms = np.unique(term_data[:, 0])
    unique_terms.sort()
    first_term = unique_terms[0]
    last_term = unique_terms[-1]

    avg_first_xy = np.nanmean(term_data[term_data[:, 0] == first_term][:, 3:5], axis=0)
    avg_last_xy = np.nanmean(term_data[term_data[:, 0] == last_term][:, 3:5], axis=0)

    displacement_distance = np.linalg.norm(avg_last_xy - avg_first_xy)
    displacement_centers = np.linspace(0, displacement_distance, len(unique_terms))

    height_centers = np.sort(np.unique(term_data[:, 1]))
    term_centers = np.sort(np.unique(term_data[:, 0]))

    height_to_i = {h: i for i, h in enumerate(height_centers)}
    term_to_j = {t: j for j, t in enumerate(term_centers)}

    grid = np.full((len(height_centers), len(term_centers)), np.nan)
    for term, height, value, *_ in term_data:
        grid[height_to_i[height], term_to_j[term]] = value

    x_edges = _edges_from_centers(displacement_centers)
    y_edges = _edges_from_centers(height_centers)

    return x_edges, y_edges, grid, displacement_centers, height_centers


def extract_data_RvsZ_Vstar_plot(
    plots_info: list[dict],
    get_termination_number,
    separation: float = 0.25,
) -> tuple:
    """Build a 2-D df* grid indexed by lateral displacement and tip height.

    Identical structure to :func:`extract_data_RvsZplot` but reads
    ``df*=`` from the line label instead of ``LCPD=``.

    Parameters
    ----------
    plots_info:
        List of plot-info dicts.
    get_termination_number:
        Callable that extracts an integer termination index from a label string.
    separation:
        Unused spacing parameter (kept for API compatibility).

    Returns
    -------
    tuple
        ``(x_edges, y_edges, grid, displacement_centers, height_centers)``
    """
    term_data = []

    for plot in plots_info:
        term_num = get_termination_number(plot['label'])
        if term_num is None:
            continue

        match = re.search(r"df\*=(-?\d+\.?\d*)", plot['line'].get_label())
        if not match:
            continue
        lcpd_value = float(match.group(1))

        location_match = re.search(r'Location=(\[[^\]]+\])', plot['label'])
        if not location_match:
            continue
        location = ast.literal_eval(location_match.group(1))
        height = float(location[2])

        term_data.append((
            term_num,
            height,
            lcpd_value,
            plot.get('x_rot', np.nan),
            plot.get('y_rot', np.nan),
        ))

    if not term_data:
        raise RuntimeError("No valid LCPD data found")

    term_data = np.array(term_data, dtype=float)

    unique_terms = np.unique(term_data[:, 0])
    unique_terms.sort()
    first_term = unique_terms[0]
    last_term = unique_terms[-1]

    avg_first_xy = np.nanmean(term_data[term_data[:, 0] == first_term][:, 3:5], axis=0)
    avg_last_xy = np.nanmean(term_data[term_data[:, 0] == last_term][:, 3:5], axis=0)

    displacement_distance = np.linalg.norm(avg_last_xy - avg_first_xy)
    displacement_centers = np.linspace(0, displacement_distance, len(unique_terms))

    height_centers = np.sort(np.unique(term_data[:, 1]))
    term_centers = np.sort(np.unique(term_data[:, 0]))

    height_to_i = {h: i for i, h in enumerate(height_centers)}
    term_to_j = {t: j for j, t in enumerate(term_centers)}

    grid = np.full((len(height_centers), len(term_centers)), np.nan)
    for term, height, value, *_ in term_data:
        grid[height_to_i[height], term_to_j[term]] = value

    x_edges = _edges_from_centers(displacement_centers)
    y_edges = _edges_from_centers(height_centers)

    return x_edges, y_edges, grid, displacement_centers, height_centers


def extract_data_XvsYplot(plots_info: list[dict]) -> tuple:
    """Build a 2-D LCPD grid indexed by lateral X and Y position.

    Used by the KPFS window to render the 'LCPD X vs Y' map.

    Parameters
    ----------
    plots_info:
        List of plot-info dicts, each containing ``'label'`` and ``'line'``.

    Returns
    -------
    tuple
        ``(x_edges, y_edges, grid, x_centers, y_centers)``
    """
    data = []

    for plot in plots_info:
        lcpd_match = re.search(r"LCPD=(-?\d+\.?\d*)", plot['line'].get_label())
        lcpd_value = float(lcpd_match.group(1))

        location_match = re.search(r'Location=(\[[^\]]+\])', plot['label'])
        location = ast.literal_eval(location_match.group(1))

        data.append((float(location[0]), float(location[1]), lcpd_value))

    if not data:
        raise RuntimeError("No valid LCPD data found")

    data = np.array(data, dtype=float)
    x_vals, y_vals, values = data[:, 0], data[:, 1], data[:, 2]

    x_centers = np.sort(np.unique(x_vals))
    y_centers = np.sort(np.unique(y_vals))

    x_to_i = {x: i for i, x in enumerate(x_centers)}
    y_to_j = {y: j for j, y in enumerate(y_centers)}

    grid = np.full((len(y_centers), len(x_centers)), np.nan)
    for x, y, value in data:
        grid[y_to_j[y], x_to_i[x]] = value
    grid = np.flipud(grid)

    x_edges = _edges_from_centers(x_centers)
    y_edges = _edges_from_centers(y_centers)

    return x_edges, y_edges, grid, x_centers, y_centers


# ---------------------------------------------------------------------------
# Smoothing filters
# ---------------------------------------------------------------------------

def svgol_low_pass(spectra_list: list[dict], column: str, level: int) -> None:
    """Apply a Savitzky-Golay low-pass filter to *column* in all spectra.

    Window length grows by 2 per *level* step (5, 7, 9, …).
    Polynomial order is fixed at 3.

    Parameters
    ----------
    spectra_list:
        List of spectrum dicts, each with a ``'data'`` DataFrame.
    column:
        Column name to filter in-place.
    level:
        Cumulative smoothing level (determines window size).
    """
    window = 5 + 2 * level
    poly = 3
    for spec in spectra_list:
        data = spec['data']
        if column in data.columns:
            y = data[column].values
            if window < len(y):
                data[column] = savgol_filter(y, window, poly)


def low_pass(spectra_list: list[dict], column: str, level: int) -> None:
    """Apply a Gaussian low-pass filter to *column* in all spectra.

    Sigma grows by 0.5 per *level* step.

    Parameters
    ----------
    spectra_list:
        List of spectrum dicts, each with a ``'data'`` DataFrame.
    column:
        Column name to filter in-place.
    level:
        Cumulative smoothing level (determines sigma).
    """
    sigma = 0.5 + level * 0.5
    for spec in spectra_list:
        data = spec['data']
        if column in data.columns:
            y = data[column].values
            if len(y) > 3:
                data[column] = gaussian_filter1d(y, sigma=sigma)


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------

def print_welcoming_message() -> None:
    """Print a formatted welcome banner to stdout when VertXtract is launched."""
    print(BANNER)
