# dIdV_window.py
# -*- coding: utf-8 -*-
"""
dI/dV Spectra Window
=====================
Displays dI/dV spectra from a list of selected spectrum dicts.  Provides
optional normalisation ((dI/dV)/(I/V)), waterfall-style splitting, and a
2-D voltage-vs-displacement map via ``plot_dIdV_vs_R``.
"""

import tkinter as tk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons

matplotlib.use("TkAgg")

from utilities import create_colormap_menu, pan_factory, zoom_factory

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': False,
    'svg.fonttype': 'none',
})


def dIdV_window(spectra_list: list[dict], index_range: tuple) -> None:
    """Open the interactive dI/dV spectra window.

    Parameters
    ----------
    spectra_list:
        List of spectrum dicts (each with ``'data'``, ``'filename'``,
        ``'offset'``, ``'height'``, ``'rotated'`` keys).
    index_range:
        ``(index_min, index_max)`` slice applied to each spectrum's data.
    """
    root = tk.Tk()
    root.withdraw()

    # Module-level state flags (preserved as globals for callback compatibility)
    global current_cmap, normalized_active, split_active
    current_cmap = tk.StringVar(value="coolwarm")
    normalized_active = False
    split_active = False

    index_min, index_max = index_range

    def on_close(event) -> None:
        plt.close(fig_dIdV)

    def plot_dIdV_vs_R() -> None:
        """Render a 2-D voltage-vs-displacement pcolormesh map."""
        if not plots_info:
            print("No spectra available.")
            return

        coords = np.array([[p['x_rot'], p['y_rot']] for p in plots_info])
        valid = ~np.isnan(coords).any(axis=1)
        coords = coords[valid]
        valid_plots = [plots_info[i] for i in range(len(plots_info)) if valid[i]]

        # Distance along lateral displacement axis relative to first point
        origin = coords[0]
        R = np.linalg.norm(coords - origin, axis=1)
        order = np.argsort(R)
        R = R[order]
        valid_plots = [valid_plots[i] for i in order]

        voltage = (
            valid_plots[0]['x'].values
            if hasattr(valid_plots[0]['x'], 'values')
            else valid_plots[0]['x']
        )

        grid = np.array([
            p['processed_y'].values if hasattr(p['processed_y'], 'values') else p['processed_y']
            for p in valid_plots
        ])

        # Build pcolormesh edges from centres
        voltage_edges = np.concatenate([
            [voltage[0] - (voltage[1] - voltage[0]) / 2],
            (voltage[:-1] + voltage[1:]) / 2,
            [voltage[-1] + (voltage[-1] - voltage[-2]) / 2],
        ])
        R_edges = np.concatenate([
            [R[0] - (R[1] - R[0]) / 2],
            (R[:-1] + R[1:]) / 2,
            [R[-1] + (R[-1] - R[-2]) / 2],
        ])

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.window.wm_geometry("+600+120")

        pcm = ax.pcolormesh(R_edges, voltage_edges, grid.T, shading='auto', cmap=current_cmap.get())
        create_colormap_menu(fig, pcm, current_cmap)

        label = r"(dI/dV)/(I/V)" if normalized_active else "dI/dV"
        fig.colorbar(pcm, ax=ax, label=f'{label} (arb. units)')
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlabel("Displacement (Å)")
        ax.set_title(f"{label}: Voltage vs R map")
        zoom_factory(ax)
        pan_factory(ax, button=2)
        plt.show()

    def normalize_dIdV() -> None:
        global normalized_active
        normalized_active = not normalized_active
        update_display()

    def split_dIdV() -> None:
        global split_active
        split_active = not split_active
        update_display()

    def update_display() -> None:
        """Recompute normalisation and/or split offsets, then redraw."""
        # --- Step 1: (re-)apply normalisation ---
        for p, spec in zip(plots_info, spectra_list):
            y = p['raw_y'].copy()

            if normalized_active:
                data = spec['data']
                if (
                    'Lock-in X' in data.columns
                    and 'Voltage' in data.columns
                ):
                    if ('ADC0' not in data.columns
                        and 'Current(filtered)' in data.columns
                    ):
                        current = data['Current(filtered)'].iloc[index_min:index_max].values
                    else:
                        current = data['ADC0'].iloc[index_min:index_max].values

                    lockin = data['Lock-in X'].iloc[index_min:index_max].values
                    voltage = data['Voltage'].iloc[index_min:index_max].values

                    eps = 1e-3
                    iv = current / (voltage + eps)
                    y = lockin / (iv + eps)
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                    # Zero out near-zero voltage region to avoid division artefacts
                    voltage_eps = 20
                    y[np.abs(voltage) <= voltage_eps] = np.nan

            p['processed_y'] = y.copy()

        # --- Step 2: apply waterfall split if active ---
        if split_active:
            ranges = []
            for p in plots_info:
                finite_y = p['processed_y'][np.isfinite(p['processed_y'])]
                ranges.append(
                    (float(np.min(finite_y)), float(np.max(finite_y)))
                    if len(finite_y) > 0 else (0.0, 0.0)
                )

            global_min = min(r[0] for r in ranges)
            global_max = max(r[1] for r in ranges)
            gap = (global_max - global_min) * 0.05

            current_offset = 0.0
            for p, (y_min, y_max) in zip(plots_info, ranges):
                p['display_y'] = p['processed_y'] - y_max + current_offset
                current_offset -= (y_max - y_min) + gap
        else:
            for p in plots_info:
                p['display_y'] = p['processed_y']

        # --- Step 3: update plot lines ---
        for p in plots_info:
            p['line'].set_ydata(p['display_y'])
        ax_dIdV.relim()
        ax_dIdV.autoscale_view()

        if normalized_active:
            ax_dIdV.set_title("Normalized dI/dV spectra")
            ax_dIdV.set_ylabel(r"(dI/dV)/(I/V) (arb. units)")
        else:
            ax_dIdV.set_title("Raw dI/dV spectra")
            ax_dIdV.set_ylabel("dI/dV (arb. units)")
        fig_dIdV.canvas.draw_idle()

    def make_button_callback(name: str):
        def callback(event) -> None:
            if name == "dI/dV vs R":
                plot_dIdV_vs_R()
            elif name == "Split dI/dV":
                split_dIdV()
            elif name == "Normalize":
                normalize_dIdV()
        return callback

    # --- Figure layout ---
    fig_dIdV = plt.figure(figsize=(10, 8))
    fig_dIdV.canvas.manager.window.wm_geometry("+400+80")
    gs = fig_dIdV.add_gridspec(1, 2, width_ratios=[7, 1], wspace=0.01)

    ax_dIdV = fig_dIdV.add_subplot(gs[0, 0])
    ax_dIdV.set_title("dI/dV Spectra")
    ax_dIdV.set_xlabel("Voltage (mV)")
    ax_dIdV.set_ylabel("dI/dV (arb. units)")
    ax_dIdV.grid(True)

    # --- Plot spectra and build plots_info ---
    plots_info = []
    for spec in spectra_list:
        data = spec['data']
        if 'Voltage' in data.columns and 'Lock-in X' in data.columns:
            x_data = data['Voltage'].iloc[index_min:index_max]
            y_data = data['Lock-in X'].iloc[index_min:index_max]

            if spec.get('height') is not None:
                label = f"{spec['filename']} (Location={spec['offset']} Å)"
            else:
                label = spec['filename']

            line, = ax_dIdV.plot(x_data, y_data, label=label, color=spec['color'], linewidth=1.5)
            x_rot, y_rot = spec.get('rotated', (np.nan, np.nan))

            plots_info.append({
                'x': x_data,
                'y': y_data.values.copy(),
                'raw_y': y_data.values.copy(),
                'processed_y': y_data.values.copy(),
                'display_y': y_data.values.copy(),
                'label': label,
                'line': line,
                'x_rot': x_rot,
                'y_rot': y_rot,
            })

    leg = ax_dIdV.legend(loc='best', fontsize='small')
    leg.set_draggable(True)

    # --- Control buttons ---
    ax_controls = fig_dIdV.add_subplot(gs[0, 1])
    ax_controls.axis('off')

    button_labels = ["dI/dV vs R", "Split dI/dV", "Normalize"]
    buttons = {}
    for i, button_label in enumerate(button_labels):
        button_ax = fig_dIdV.add_axes([0.85, 0.75 - i * 0.1, 0.12, 0.05])
        btn = Button(button_ax, button_label)
        btn.on_clicked(make_button_callback(button_label))
        buttons[button_label] = btn

    # --- Legend toggle ---
    legend_visible = [0]  # hidden by default

    if legend_visible[0]:
        leg = ax_dIdV.legend(loc='best', fontsize='small')
        leg.set_draggable(True)
    else:
        if ax_dIdV.get_legend():
            ax_dIdV.legend_.remove()

    check_ax = fig_dIdV.add_axes([0.83, 0.825, 0.12, 0.08])
    check_ax.set_frame_on(False)
    legend_check = CheckButtons(check_ax, ["Show Legend"], legend_visible)

    def toggle_legend(label) -> None:
        if legend_check.get_status()[0]:
            leg = ax_dIdV.legend(loc='best', fontsize='small')
            leg.set_draggable(True)
        else:
            leg = ax_dIdV.get_legend()
            if leg:
                leg.remove()
        fig_dIdV.canvas.draw_idle()

    zoom_factory(ax_dIdV)
    pan_factory(ax_dIdV, button=2)
    legend_check.on_clicked(toggle_legend)

    plt.show(block=False)
    fig_dIdV.canvas.mpl_connect('close_event', on_close)
    root.mainloop()
