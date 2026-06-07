# spectra_window.py
# -*- coding: utf-8 -*-
"""
Spectra Pre-Processing Window
==============================
Displays all spectra associated with a clicked image region.  Provides
interactive controls for choosing X/Y channels, selecting an index sub-range,
applying Gaussian low-pass smoothing, toggling the legend, and launching
specialist analysis windows (dI/dV, KPFS, force spectroscopy).

Intended to be called from main_viewer.py via ``open_spectra_window``.
"""

import copy

import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RangeSlider, TextBox

matplotlib.use("TkAgg")

from dIdV_window import dIdV_window
from force_spectroscopy_window import force_spectroscopy_window
from kpfs_window import kpfs_window
from utilities import export_to_csv, low_pass, pan_factory, zoom_factory

plt.rcParams.update({'font.size': 12})


def open_spectra_window(spectra_list: list[dict], click_pos: tuple) -> None:
    """Open the interactive spectra pre-processing window.

    Parameters
    ----------
    spectra_list:
        List of spectrum dicts (each with ``'data'``, ``'filename'``,
        ``'offset'``, ``'height'``, ``'rotated'`` keys) for the spectra
        near the clicked position.
    click_pos:
        (x, y) coordinates of the click in data units (ångström) — passed
        through but not currently used inside this window.
    """
    root = tk.Tk()
    root.withdraw()

    spectra_processed = copy.deepcopy(spectra_list)
    smooth_state = {'level': 0}
    current_filtered_y = [None]

    def on_close(event) -> None:
        plt.close(fig)

    def update_plot(x_col: str, y_col: str, index_min: int = 0, index_max: int = None) -> None:
        ax_plot.clear()
        for spec in spectra_processed:
            data = spec['data']
            if x_col in data.columns and y_col in data.columns:
                x_data = data[x_col]
                y_data = data[y_col]
                if index_max is None:
                    index_max = len(x_data)
                ax_plot.plot(
                    x_data.iloc[index_min:index_max],
                    y_data.iloc[index_min:index_max],
                    label=spec['filename'],
                    color=spec['color'],
                    linewidth=1.5,
                )
        ax_plot.set_xlabel(x_col)
        ax_plot.set_ylabel(y_col)
        ax_plot.set_title("Spectra Pre-Processing")
        if legend_visible[0]:
            leg = ax_plot.legend(loc='best', fontsize='small')
            leg.set_draggable(True)
        ax_plot.grid(True)
        fig.canvas.draw_idle()

    def make_callback(lbl: str):
        """Return the button callback for the given *lbl*."""
        def callback(event) -> None:
            index_range = tuple(map(int, index_slider.val))
            if lbl == "Low-pass":
                if current_y not in sample_data.columns:
                    return
                # Reset smoothing counter when Y channel changed
                if current_filtered_y[0] != current_y:
                    smooth_state['level'] = 0
                    current_filtered_y[0] = current_y
                smooth_state['level'] += 1
                low_pass(spectra_processed, current_y, smooth_state['level'])
                update_plot(current_x, current_y, *current_index_range)

            elif lbl == "dI/dV":
                dIdV_window(spectra_processed, index_range)
                fig.canvas.mpl_connect('close_event', on_close)
                root.mainloop()

            elif lbl == "KPFS":
                kpfs_window(spectra_processed, index_range)
                fig.canvas.mpl_connect('close_event', on_close)
                root.mainloop()

            elif lbl == "Freq Spec":
                force_spectroscopy_window(spectra_processed, index_range)
                fig.canvas.mpl_connect('close_event', on_close)
                root.mainloop()

            elif lbl == "Export to CSV":
                export_to_csv(ax_plot, current_x, current_y)

        return callback

    # --- Data setup ---
    sample_data = spectra_list[0]['data']
    columns = list(sample_data.columns)

    default_x = 'index' if 'index' in columns else columns[0]
    default_y = 'df' if 'df' in columns else columns[1]

    current_x = default_x
    current_y = default_y
    index_max_initial = len(sample_data)
    current_index_range = [0, index_max_initial]

    # --- Figure layout ---
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    fig.canvas.manager.window.wm_geometry("+200+40")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 5, 1], height_ratios=[12, 1])

    ax_plot = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, 1])

    # Fine-tune axes positions
    pos_plot = ax_plot.get_position()
    ax_plot.set_position([
        pos_plot.x0 + 0.01,
        pos_plot.y0,
        pos_plot.width * 1.1,
        pos_plot.height,
    ])
    pos_slider = ax_slider.get_position()
    ax_slider.set_position([
        pos_slider.x0 + 0.01,
        pos_slider.y0 - 0.05,
        pos_slider.width * 1.1,
        pos_slider.height,
    ])

    # --- Button list (conditionally filtered) ---
    button_labels = ["Low-pass", "dI/dV", "KPFS", "Freq Spec", "Export to CSV"]
    if 'df' not in columns:
        button_labels.remove("KPFS")
        button_labels.remove("Freq Spec")
    elif 'Lock-in X' not in columns:
        button_labels.remove("dI/dV")

    buttons = {}
    # Note: loop variable intentionally reuses `button_labels` name (preserved behaviour)
    for i, button_labels in enumerate(button_labels):
        button_ax = fig.add_axes([0.85, 0.75 - i * 0.1, 0.12, 0.05])
        btn = Button(button_ax, button_labels)
        btn.on_clicked(make_callback(button_labels))
        buttons[button_labels] = btn

    # --- Legend toggle ---
    legend_visible = [len(spectra_list) < 5]

    def toggle_legend(label) -> None:
        legend_visible[0] = not legend_visible[0]
        if legend_visible[0]:
            leg = ax_plot.legend(loc='best', fontsize='small')
            leg.set_draggable(True)
        else:
            leg = ax_plot.get_legend()
            if leg:
                leg.remove()
        fig.canvas.draw_idle()

    check_ax = fig.add_axes([0.83, 0.825, 0.12, 0.08])
    check_ax.set_frame_on(False)
    legend_check = CheckButtons(check_ax, ["Show Legend"], legend_visible)
    legend_check.on_clicked(toggle_legend)

    # --- Initial plot ---
    update_plot(current_x, current_y, *current_index_range)

    # --- Radio buttons ---
    ax_x_radio = fig.add_axes([0.012, 0.6, 0.185, 0.3])
    x_radio = RadioButtons(ax_x_radio, columns, active=columns.index(default_x))
    ax_x_radio.set_title("X Variable")

    ax_y_radio = fig.add_axes([0.012, 0.15, 0.185, 0.3])
    y_radio = RadioButtons(ax_y_radio, columns, active=columns.index(default_y))
    ax_y_radio.set_title("Y Variable")

    # --- Range slider ---
    index_slider = RangeSlider(
        ax_slider,
        "",
        0,
        index_max_initial,
        valinit=(0, index_max_initial),
        valstep=1,
    )
    index_slider.valtext.set_visible(False)

    # --- Min/max textboxes ---
    ax_min_box = fig.add_axes([
        ax_slider.get_position().x0 - 0.07,
        ax_slider.get_position().y0,
        0.06, 0.05,
    ])
    ax_max_box = fig.add_axes([
        ax_slider.get_position().x1 + 0.01,
        ax_slider.get_position().y0,
        0.06, 0.05,
    ])
    textbox_min = TextBox(ax_min_box, "", initial=str(current_index_range[0]))
    textbox_max = TextBox(ax_max_box, "", initial=str(current_index_range[1]))
    ax_min_box.text(0.5, -0.65, "Min", transform=ax_min_box.transAxes, ha="center", va="bottom")
    ax_max_box.text(0.5, -0.65, "Max", transform=ax_max_box.transAxes, ha="center", va="bottom")

    # --- Half/full range toggle button ---
    range_mode = [0]  # 0 = full, 1 = first half, 2 = second half
    ax_half_button = fig.add_axes([0.85, 0.14, 0.12, 0.05])
    half_button = Button(ax_half_button, "Full range")

    def toggle_range(event) -> None:
        total_min = 0
        total_max = index_max_initial
        mid = (total_min + total_max) // 2

        if range_mode[0] == 0:
            new_range = (total_min, mid)
            half_button.label.set_text("First half")
            range_mode[0] = 1
        elif range_mode[0] == 1:
            new_range = (mid + 1, total_max)
            half_button.label.set_text("Second half")
            range_mode[0] = 2
        else:
            new_range = (total_min, total_max)
            half_button.label.set_text("Full range")
            range_mode[0] = 0

        index_slider.set_val(new_range)

    # --- Slider / radio / textbox callbacks ---
    def on_index_slider_change(val) -> None:
        nonlocal current_index_range
        vmin, vmax = map(int, val)
        current_index_range[0] = vmin
        current_index_range[1] = vmax
        if textbox_min.text != str(vmin):
            textbox_min.set_val(str(vmin))
        if textbox_max.text != str(vmax):
            textbox_max.set_val(str(vmax))
        update_plot(current_x, current_y, vmin, vmax)

    def on_x_change(label: str) -> None:
        nonlocal current_x
        current_x = label
        update_plot(current_x, current_y, *current_index_range)

    def on_y_change(label: str) -> None:
        nonlocal current_y
        current_y = label
        smooth_state['level'] = 0
        current_filtered_y[0] = None
        for i in range(len(spectra_processed)):
            spectra_processed[i]['data'] = spectra_list[i]['data'].copy()
        update_plot(current_x, current_y, *current_index_range)

    def on_min_submit(text: str) -> None:
        try:
            vmin = max(int(index_slider.valmin), min(int(text), int(index_slider.val[1])))
            index_slider.set_val((vmin, int(index_slider.val[1])))
        except ValueError:
            textbox_min.set_val(str(int(index_slider.val[0])))

    def on_max_submit(text: str) -> None:
        try:
            vmax = min(int(index_slider.valmax), max(int(text), int(index_slider.val[0])))
            index_slider.set_val((int(index_slider.val[0]), vmax))
        except ValueError:
            textbox_max.set_val(str(int(index_slider.val[1])))

    # --- Connect all callbacks ---
    zoom_factory(ax_plot)
    pan_factory(ax_plot, button=2)
    index_slider.on_changed(on_index_slider_change)
    x_radio.on_clicked(on_x_change)
    y_radio.on_clicked(on_y_change)
    textbox_min.on_submit(on_min_submit)
    textbox_max.on_submit(on_max_submit)
    half_button.on_clicked(toggle_range)

    plt.show(block=False)
    fig.canvas.mpl_connect('close_event', on_close)
    root.mainloop()
