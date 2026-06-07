# force_spectroscopy_window.py
# -*- coding: utf-8 -*-
"""
Force Spectroscopy Window
==========================
Displays df-vs-Z frequency spectra and provides interactive fitting tools:

- **Mie potential fit** — fits ``f(z) = (a/(z0-z))^b - (u/(z0-z))^v`` via
  ``scipy.optimize.curve_fit`` with interactive parameter sliders.
- **Polynomial fit** — fits an adjustable-degree polynomial with a
  degree slider and textbox.

Both sub-windows open as separate Matplotlib figures.
"""

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
from scipy.optimize import curve_fit

from utilities import export_to_csv, pan_factory, zoom_factory

plt.rcParams.update({'font.size': 12})


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def polynomial_fit(x: np.ndarray, y: np.ndarray, degree: int):
    """Return a callable polynomial fit of *degree* through (x, y).

    X values are normalised internally for numerical stability.

    Parameters
    ----------
    x, y:
        Data arrays to fit.
    degree:
        Polynomial degree.

    Returns
    -------
    callable
        Function ``f(x_val)`` that evaluates the fitted polynomial.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_scaled = (x - x_mean) / x_std
    try:
        coeffs = np.polyfit(x_scaled, y, degree)
        poly_scaled = np.poly1d(coeffs)
        return lambda x_val: poly_scaled((x_val - x_mean) / x_std)
    except np.linalg.LinAlgError as e:
        print(f"Error during polynomial fit: {e}")
        return lambda x_val: np.full_like(x_val, np.nan)


def mie_potential(z: np.ndarray, a: float, b: float, u: float, v: float, z_0: float) -> np.ndarray:
    """Evaluate the Mie potential: ``(a/(z0-z))^b - (u/(z0-z))^v``.

    Only evaluated for z < z_0; remaining values are NaN.

    Parameters
    ----------
    z:
        Z-position array (ångström).
    a, b, u, v, z_0:
        Mie potential parameters.

    Returns
    -------
    np.ndarray
        Potential values in Hz (or NaN outside the valid range).
    """
    z = np.asarray(z)
    result = np.full_like(z, np.nan, dtype=np.float64)
    mask = z < z_0
    result[mask] = (a / (z_0 - z[mask])) ** b - (u / (z_0 - z[mask])) ** v
    return result


def fit_potential(x, y, function, parameter_labels, initial_guess, param_bounds):
    """Wrapper around ``scipy.optimize.curve_fit`` with a large maxfev.

    Parameters
    ----------
    x, y:
        Data to fit.
    function:
        Model function ``f(x, *params)``.
    parameter_labels:
        Human-readable names (used for printing, currently commented out).
    initial_guess:
        Initial parameter values.
    param_bounds:
        ``([lower, ...], [upper, ...])`` bounds for each parameter.

    Returns
    -------
    np.ndarray
        Optimal parameter values.
    """
    popt, _ = curve_fit(
        function,
        x,
        y,
        p0=initial_guess,
        bounds=param_bounds,
        maxfev=int(1e8),
    )
    return popt


# ---------------------------------------------------------------------------
# Main window function
# ---------------------------------------------------------------------------

def force_spectroscopy_window(spectra_list: list[dict], index_range: tuple) -> None:
    """Open the interactive force spectroscopy window.

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

    def on_close(event) -> None:
        plt.close(fig_force_spec)
        root.quit()
        root.destroy()

    # ------------------------------------------------------------------
    # Mie potential sub-window
    # ------------------------------------------------------------------

    def fit_mie_potential(event) -> None:
        """Open an interactive Mie potential fitting window for the first spectrum."""
        if not plots_info:
            print("No spectra to fit.")
            return

        plot = plots_info[0]
        x_data = plot['x'].values
        y_data = plot['y'].values

        param_labels = ['a', 'b', 'u', 'v', r'$z_0$']
        param_bounds = ([0, 1, 0, 1, -99], [99, 20, 99, 20, 99])
        init_guess = [0.1, 10, 5, 4, max(x_data) + 0.1]

        fig_miepotential, ax = plt.subplots(figsize=(8, 8))
        fig_miepotential.canvas.manager.window.wm_geometry("+600+120")
        plt.subplots_adjust(left=0.15, bottom=0.45)

        # Z-range textboxes
        ax_text_zmin = plt.axes([0.82, 0.29, 0.08, 0.04])
        ax_text_zmax = plt.axes([0.82, 0.24, 0.08, 0.04])
        textbox_zmin = TextBox(ax_text_zmin, 'z_min', initial=str(round(min(x_data) - 1, 2)))
        textbox_zmax = TextBox(ax_text_zmax, 'z_max', initial=str(round(max(x_data) + 1, 2)))
        textbox_zmin.on_submit(lambda text: update(None))
        textbox_zmax.on_submit(lambda text: update(None))

        # Action buttons
        ax_button_refit = fig_miepotential.add_axes([0.15, 0.02, 0.25, 0.05])
        ax_button_save = fig_miepotential.add_axes([0.7, 0.02, 0.2, 0.05])
        button_refit = Button(ax_button_refit, 'Optimize parameters')
        button_save = Button(ax_button_save, 'Export to CSV')

        def textbox_submit_factory(i: int):
            def submit(text: str) -> None:
                try:
                    val = float(text)
                    val = max(sliders[i].valmin, min(sliders[i].valmax, val))
                    sliders[i].set_val(val)
                except ValueError:
                    print("Invalid parameter value")
            return submit

        def save_callback(event) -> None:
            export_to_csv(ax, 'Z', 'df')
            root.mainloop()

        def refit_callback(event) -> None:
            current_guess = [s.val for s in sliders]
            try:
                popt = fit_potential(
                    x_data, y_data, mie_potential,
                    param_labels, current_guess, param_bounds,
                )
                for slider, val in zip(sliders, popt):
                    slider.set_val(val)
            except Exception as e:
                print("Refit failed:", e)

        button_save.on_clicked(save_callback)
        button_refit.on_clicked(refit_callback)
        zoom_factory(ax)
        pan_factory(ax, button=2)

        # Initial fit
        try:
            popt = fit_potential(x_data, y_data, mie_potential, param_labels, init_guess, param_bounds)
            x_fit = np.arange(float(textbox_zmin.text), float(textbox_zmax.text) + 0.01, 0.01)
            y_fit = mie_potential(x_fit, *popt)
        except Exception as e:
            print("Initial fit failed:", e)
            return

        # Note: `spec` here refers to the last spectrum in spectra_list from
        # the outer loop — this is a pre-existing behaviour, preserved exactly.
        # Before
        line_data, = ax.plot(x_data, y_data, linewidth=1.5, label=spec['filename'], color=spec['color'])
        line_fit, = ax.plot(x_fit, y_fit, '--', color='black', linewidth=2.5, label='Mie fit')

        dx = (max(x_data) - min(x_data)) / 20
        dy = (max(y_data) - min(y_data)) / 20
        ax.axis([min(x_data) - dx, max(x_data) + dx, min(y_data) - dy, max(y_data) + dy])
        ax.set_xlabel('Z (Å)')
        ax.set_ylabel('df (Hz)')
        ax.legend()
        ax.grid(True)
        mie_label = r"$f(z) = \left(\frac{a}{z_0 - z}\right)^{b} - \left(\frac{u}{z_0 - z}\right)^{v}$"
        ax.set_title('Interactive Mie Potential Fit\n' + mie_label + '\n(only one spectrum supported)')

        # Parameter sliders
        fig_miepotential.text(0.38, 0.34, "Current values of parameters", ha='center')
        slider_positions = [0.29, 0.24, 0.19, 0.14, 0.09]
        slider_ranges = np.array(param_bounds).T
        sliders = []
        param_boxes = []

        for i, label in enumerate(param_labels):
            ax_slider = plt.axes([0.15, slider_positions[i], 0.45, 0.04])
            slider = Slider(
                ax=ax_slider,
                label=label,
                valmin=slider_ranges[i][0],
                valmax=slider_ranges[i][1],
                valinit=popt[i],
            )
            sliders.append(slider)
            ax_box = plt.axes([0.61, slider_positions[i], 0.08, 0.04])
            box = TextBox(ax_box, "", initial=f"{popt[i]:.3f}")
            param_boxes.append(box)

        for i, box in enumerate(param_boxes):
            box.on_submit(textbox_submit_factory(i))

        def update(val) -> None:
            try:
                params = [s.val for s in sliders]
                a, b, u, v, z_0 = params

                for i, box in enumerate(param_boxes):
                    if box.text != f"{sliders[i].val:.4f}":
                        box.set_val(f"{sliders[i].val:.4f}")

                z_min_val = float(textbox_zmin.text)
                z_max_val = float(textbox_zmax.text)
                x_fit = np.arange(z_min_val, z_max_val + 0.01, 0.01)
                y_fit = mie_potential(x_fit, a, b, u, v, z_0)

                line_fit.set_xdata(x_fit)
                line_fit.set_ydata(y_fit)
                ax.relim()
                ax.autoscale_view()
                fig_miepotential.canvas.draw_idle()
            except Exception as e:
                print("Update failed:", e)

        for s in sliders:
            s.on_changed(update)

        plt.show(block=False)
        root.mainloop()

    # ------------------------------------------------------------------
    # Polynomial fit sub-window
    # ------------------------------------------------------------------

    def fit_polynomial(event) -> None:
        """Open an interactive polynomial fitting window for all spectra."""
        if not plots_info:
            print("No spectra to fit.")
            return

        fig_polyfit, ax = plt.subplots(figsize=(8, 8))
        legend_visible = [len(spectra_list) < 5]
        fig_polyfit.canvas.manager.window.wm_geometry("+600+120")
        plt.subplots_adjust(left=0.15, bottom=0.25)

        # Degree controls
        ax_text_degree = plt.axes([0.15, 0.02, 0.1, 0.06])
        textbox_degree = TextBox(ax_text_degree, 'Degree ', initial=str(17))

        ax_slider_degree = plt.axes([0.15, 0.10, 0.6, 0.06])
        slider_degree = Slider(
            ax=ax_slider_degree,
            label='Degree',
            valmin=0,
            valmax=50,
            valinit=17,
            valstep=1,
        )

        degree = int(textbox_degree.text)

        all_x = np.concatenate([plot['x'].values for plot in plots_info])
        all_y = np.concatenate([plot['y'].values for plot in plots_info])
        x_range = max(all_x) - min(all_x)
        x_fit = np.linspace(min(all_x) - x_range / 20, max(all_x) + x_range / 20, 10000)

        fit_lines = []
        for plot in plots_info:
            x_data = plot['x'].values
            y_data = plot['y'].values
            label = plot['label']
            color = plot['color']
            ax.plot(x_data, y_data, linewidth=1.5, label=label, color=color)
            poly_fun = polynomial_fit(x_data, y_data, degree)
            y_fit = poly_fun(x_fit)
            line_fit, = ax.plot(
                x_fit, y_fit,
                linewidth=2, linestyle="--",
                # Note: `spec` here refers to the last item from the outer
                # `for spec in spectra_list` loop — pre-existing behaviour,
                # preserved exactly.
                label=f"{spec['filename']}_Polyfit_{degree}",
                color='k',
            )
            fit_lines.append({'line': line_fit, 'x': x_data, 'y': y_data})

        dx = (max(all_x) - min(all_x)) / 10
        dy = (max(all_y) - min(all_y)) / 10
        ax.axis([min(all_x) - dx, max(all_x) + dx, min(all_y) - dy, max(all_y) + dy])

        if legend_visible[0]:
            leg = ax.legend(loc='best', fontsize='small')
            leg.set_draggable(True)
        ax.grid(True)
        ax.set_xlabel("Z (Å)")
        ax.set_ylabel("df (Hz)")
        ax.set_title("Interactive Polynomial Fit")

        ax_button_save = fig_polyfit.add_axes([0.7, 0.02, 0.2, 0.06])
        button_save = Button(ax_button_save, 'Export to CSV')

        check_ax = fig_polyfit.add_axes([0.73, 0.86, 0.15, 0.08])
        check_ax.set_frame_on(False)
        check = CheckButtons(check_ax, ["Show Legend"], [legend_visible[0]])

        def toggle_legend(event) -> None:
            legend_visible[0] = not legend_visible[0]
            if legend_visible[0]:
                leg = ax.legend(loc='upper left', fontsize='small')
                leg.set_draggable(True)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            fig_polyfit.canvas.draw_idle()

        check.on_clicked(toggle_legend)
        button_save.on_clicked(lambda event: export_to_csv(ax, 'Z', 'df'))
        zoom_factory(ax)
        pan_factory(ax, button=2)

        def update(val=None) -> None:
            try:
                degree = int(slider_degree.val)
                if textbox_degree.text != str(degree):
                    textbox_degree.set_val(str(degree))
                for fit in fit_lines:
                    poly_fun = polynomial_fit(fit['x'], fit['y'], degree)
                    fit['line'].set_ydata(poly_fun(x_fit))
                    fit['line'].set_label(f"{spec['filename']}_Polyfit_{degree}")
                if legend_visible[0]:
                    leg = ax.legend(loc='upper left', fontsize='small')
                    leg.set_draggable(True)
                ax.relim()
                ax.autoscale_view()
                fig_polyfit.canvas.draw_idle()
            except Exception as e:
                print("Polynomial fit update failed:", e)

        def on_textbox_submit(text: str) -> None:
            try:
                deg = int(text)
                deg = max(int(slider_degree.valmin), min(int(slider_degree.valmax), deg))
                slider_degree.set_val(deg)
            except ValueError:
                print("Invalid degree entered")

        textbox_degree.on_submit(on_textbox_submit)
        slider_degree.on_changed(update)

        plt.show(block=False)
        root.mainloop()

    # ------------------------------------------------------------------
    # Button dispatch
    # ------------------------------------------------------------------

    def make_button_callback(name: str):
        def callback(event) -> None:
            if name == "Fit Mie potential":
                fit_mie_potential(event)
            elif name == "Fit polynomial":
                fit_polynomial(event)
        return callback

    # --- Figure layout ---
    index_min, index_max = index_range

    fig_force_spec = plt.figure(figsize=(10, 8))
    fig_force_spec.canvas.manager.window.wm_geometry("+400+80")
    gs = fig_force_spec.add_gridspec(1, 2, width_ratios=[7, 1], wspace=0.01)

    ax_force_spec = fig_force_spec.add_subplot(gs[0, 0])
    ax_force_spec.set_title("Frequency Spectra")
    ax_force_spec.set_xlabel("Z (Å)")
    ax_force_spec.set_ylabel("df (Hz)")
    ax_force_spec.grid(True)

    # --- Plot spectra and build plots_info ---
    plots_info = []
    for spec in spectra_list:
        data = spec['data']
        if 'Z' in data.columns and 'df' in data.columns:
            x_data = data['Z'].iloc[index_min:index_max]
            y_data = data['df'].iloc[index_min:index_max]
            x_rot, y_rot = spec.get('rotated', (np.nan, np.nan))
            label = f"{spec['filename']} [{x_rot:.1f}, {y_rot:.1f}] Å"
            line, = ax_force_spec.plot(x_data, y_data, label=label, linewidth=1.5, color=spec['color'])
            x_rot, y_rot = spec.get('rotated', (np.nan, np.nan))
            color = spec['color']
            plots_info.append({
                'x': x_data,
                'y': y_data,
                'label': label,
                'line': line,
                'x_rot': x_rot,
                'y_rot': y_rot,
                'color': color
            })

    leg = ax_force_spec.legend(loc='best', fontsize='small')
    leg.set_draggable(True)

    # Right: control panel
    ax_controls = fig_force_spec.add_subplot(gs[0, 1])
    ax_controls.axis('off')

    button_labels = ["Fit Mie potential", "Fit polynomial"]
    buttons = {}
    for i, button_label in enumerate(button_labels):
        button_ax = fig_force_spec.add_axes([0.83, 0.75 - i * 0.1, 0.14, 0.05])
        btn = Button(button_ax, button_label)
        btn.on_clicked(make_button_callback(button_label))
        buttons[button_label] = btn

    # --- Legend toggle ---
    legend_visible = [len(spectra_list) < 5]

    if legend_visible[0]:
        leg = ax_force_spec.legend(loc='best', fontsize='small')
        leg.set_draggable(True)
    else:
        if ax_force_spec.get_legend():
            ax_force_spec.legend_.remove()

    check_ax = fig_force_spec.add_axes([0.83, 0.825, 0.12, 0.08])
    check_ax.set_frame_on(False)
    legend_check = CheckButtons(check_ax, ["Show Legend"], legend_visible)

    def toggle_legend(label) -> None:
        if legend_check.get_status()[0]:
            leg = ax_force_spec.legend(loc='best', fontsize='small')
            leg.set_draggable(True)
        else:
            leg = ax_force_spec.get_legend()
            if leg:
                leg.remove()
        fig_force_spec.canvas.draw_idle()

    legend_check.on_clicked(toggle_legend)
    zoom_factory(ax_force_spec)
    pan_factory(ax_force_spec, button=2)

    plt.show(block=False)
    fig_force_spec.canvas.mpl_connect('close_event', on_close)
    root.mainloop()
