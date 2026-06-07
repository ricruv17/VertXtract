# kpfs_window.py
import re
import ast
import matplotlib
matplotlib.use("TkAgg")
from utilities import extract_data_RvsZplot, extract_data_XvsYplot, zoom_factory, pan_factory, create_colormap_menu, export_grid_csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import tkinter as tk

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'text.usetex':False, "svg.fonttype": 'none'})

def kpfs_window(spectra_list, index_range):
    root = tk.Tk()
    root.withdraw()
    global current_cmap
    current_cmap = tk.StringVar(value="coolwarm")

    def on_close(event):
        plt.close(fig_kpfs)

    def fit_parabolas(event):
        for plot in plots_info:
            x = plot['x'].values
            y = plot['y'].values
            if len(x) < 3:
                continue  # Need at least 3 points to fit a parabola
            try:
                coeffs = np.polyfit(x, y, 2)
                a, b, c = coeffs
                x_fit = np.linspace(min(x), max(x), 200)
                y_fit = a * x_fit**2 + b * x_fit + c
                ax_kpfs.plot(x_fit, y_fit, '--', color='k')

                # Optional: mark the vertex (LCPD)
                x_vertex = -b / (2 * a)
                y_vertex = a * x_vertex**2 + b * x_vertex + c
                ax_kpfs.plot(x_vertex, y_vertex, 'o', color='k')

                # Update the original line's label
                plot['line'].set_label(f"{plot['label']}, df*={y_vertex:.3f}, LCPD={x_vertex:.2f} mV")

            except Exception as e:
                print(f"Fit failed for {plot['label']}: {e}")

        fig_kpfs.canvas.draw_idle()

    def get_termination_number(filename):
        match = re.search(r"[LM](\d{4})", filename)
        return int(match.group(1)) if match else None

    def plot_lcpd_vs_r():
        term_data = []
        for plot in plots_info:
            term_num = get_termination_number(plot['label'])
            if term_num is None:
                continue

            match = re.search(r"LCPD=(-?\d+\.?\d*)", plot['line'].get_label())
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
                plot.get('y_rot', np.nan)
            ))

        if not term_data:
            print("No LCPD data available. Fit first.")
            return

        term_data = np.array(term_data, dtype=float)  # (N, 5)
        unique_terms = np.unique(term_data[:, 0])
        unique_terms.sort()
        first_term = unique_terms[0]
        last_term = unique_terms[-1]

        avg_first_x = np.nanmean(term_data[term_data[:, 0] == first_term, 3])
        avg_first_y = np.nanmean(term_data[term_data[:, 0] == first_term, 4])
        avg_last_x = np.nanmean(term_data[term_data[:, 0] == last_term, 3])
        avg_last_y = np.nanmean(term_data[term_data[:, 0] == last_term, 4])

        displacement_distance = np.sqrt(
            (avg_last_x - avg_first_x) ** 2 +
            (avg_last_y - avg_first_y) ** 2
        )

        displacement = np.linspace(0, displacement_distance, len(unique_terms))

        fig, ax2 = plt.subplots(figsize=(8, 8))
        fig.canvas.manager.window.wm_geometry("+600+120")

        term_data = term_data[np.lexsort((term_data[:, 0], term_data[:, 1]))]
        nterms=int(term_num)
        displacement_grid = np.sort(np.unique(displacement))
        lcpd_grid = np.array(term_data.T[2][:nterms])

        # Plot
        plt.plot(displacement_grid, lcpd_grid, '-o', color='black', zorder=-1)
        
        check_ax2 = fig.add_axes([0.78, 0.87, 0.12, 0.08])
        check_ax2.set_frame_on(False)
        values_check = CheckButtons(check_ax2, ["Show Values"], legend_visible)

        text_artists = []
        def toggle_legend(label):
            if values_check.get_status()[0]:
                if not text_artists:
                   for a, lcpd in enumerate(lcpd_grid):
                    if a % 2 == 1:
                        yval = lcpd_grid[a]
                    else:
                        yval = lcpd_grid[a]
                    t = ax2.text(displacement[a]-0.4, yval, f'{lcpd}')
                    text_artists.append(t)
            else:
                for t in text_artists:
                    t.remove()
                text_artists.clear()
            fig.canvas.draw_idle()

        values_check.on_clicked(toggle_legend)
        
        if np.sign(height) == -1:
            sign = '-'
        else:
            sign = '+'
        ax2.set_title(f'LCPD vs R map at setpoint {sign}{np.abs(height)} Å')
        ax2.set_ylabel(r"LCPD (mV)")
        ax2.set_xlabel(r"Displacement (Å)")
        zoom_factory(ax2)
        pan_factory(ax2, button=2)
        plt.show()

    def plot_lcpd_vs_z():
        xy_tol = 1  # Å

        extracted_data = []
        for plot in plots_info:
            match = re.search(r"LCPD=(-?\d+\.?\d*)", plot['line'].get_label())
            if not match:
                continue

            lcpd_value = float(match.group(1))
            location_match = re.search(r'Location=(\[[^\]]+\])', plot['label'])
            if not location_match:
                continue

            location = ast.literal_eval(location_match.group(1))
            x = float(location[0])
            y = float(location[1])
            z = float(location[2])
            extracted_data.append({'x': x,
                                   'y': y,
                                   'z': z,
                                   'lcpd': lcpd_value,
                                   'color': plot['color']
                                   })

        if not extracted_data:
            print("No LCPD data available. Fit first.")
            return

        groups = []
        for item in extracted_data:
            added = False
            for group in groups:
                gx, gy = group['center']
                if (
                    abs(item['x'] - gx) < xy_tol and
                    abs(item['y'] - gy) < xy_tol and
                    item['color'] == group['color']
                ):
                    group['data'].append(item)
                    added = True
                    break

            if not added:
                groups.append({
                    'center': (item['x'], item['y']),
                    'data': [item],
                    'color': item['color']
                })

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.window.wm_geometry("+600+120")

        text_artists = []
        for i, group in enumerate(groups):
            group_data = group['data']
            group_data = sorted(group_data, key=lambda d: d['z'])

            heights = [d['z'] for d in group_data]
            lcpd_values = [d['lcpd'] for d in group_data]

            x_mean = np.mean([d['x'] for d in group_data])
            y_mean = np.mean([d['y'] for d in group_data])
            label = f"Position {i+1} [{x_mean:.1f}, {y_mean:.1f}] Å"
            ax.plot(heights, lcpd_values, '-o', label=label, color=group['color'])

        legend_visible = [False]
        check_ax2 = fig.add_axes([0.78, 0.87, 0.12, 0.08])
        check_ax2.set_frame_on(False)
        values_check = CheckButtons(check_ax2, ["Show Values"], legend_visible)

        def toggle_legend(label):
            if values_check.get_status()[0]:
                if not text_artists:
                    for line in ax.lines:
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        for x, y in zip(xdata, ydata):
                            t = ax.text(x, y, f'{y:.1f}', fontsize=9)
                            text_artists.append(t)
            else:
                for t in text_artists:
                    t.remove()
                text_artists.clear()
            fig.canvas.draw_idle()
        values_check.on_clicked(toggle_legend)

        ax.set_xlabel("Height (Å)")
        ax.set_ylabel("LCPD (mV)")
        ax.set_title("LCPD vs Z")
        ax.legend().set_draggable(True)
        ax.grid(True)
        zoom_factory(ax)
        pan_factory(ax, button=2)
        plt.show()
        
    def plot_lcpd_z_vs_r():
        (x_edges,
         y_edges,
         grid,
         displacement,
         heights) = extract_data_RvsZplot(plots_info, get_termination_number)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.window.wm_geometry("+600+120")
        
        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            cmap=current_cmap.get(),
            shading='auto'
        )
        create_colormap_menu(fig, pcm, current_cmap)

        check_ax = fig.add_axes([0.65, 0.875, 0.12, 0.08])
        check_ax.set_frame_on(False)
        values_check = CheckButtons(check_ax, ["Show Values"], legend_visible)

        text_artists = []
        def toggle_legend(label):
            if values_check.get_status()[0]:
                if not text_artists:
                    for a in range(len(x_edges) - 1):
                        x = x_edges[a] + (x_edges[a+1] - x_edges[a]) / 2
                        for b in range(len(y_edges) - 1):
                            y = y_edges[b] + (y_edges[b+1] - y_edges[b]) / 2
                            t = ax.text(x, y, f'{grid[b, a]:.2f}', fontsize=8, ha='center', va='center', color='black', zorder=10)
                            text_artists.append(t)
            else:
                for t in text_artists:
                    t.remove()
                text_artists.clear()
            fig.canvas.draw_idle()

        values_check.on_clicked(toggle_legend)

        fig.colorbar(pcm, ax=ax, label='LCPD (mV)')
        ax.set_xlabel("Displacement (Å)")
        ax.set_ylabel("Relative height (Å)")
        ax.set_yticks(heights)
        ax.invert_yaxis()
        ax.set_title("LCPD: Z vs R map")
        zoom_factory(ax)
        pan_factory(ax, button=2)
        plt.show()

    def plot_lcpd_x_vs_y():
        (x_edges,
         y_edges,
         grid,
         x_centers,
         y_centers) = extract_data_XvsYplot(plots_info)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.window.wm_geometry("+600+120")
        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            cmap=current_cmap.get(),
            shading='auto'
        )
        create_colormap_menu(fig, pcm, current_cmap)

        check_ax = fig.add_axes([0.65, 0.875, 0.12, 0.08])
        check_ax.set_frame_on(False)
        values_check = CheckButtons(check_ax, ["Show Values"], legend_visible)

        text_artists = []
        def toggle_legend(label):
            if values_check.get_status()[0]:
                if not text_artists:
                    for a in range(len(x_edges) - 1):
                        x = x_edges[a] + (x_edges[a+1] - x_edges[a]) / 2
                        for b in range(len(y_edges) - 1):
                            y = y_edges[b] + (y_edges[b+1] - y_edges[b]) / 2
                            t = ax.text(x, y, f'{grid[b, a]:.2f}', fontsize=8, ha='center', va='center', color='black', zorder=10)
                            text_artists.append(t)
            else:
                for t in text_artists:
                    t.remove()
                text_artists.clear()
            fig.canvas.draw_idle()

        values_check.on_clicked(toggle_legend)

        fig.colorbar(pcm, ax=ax, label='LCPD (mV)')
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_title("LCPD: X vs Y map")
        zoom_factory(ax)
        pan_factory(ax, button=2)
        plt.show()

    def make_button_callback(name):
        def callback(event):
            if name == "Fit LCPD":
                fit_parabolas(event)
            elif name == "LCPD X vs Y":
                plot_lcpd_x_vs_y()
            elif name == "LCPD X vs Z":
                plot_lcpd_z_vs_r()
            elif name == "LCPD vs X":
                plot_lcpd_vs_r()
            elif name == "LCPD vs Z":
                plot_lcpd_vs_z()
        return callback

    index_min, index_max = index_range

    fig_kpfs = plt.figure(figsize=(10, 8))
    fig_kpfs.canvas.manager.window.wm_geometry("+400+80")
    gs = fig_kpfs.add_gridspec(1, 2, width_ratios=[7, 1], wspace=0.01)

    # Left: KPFS plot
    ax_kpfs = fig_kpfs.add_subplot(gs[0, 0])
    ax_kpfs.set_title("KPFS Spectra")
    ax_kpfs.set_xlabel("Voltage (mV)")
    ax_kpfs.set_ylabel("df (Hz)")
    ax_kpfs.grid(True)

    # Store original plots so we can refit later
    plots_info = []

    for spec in spectra_list:
        data = spec['data']
        if 'Voltage' in data.columns and 'df' in data.columns:
            x_data = data['Voltage'].iloc[index_min:index_max]
            y_data = data['df'].iloc[index_min:index_max]

            if spec.get('height') is not None:
                label = f"{spec['filename']} (Location={spec['offset']} Å)"
            else:
                label = spec['filename']

            line, = ax_kpfs.plot(x_data, y_data, label=label, linewidth=1.5, color=spec['color'])

            # unpack rotated coords
            x_rot, y_rot = spec.get('rotated', (np.nan, np.nan))

            plots_info.append({
                'x': x_data,
                'y': y_data,
                'label': label,
                'line': line,
                'x_rot': x_rot,
                'y_rot': y_rot,
                'color': spec['color']
            })

    leg = ax_kpfs.legend(loc='best', fontsize='small')
    leg.set_draggable(True)


    # Buttons
    ax_controls = fig_kpfs.add_subplot(gs[0, 1])
    ax_controls.axis('off')
    is_specgrid = any(spec['filename'].endswith(".Vert") for spec in spectra_list)
    heights = []

    for spec in spectra_list:
        h = spec.get('height')
        if h is not None:
            heights.append(round(float(h), 3))
    unique_heights = np.unique(heights)
    button_labels = ["Fit LCPD"]

    multiple_heights = len(unique_heights) > 1

    if is_specgrid:
        button_labels.append("LCPD X vs Y")
    else:
        if multiple_heights:
            button_labels.append("LCPD vs Z")
            button_labels.append("LCPD X vs Z")
        else:
            button_labels.append("LCPD X vs Z")
            button_labels.append("LCPD vs X")

    buttons = {}
    for i, button_label in enumerate(button_labels):
        button_ax = fig_kpfs.add_axes([0.85, 0.75 - i * 0.1, 0.12, 0.05])
        button = Button(button_ax, button_label)
        button.on_clicked(make_button_callback(button_label))
        buttons[button_label] = button  # Store button by label

    # Legend toggle check button
    legend_visible = [0] #[len(spectra_list) < 5]  # alternative to make True if less than 5 spectra

    if legend_visible[0]:
        leg = ax_kpfs.legend(loc='best', fontsize='small')
        leg.set_draggable(True)
    else:
        ax_kpfs.legend_.remove() if ax_kpfs.get_legend() else None  # ensure hidden

    check_ax = fig_kpfs.add_axes([0.83, 0.825, 0.12, 0.08])
    check_ax.set_frame_on(False)
    legend_check = CheckButtons(check_ax, ["Show Legend"], legend_visible)

    def toggle_legend(label):
        if legend_check.get_status()[0]:
            leg = ax_kpfs.legend(loc='best', fontsize='small')
            leg.set_draggable(True)
            for text in leg.get_texts():
                print(text.get_text())
        else:
            leg = ax_kpfs.get_legend()
            if leg:
                leg.remove()
        fig_kpfs.canvas.draw_idle()

    zoom_factory(ax_kpfs)
    pan_factory(ax_kpfs, button=2)
    legend_check.on_clicked(toggle_legend)

    plt.show(block=False)
    fig_kpfs.canvas.mpl_connect('close_event', on_close)
    root.mainloop()

