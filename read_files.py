# read_files.py
# -*- coding: utf-8 -*-
"""
STM/AFM File Reader
====================
Handles all file I/O and data extraction for Createc STM/AFM experiment
folders.  Reads .DAT image files and .VERT / .Vert spectra files, applies
instrument-specific calibration, and exposes the results as structured
Python/NumPy/Pandas objects ready for visualisation.

Intended to be imported by main_viewer.py, which handles all GUI logic.

Author: Ricardo Ruvalcaba, August 17, 2025
"""

import os
import re
from typing import Optional

import createc
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Calibration constants (empirically extracted from instrument characterisation)
# ---------------------------------------------------------------------------

# Piezo sensitivity: ångström per volt of Z-piezo drive signal
PIEZO_CONSTANT: float = 5 / 667

# Vertman time base: seconds per vertmandelay unit
TIME_CONSTANT: float = 31 / 250

# DAC-to-ångström conversion for scan offset registers (×10 gives nm→Å)
DAC_2_ANGSTROM: float = -0.0009155636965256441

# Topography channel: ADC counts → ångström (ratio of calibrated reference values)
TOPO_SCALE: float = 280.019 / 163_117.688

# Current channel: ADC counts → amperes
CURRENT_SCALE: float = 6.329e-12 / 331.8

# Damping channel: ADC counts → volts
DAMPING_SCALE: float = 1.748e-01 / 8_335.7

# Amplitude channel: ADC counts → volts
AMPLITUDE_SCALE: float = 9.288e-04 / 48.6968

# Percentile range used for colour-scale normalisation
CLIM_PERCENTILE_LOW: float = 0.0
CLIM_PERCENTILE_HIGH: float = 100.0

# Slider range for selection radius (log₁₀ scale, ångström)
SLIDER_LOG_MIN: float = np.log10(1)
SLIDER_LOG_MAX: float = np.log10(200)
SLIDER_STEP: float = 0.01
SELECTION_RADIUS_DEFAULT: float = 2.0

# Padding factor applied to the auto-computed square plot extent
PLOT_EXTENT_PADDING: float = 1.05

# ---------------------------------------------------------------------------
# Channel name lists (index-matched to Createc bit-field codes)
# Empty strings "" are intentional placeholders for unused bit positions.
# ---------------------------------------------------------------------------

OUT_CHANNEL_LIST: list[str] = ["Voltage", "Z", "Vaux"]

IMAGE_CHANNELS_LIST: list[str] = [
    "Topography", "Current", "ADC1", "ADC2", "ADC3",
    "Lock-in X", "Lock-in X(2f)", "df",
    "Damping", "Amplitude", "BIAS FB", "Aux1", "Aux2",
    "", "", "ADC4", "ADC5", "ADC6", "ADC7", "",
    "Pot.Volt", "Pot.Current", "", "", "", "", "", "", "", "",
    "Aux3", "Aux4", "Lock-in Y", "Lock-in Y(2f)",
    "", "", "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "",
    "XPLLFrequency", "XPLLAmplitude", "XPLLExcitation",
    "", "", "", "", "", "", "",
    "Digout", "Marker",
]

CHANNEL_LABELS: dict[str, str] = {
    "Topography": "Topography (Å)",
    "Current": "Current (A)",
    "df": "df (Hz)",
    "Damping": "Damping (V)",
    "Amplitude": "Amplitude (V)",
}

VERTMAN_CHANNELS_LIST: list[str] = [
    "Current(filtered)", "Lock-in X", "Lock-in X(2f)",
    "ACD0", "ACD1", "ACD2", "ACD3",
    "df", "Damping", "Amplitude", "Lock-in Y", "Lock-in Y(2f)",
    "Z-Signal", "BIAS FB", "",
    "ACD4", "ACD5", "ACD6", "ACD7", " ",
    "Z-Topography", "Pot.Volt", "Pot.Current",
    "", "", "", "", "", "", "",
    "Aux3", "Aux4", "SignalBias",
    "", "", "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "",
    "XPLLFrequency", "XPLLAmplitude", "XPLLExcitation",
    "", "", "", "", "", "", "", "", "Marker",
]

tab10_colors = plt.get_cmap('tab10').colors
color_cycle = itertools.cycle(tab10_colors)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def rotate_point_around_offset(
    x: float,
    y: float,
    angle_deg: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float]:
    """Rotate point (x, y) around (offset_x, offset_y) by angle_deg degrees.

    Uses the standard 2-D rotation matrix expressed in homogeneous coordinates
    so that the pivot point is handled by a translate-rotate-untranslate
    composition, which keeps the arithmetic identical to the original.

    Parameters
    ----------
    x, y:
        Coordinates of the point to rotate (ångström).
    angle_deg:
        Clockwise rotation angle in degrees (sign convention follows Createc).
    offset_x, offset_y:
        Pivot point coordinates (ångström).

    Returns
    -------
    tuple[float, float]
        Rotated (x, y) coordinates in ångström.
    """
    theta = np.radians(angle_deg)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    translation_matrix = np.array([
        [1, 0, -offset_x],
        [0, 1, -offset_y],
        [0, 0,  1        ],
    ])

    point_h = np.array([x, y, 1.0])
    final_point = (
        np.linalg.inv(translation_matrix)
        @ rotation_matrix
        @ translation_matrix
        @ point_h
    )
    return float(final_point[0]), float(final_point[1])


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------

class STMAFMReader:
    """Loads and calibrates all data from a single Createc STM/AFM experiment folder.

    Reads one .DAT image file and all .VERT / .Vert spectra files, applies
    instrument-specific calibration, and stores the results as instance
    attributes for consumption by a visualisation layer.

    Parameters
    ----------
    folder_path:
        Absolute path to the folder containing exactly one .DAT file and
        any number of .VERT / .Vert spectra files.

    Attributes
    ----------
    folder_path : str
    dat_file : str
    image_location : str
    scan : createc.DAT_IMG
    pixels_x, pixels_y : int
    ImageSizeX, ImageSizeY : float
    ScanRotation : float
    offset_x, offset_y : float          Raw DAC register values.
    offset_A_x, offset_A_y : float      Converted to ångström.
    image_size_x_A, image_size_y_A : float
    pixel_size_x_A, pixel_size_y_A : float
    X0, Y0 : np.ndarray                 Physical coordinate meshgrid (ångström).
    available_channels : list[tuple[str, int]]
    nonempty_channels : list[tuple[str, int]]
    current_channel : str
    first_valid_index : int
    image : np.ndarray                  Current channel image (set by _load_image_channel).
    trace : np.ndarray                  Calibrated 1-D trace (set by _load_image_channel).
    spectra : list[dict]
    """

    def __init__(self, folder_path: str) -> None:
        self.folder_path: str = folder_path

        self.dat_file: str = self._find_dat_file()
        self.image_location: str = os.path.join(self.folder_path, self.dat_file)
        self.scan: createc.DAT_IMG = createc.DAT_IMG(self.image_location)

        self._load_metadata()
        self._load_spectra()
        self._detect_channels()

    # ------------------------------------------------------------------
    # DAT file helpers
    # ------------------------------------------------------------------

    def _find_dat_file(self) -> str:
        """Return the single .DAT filename found in *folder_path*.

        Raises
        ------
        RuntimeError
            If there is not exactly one .DAT file in the folder.
        """
        dat_files = [
            f for f in os.listdir(self.folder_path)
            if f.lower().endswith('.dat')
        ]
        if len(dat_files) != 1:
            raise RuntimeError(
                f"Expected exactly one .DAT file in '{self.folder_path}', "
                f"found {len(dat_files)}."
            )
        return dat_files[0]

    def _load_metadata(self) -> None:
        """Parse scan metadata from the .DAT file header.

        Reads pixel dimensions, physical size, rotation angle, and piezo
        offset registers.  Derived quantities (ångström offsets, pixel sizes,
        and the coordinate meshgrid) are computed and stored as instance
        attributes.
        """
        print("[INFO] Loading image metadata...")

        self.pixels_x: int = self.scan.xPixel
        self.pixels_y: int = self.scan.yPixel
        self.ImageSizeX: float
        self.ImageSizeY: float
        self.ImageSizeX, self.ImageSizeY = self.scan.size

        self.ScanRotation: float = 0.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

        with open(self.image_location, 'r', encoding='latin1') as fh:
            for line in fh:
                if 'Rotation / Rotation' in line:
                    self.ScanRotation = float(line.split('=')[-1])
                elif 'Scanrotoffx / OffsetX' in line:
                    self.offset_x = float(line.split('=')[-1])
                elif 'Scanrotoffy / OffsetY' in line:
                    self.offset_y = float(line.split('=')[-1])

        print(
            f"[OK] Metadata loaded: {self.pixels_x}×{self.pixels_y} pixels, "
            f"{self.ImageSizeX} Å × {self.ImageSizeY} Å, "
            f"rotation={self.ScanRotation:.2f}°"
        )

        # Convert DAC register values to ångström (×10 converts nm → Å)
        self.offset_A_x: float = self.offset_x * DAC_2_ANGSTROM * 10
        self.offset_A_y: float = self.offset_y * DAC_2_ANGSTROM * 10

        self.image_size_x_A: float = self.ImageSizeX
        self.image_size_y_A: float = self.ImageSizeY
        self.pixel_size_x_A: float = self.image_size_x_A / self.pixels_x
        self.pixel_size_y_A: float = self.image_size_y_A / self.pixels_y

        # Physical coordinate grid used for imshow *extent* and scatter plots
        x_vals = np.linspace(
            -self.image_size_x_A / 2,
             self.image_size_x_A / 2,
             self.pixels_x,
        )
        y_vals = np.linspace(0, self.image_size_y_A, self.pixels_y)
        self.X0, self.Y0 = np.meshgrid(x_vals, y_vals)
        self.X0 += self.offset_A_x
        self.Y0 += self.offset_A_y

    def _detect_channels(self) -> None:
        """Identify which image channels are present and non-trivial.

        Decodes the Createc channel bitmask, then inspects each channel's
        data for emptiness and zero-variance, populating:

        * ``self.available_channels`` — all non-empty channels (name, imgs_index).
        * ``self.nonempty_channels`` — channels with non-zero variance.
        * ``self.current_channel`` — default channel name (Topography > df > first).
        * ``self.first_valid_index`` — scan.imgs index for the default channel.
        """
        print("[INFO] Detecting available channels...")

        channel_bits = list(reversed(bin(int(self.scan.channels_code))[2:]))

        self.available_channels: list[tuple[str, int]] = []
        self.nonempty_channels: list[tuple[str, int]] = []

        for bit_pos, bit_val in enumerate(channel_bits):
            if bit_val != '1':
                continue
            if bit_pos >= len(IMAGE_CHANNELS_LIST):
                continue
            name = IMAGE_CHANNELS_LIST[bit_pos]
            if name == "":
                continue

            # Sequential imgs index: count preceding set bits with a valid name
            imgs_index = sum(
                1
                for j, bv in enumerate(channel_bits)
                if bv == '1' and j < bit_pos and IMAGE_CHANNELS_LIST[j] != ""
            )

            trace = np.array(self.scan.imgs[imgs_index])
            if trace.size == 0:
                continue

            self.available_channels.append((name, imgs_index))
            if np.std(trace) > 0:
                self.nonempty_channels.append((name, imgs_index))

        print("[OK] Channels found:", [c[0] for c in self.available_channels])

        # Choose the best default channel
        channel_names = [c[0] for c in self.nonempty_channels]
        if 'Topography' in channel_names:
            self.current_channel = 'Topography'
        elif 'df' in channel_names:
            self.current_channel = 'df'
        else:
            self.current_channel = self.nonempty_channels[0][0]

        self.first_valid_index: int = next(
            idx for name, idx in self.nonempty_channels
            if name == self.current_channel
        )

    # ------------------------------------------------------------------
    # Image loading and processing
    # ------------------------------------------------------------------

    def _is_valid_image(self, image: np.ndarray) -> bool:
        """Return True if *image* contains finite, non-zero data."""
        if image.size == 0:
            return False
        if not np.any(np.isfinite(image)):
            return False
        if np.all(image == 0):
            return False
        return True

    def _trim_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Strip trailing all-zero / non-finite rows from a 2-D image array.

        Parameters
        ----------
        image:
            2-D NumPy array of raw channel data.

        Returns
        -------
        np.ndarray or None
            Trimmed image, or None if no valid rows exist.
        """
        valid_row_mask = (
            np.any(np.isfinite(image), axis=1)
            & np.any(image != 0, axis=1)
        )
        if not np.any(valid_row_mask):
            return None
        last_valid_row = int(np.where(valid_row_mask)[0].max())
        return image[: last_valid_row + 1, :]

    def _apply_channel_calibration(self, raw: np.ndarray) -> np.ndarray:
        """Apply the instrument-specific ADC → physical-unit conversion.

        All scale factors are empirically extracted from calibration
        measurements.  The Topography channel is additionally shifted so that
        its minimum value is zero (relative height).

        Parameters
        ----------
        raw:
            1-D array of raw ADC counts for the current channel.

        Returns
        -------
        np.ndarray
            Calibrated 1-D array in physical units.
        """
        data = raw.copy()
        if self.current_channel == 'Topography':
            data *= TOPO_SCALE
            data -= data.min()
        elif self.current_channel == 'Current':
            data *= CURRENT_SCALE
        elif self.current_channel == 'Damping':
            data *= DAMPING_SCALE
        elif self.current_channel == 'Amplitude':
            data *= AMPLITUDE_SCALE
        return data

    def _load_image_channel(self, channel_index: int) -> None:
        """Load, calibrate, and reshape a channel from ``self.scan.imgs``.

        The result is stored in ``self.image``.  Incomplete scans (where the
        raw data length is not a perfect multiple of either pixel dimension)
        are handled by trimming trailing zero/non-finite rows.

        Parameters
        ----------
        channel_index:
            Index into ``self.scan.imgs`` for the desired channel.
        """
        raw = np.array(self.scan.imgs[channel_index])
        self.trace = self._apply_channel_calibration(raw)

        total_pts = self.trace.size
        if total_pts % self.pixels_x == 0:
            n_rows = total_pts // self.pixels_x
            n_cols = self.pixels_x
        else:
            n_cols = total_pts // self.pixels_y
            n_rows = self.pixels_y

        image = self.trace[: n_rows * n_cols].reshape((n_rows, n_cols))

        if not self._is_valid_image(image):
            print(f"[WARNING] Channel '{self.current_channel}' is empty or invalid.")
            self.image = image
            return

        trimmed = self._trim_image(image)
        if trimmed is not None and trimmed.shape[0] < image.shape[0]:
            print(f"[INFO] Trimmed image: new shape = {trimmed.shape}")
            self.image = trimmed
            self.current_pixels_y: int = trimmed.shape[0]
        else:
            self.image = image

    # ------------------------------------------------------------------
    # Spectra loading
    # ------------------------------------------------------------------

    def _extract_vertman_data(
        self,
        filename: str,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Parse a Createc .VERT or .Vert spectra file.

        Reads the file header for acquisition parameters, then parses the
        DATA block into a Pandas DataFrame with named columns.  The Z column
        is converted from volts to ångström using the piezo calibration
        constant.  A 'Time' column is prepended derived from the vertmandelay
        parameter.

        Parameters
        ----------
        filename:
            Full path to the .VERT or .Vert file.

        Returns
        -------
        offset : np.ndarray, shape (2,)
            (x, y) position of the spectrum in ångström.
        data : pd.DataFrame
            Calibrated spectrum data with named columns.

        Raises
        ------
        ValueError
            If no DATA section is found in the file.
        """
        with open(filename, 'r', encoding='latin1') as fh:
            lines = fh.readlines()

        vertmandelay: float = 0.0
        specgrid_nx: int = 1
        specgrid_ny: int = 1
        data_index: Optional[int] = None

        for i, line in enumerate(lines):
            if "Vertmandelay" in line:
                vertmandelay = float(line.split('=')[-1])
            elif "SpecGridNX" in line:
                specgrid_nx = int(line.split('=')[-1])
            elif "SpecGridNY" in line:
                specgrid_ny = int(line.split('=')[-1])
            elif line.strip() == "DATA":
                data_index = i
                break

        if data_index is None:
            raise ValueError(f"No 'DATA' section found in '{filename}'.")

        header_line = lines[data_index + 1].split()
        n_points = int(header_line[0])

        # Determine spatial offset depending on file extension
        if filename.endswith('.VERT'):
            coords = header_line[-3:]
            if coords[-1] == '0':       # for version 20260623
                coords = coords[:2]
            else:                       # for version 20240417
                coords = coords[-1:]
            offset = np.array(list(map(float, coords))) * 10
        elif filename.endswith('.Vert'):
            basename = filename.split('\\')[-1].split('.')[-2]
            xcoord = int(basename[3:])
            ycoord = int(basename[:3])
            offset_A_x = (
                self.offset_A_x
                - self.image_size_x_A * ((specgrid_nx + 1) / (specgrid_nx * 2))
                + self.image_size_x_A / specgrid_nx * xcoord
            )
            offset_A_y = (
                self.offset_A_y
                - self.image_size_y_A / (specgrid_ny * 2)
                + self.image_size_y_A / specgrid_ny * ycoord
            )
            offset = np.array([offset_A_x, offset_A_y])
        else:
            offset = np.zeros(2)

        # Decode channel bitmasks
        vertman_channel_bits = list(reversed(bin(int(header_line[3]))[2:]))
        out_channel_bits = list(reversed(bin(int(header_line[4]))[2:]))

        # Parse data block
        data_str = ''.join(lines[data_index + 2: data_index + 2 + n_points])
        data = pd.read_csv(pd.io.common.StringIO(data_str), sep='\t', header=None)

        # Build column names from bitmasks
        columns_in_file = ['index']
        for i, bit_val in enumerate(out_channel_bits):
            if bit_val == '1':
                columns_in_file.append(OUT_CHANNEL_LIST[i])
        for i, bit_val in enumerate(vertman_channel_bits):
            if bit_val == '1':
                columns_in_file.append(VERTMAN_CHANNELS_LIST[i])
        columns_in_file.append('NaNs')

        data.columns = columns_in_file

        # Convert Z from piezo-drive volts to ångström
        if 'Z' in data.columns:
            data['Z'] *= PIEZO_CONSTANT

        data.drop(columns=['NaNs'], inplace=True)

        # Prepend absolute time axis
        max_index = data['index'].max()
        data.insert(
            0,
            'Time',
            data['index'] * vertmandelay / max_index * TIME_CONSTANT,
            allow_duplicates=True,
        )

        return offset, data

    def _load_spectra(self) -> None:
        """Discover and load all .VERT / .Vert spectra in *folder_path*.

        Each spectrum is stored as a dict in ``self.spectra`` with keys:

        * ``'filename'`` — basename of the file.
        * ``'data'`` — calibrated Pandas DataFrame.
        * ``'rotated'`` — (x, y) ndarray in the rotated image frame (ångström).
        * ``'offset'`` — [x, y, z] list rounded to 3 d.p. for display.
        * ``'height'`` — Z-approach height in ångström (or None if unavailable).
        """
        print("[INFO] Loading spectra data (.VERT files)...")
        self.spectra: list[dict] = []

        vert_files = [
            f for f in os.listdir(self.folder_path)
            if f.endswith(('.VERT', '.Vert'))
        ]
        print(f"[INFO] Found {len(vert_files)} spectra files.")

        for filename in tqdm(vert_files, desc="Processing spectra", unit="file"):
            vert_path = os.path.join(self.folder_path, filename)

            # Extract approach height from 'Zpoint2.z' header line
            height_val: Optional[float] = None
            with open(vert_path, 'r', encoding='latin1') as fh:
                zpoint_line = next(
                    (ln for ln in fh if 'Zpoint2.z' in ln), None
                )
            if zpoint_line is not None:
                try:
                    raw_height = float(zpoint_line.split('=')[-1].strip())
                    height_val = float(
                        '%s' % float('%.2g' % (raw_height * PIEZO_CONSTANT))
                    )
                except ValueError:
                    print(f"[WARN] Could not parse height from '{filename}'.")

            vertman_offset, data = self._extract_vertman_data(vert_path)
            x_rot, y_rot = rotate_point_around_offset(
                vertman_offset[0],
                vertman_offset[1],
                self.ScanRotation,
                self.offset_A_x,
                self.offset_A_y,
            )

            offset_4_plot = list(
                map(float, np.round([x_rot, y_rot, height_val], 3))
            )

            color = None
            for spec in self.spectra:
                dist = np.linalg.norm(spec['rotated'] - np.array([x_rot, y_rot]))
                if dist <= 0.5:
                    color = spec['color']
                    break
            if color is None:
                color = next(color_cycle)
                
            self.spectra.append({
                'filename': filename,
                'data': data,
                'rotated': np.array([x_rot, y_rot]),
                'offset': offset_4_plot,
                'height': height_val,
                'color': color
            })

        print(f"[OK] Finished loading {len(self.spectra)} spectra.")
