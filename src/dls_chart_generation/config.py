# -*- coding: utf-8 -*-
"""Configuration constants for the DLS Chart Generation tool."""

# File Paths
VISU_OUTPUT_DIR = "/var/opt/codesys/PlcLogic/visu"
VISU_OUTPUT_FILENAME = "pdf.png"

# PDF Generation
PDF_IMAGE_ZOOM_FACTOR = 3

# Data Loading - Test Details CSV Parsing
# These values define the structure of the test details CSV file.
METADATA_ROWS = 19
TRANSDUCER_ROWS = 26
CHANNELS_TO_RECORD_SKIP_ROWS = METADATA_ROWS
CHANNELS_TO_RECORD_ROWS = 26
PART_WINDOW_SKIP_ROWS = METADATA_ROWS + TRANSDUCER_ROWS
PART_WINDOW_ROWS = 13
ADDITIONAL_INFO_SKIP_ROWS = PART_WINDOW_SKIP_ROWS + PART_WINDOW_ROWS

# Datetime format for parsing.
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S.%f"

# Channel Names
# Default names for specific channels used in the logic.
MASS_SPECTROMETER_CHANNEL = "Mass Spectrometer"
TORQUE_CHANNEL = "Torque"
ACTUATOR_CHANNEL = "Actuator"
PRESSURE_AXIS = "Pressure"
TORQUE_AXIS = "Torque"


# Report Generation - Breakouts/Signatures
# Configuration for how cycles are grouped for reporting.
BREAKOUTS_MAX_CYCLES_FOR_SINGLE_PAGE = 9  # If <= this, all cycles on one page.
BREAKOUTS_GROUP_SIZE = 3  # Number of cycles to show at start, middle, and end.
BREAKOUTS_MULTI_CYCLE_PAGE_SIZE = 40  # Number of cycles per page for remaining.

# Plotting
# Styling for calibration points on plots.
CALIBRATION_MARKER_STYLE = "x"
CALIBRATION_MARKER_SIZE = 50
CALIBRATION_MARKER_COLOR = "black"

# Thresholds for calibration report highlighting.
CALIBRATION_THRESHOLDS = {
    "Abs Error (µA) - ±3.6 µA": 3.6,
    "Abs Error (mV) - ±0.12 mV": 0.12,
    "Abs Error (mV) - ±1.0 mV": 1.0,
}