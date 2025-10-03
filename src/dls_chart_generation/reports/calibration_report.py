# -*- coding: utf-8 -*-
"""
Defines the strategy for generating reports for calibration tests.
"""

from typing import Any
import pandas as pd

from dls_chart_generation import config
from dls_chart_generation.reports.base_report import ReportStrategy
from dls_chart_generation.graph_plotter import plot_channel_data
from dls_chart_generation.plotter_info import CHANNEL_AXIS_NAMES_MAP
from dls_chart_generation.pdf_helpers import draw_table, draw_regression_table
from dls_chart_generation.additional_info_functions import (
    locate_calibration_points,
    calculate_succesful_calibration,
    calculate_calibration_regression,
    evaluate_calibration_thresholds,
)

class CalibrationReportGenerator(ReportStrategy):
    """
    Generates reports for calibration tests. This strategy uses the template
    method from the base class and implements hooks for plotting and tables.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pre_calculate_values()

    def _pre_calculate_values(self):
        """Pre-calculates values needed for plot and table generation."""
        self.calibration_indices, _ = locate_calibration_points(self.cleaned_data, self.additional_info)
        (
            self.average_values,
            counts_series,
            expected_series,
            abs_errors,
        ) = calculate_succesful_calibration(
            self.cleaned_data, self.calibration_indices, self.additional_info
        )
        breach_mask = evaluate_calibration_thresholds(self.average_values, abs_errors)
        has_breach = not breach_mask.empty and breach_mask.to_numpy().any()
        self.regression_coefficients = None
        if has_breach:
            self.regression_coefficients = calculate_calibration_regression(counts_series, expected_series)

    def is_table_report(self) -> bool:
        """This report type includes a table."""
        return True

    def create_plot(self, is_table: bool):
        """
        Creates the main plot for the calibration report, including scatter
        points for calibration phases.
        """
        figure, axes, axis_map = plot_channel_data(
            active_channels=self._channels_for_main_plot(include_mass_spec=True),
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
            lock_temperature_axis=False,
        )
        custom_to_default_map = {v: k for k, v in self.channel_map.items()}
        for phase in self.calibration_indices.index:
            positions = self.calibration_indices.loc[phase].dropna().astype(int).tolist()
            times = self.cleaned_data["Datetime"].iloc[positions]
            channel_name = self.additional_info.at[1, 0]
            default_channel_name = custom_to_default_map.get(channel_name, channel_name)
            axis_type = CHANNEL_AXIS_NAMES_MAP.get(default_channel_name)
            axis_location = axis_map.get(axis_type, "left")
            axis = axes.get(axis_location, axes["left"])
            values = self.cleaned_data[channel_name].iloc[positions]
            axis.scatter(
                times, values, marker=config.CALIBRATION_MARKER_STYLE,
                s=config.CALIBRATION_MARKER_SIZE, color=config.CALIBRATION_MARKER_COLOR,
                label=f'calib_{phase}'
            )
        return figure, axes, axis_map

    def add_extra_content(self, pdf_canvas: Any):
        """Adds the calibration and regression tables to the PDF."""
        draw_table(pdf_canvas=pdf_canvas, dataframe=self.average_values, calibration=True)
        if self.regression_coefficients is not None and not self.regression_coefficients.dropna().empty:
            draw_regression_table(pdf_canvas, self.regression_coefficients)