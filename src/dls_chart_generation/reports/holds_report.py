# -*- coding: utf-8 -*-
"""
Defines the strategy for generating reports for hold tests.
"""

from pathlib import Path
import pandas as pd

from .. import config
from .base_report import ReportStrategy
from ..graph_plotter import plot_channel_data, plot_crosses
from ..pdf_helpers import draw_table, insert_plot_and_logo
from ..additional_info_functions import locate_key_time_rows

class HoldsReportGenerator(ReportStrategy):
    """
    Generate reports for hold tests. This strategy overrides the default
    `generate` method to handle multiple holds within a single test file.
    """
    def generate(self) -> Path:
        """
        Generates a report for each hold specified in the additional info.
        """
        title_prefix = self.test_metadata.at['Test Section Number', 1]
        header = self.additional_info.iloc[[0]]
        group_count = (len(self.additional_info) - 1) // 3

        if group_count > 1:
            for group_idx in range(group_count):
                start = 1 + group_idx * 3
                group = pd.concat(
                    [header, self.additional_info.iloc[start:start + 3]],
                    ignore_index=True,
                )
                self._generate_single_hold_report(title_prefix, group, group_idx)
        else:
            self._generate_single_hold_report(title_prefix, self.additional_info, None)

        # Note: This returns the path for the last generated report, matching original logic.
        return self._build_output_path(self.test_metadata)

    def _generate_single_hold_report(self, title_prefix: str, info_df: pd.DataFrame, group_idx: int = None):
        """Generates a PDF report for a single hold test."""
        is_table = True
        if group_idx is not None:
            self.test_metadata.at['Test Section Number', 1] = f"{title_prefix}.{group_idx + 1}"

        unique_path = self._build_output_path(self.test_metadata)
        holds_indices, display_table = locate_key_time_rows(self.cleaned_data, info_df)

        figure, axes, axis_map = plot_channel_data(
            active_channels=self._channels_for_main_plot(),
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
        )
        plot_crosses(
            df=holds_indices,
            channel=info_df.iloc[0, 2],
            data=self.cleaned_data,
            ax=axes[axis_map[config.PRESSURE_AXIS]],
        )
        pdf = self.create_pdf(unique_path, is_table)
        draw_table(pdf_canvas=pdf, dataframe=display_table)
        insert_plot_and_logo(figure, pdf, is_table)