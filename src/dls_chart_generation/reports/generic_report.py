# -*- coding: utf-8 -*-
"""
Defines the strategy for generating a generic report.
"""

from .base_report import ReportStrategy
from ..graph_plotter import plot_channel_data

class GenericReportGenerator(ReportStrategy):
    """
    A strategy for generating a generic report, which includes a single plot
    of the channel data without any additional tables.
    """

    def is_table_report(self) -> bool:
        """This report type does not include a table."""
        return False

    def create_plot(self, is_table: bool):
        """
        Creates the main plot for the generic report.

        Args:
            is_table: A boolean indicating if the report format includes a table.

        Returns:
            The plot figure, axes, and axis map.
        """
        figure, axes, axis_map = plot_channel_data(
            active_channels=self._channels_for_main_plot(),
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
        )
        return figure, axes, axis_map