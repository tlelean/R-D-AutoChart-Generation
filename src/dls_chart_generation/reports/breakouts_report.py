# -*- coding: utf-8 -*-
"""
Defines the strategy for generating reports for breakout tests.
"""

from typing import List
from pathlib import Path
import math
import pandas as pd

from dls_chart_generation import config
from dls_chart_generation.reports.base_report import ReportStrategy
from dls_chart_generation.graph_plotter import plot_channel_data, plot_crosses
from dls_chart_generation.pdf_helpers import draw_table, insert_plot_and_logo
from dls_chart_generation.additional_info_functions import locate_bto_btc_rows, find_cycle_breakpoints

class BreakoutsReportGenerator(ReportStrategy):
    """
    Generates reports for breakout tests, potentially creating multiple pages
    for a large number of cycles. Overrides the default `generate` method.
    """

    def create_plot(self, is_table: bool):
        """
        This method is not used by this strategy as it overrides the main
        `generate` method. It is implemented to satisfy the abstract base class.
        """
        pass

    def generate(self) -> List[Path]:
        """
        Generates a multi-page report for breakout tests, grouping cycles
        to provide a summary and detailed views.
        """
        breakout_values, breakout_indices = locate_bto_btc_rows(
            self.raw_data, self.additional_info, self.channels_to_record, self.channel_map
        )
        cycle_ranges, max_cycle = find_cycle_breakpoints(self.raw_data, self.channels_to_record, self.channel_map)
        base_section = self.test_metadata.at['Test Name', 1]
        all_cycles = list(range(1, max_cycle + 1))
        generated_paths = []

        if breakout_values is None or breakout_indices is None:
            breakout_values = pd.DataFrame(columns=['Cycle'])
            breakout_indices = pd.DataFrame(columns=['Cycle'])

        if max_cycle >= config.BREAKOUTS_MAX_CYCLES_FOR_SINGLE_PAGE + 1:
            first_cycles = all_cycles[:config.BREAKOUTS_GROUP_SIZE]
            middle_start = max(0, (max_cycle // 2) - 1)
            middle_cycles = all_cycles[middle_start:middle_start + config.BREAKOUTS_GROUP_SIZE]
            last_cycles = all_cycles[-config.BREAKOUTS_GROUP_SIZE:]
            grouped_cycles = [first_cycles, middle_cycles, last_cycles]
            pre_middle_cycles = all_cycles[config.BREAKOUTS_GROUP_SIZE:middle_start]
            post_middle_cycles = all_cycles[middle_start + config.BREAKOUTS_GROUP_SIZE:-config.BREAKOUTS_GROUP_SIZE]
            remaining_segments = [seg for seg in (pre_middle_cycles, post_middle_cycles) if seg]
            total_group_pages = sum(1 for grp in grouped_cycles if grp)
            total_remaining_pages = sum(math.ceil(len(seg) / config.BREAKOUTS_MULTI_CYCLE_PAGE_SIZE) for seg in remaining_segments)
            total_pages = total_group_pages + total_remaining_pages
            page_idx = 1
            for group in grouped_cycles:
                if not group: continue
                path = self._generate_grouped_cycle_page(group, page_idx, total_pages, base_section, cycle_ranges, breakout_values, breakout_indices)
                generated_paths.append(path)
                page_idx += 1
            for seg in remaining_segments:
                for i in range(0, len(seg), config.BREAKOUTS_MULTI_CYCLE_PAGE_SIZE):
                    group = seg[i:i + config.BREAKOUTS_MULTI_CYCLE_PAGE_SIZE]
                    path = self._generate_multi_cycle_page(group, page_idx, total_pages, base_section, cycle_ranges)
                    generated_paths.append(path)
                    page_idx += 1
        else:
            path = self._generate_single_page_report(breakout_values, breakout_indices)
            generated_paths.append(path)
        return generated_paths

    def _generate_grouped_cycle_page(self, group, page_idx, total_pages, base_section, cycle_ranges, breakout_values, breakout_indices):
        """Generates a single page containing a group of cycles."""
        meta = self.test_metadata.copy()
        meta.at['Test Name', 1] = f"{base_section} Cycles {self._cycle_str(group)} (Page {page_idx} of {total_pages})"
        unique_path = self._build_output_path(meta)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, group)
        result_slice = breakout_values[breakout_values['Cycle'].isin(group)]
        index_slice = breakout_indices[breakout_indices['Cycle'].isin(group)]
        figure, axes, axis_map = plot_channel_data(self._channels_for_main_plot(), data_slice, self.channels_to_record, is_table=True, channel_map=self.channel_map)
        plot_crosses(df=index_slice, channel=config.TORQUE_CHANNEL, data=data_slice, ax=axes[axis_map[config.TORQUE_AXIS]])
        pdf = self.create_pdf(unique_path, True)
        draw_table(pdf_canvas=pdf, dataframe=result_slice)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path

    def _generate_multi_cycle_page(self, group, page_idx, total_pages, base_section, cycle_ranges):
        """Generates a single page for a large number of cycles without a table."""
        meta = self.test_metadata.copy()
        meta.at['Test Name', 1] = f"{base_section} Cycles {self._cycle_str(group)} (Page {page_idx} of {total_pages})"
        unique_path = self._build_output_path(meta)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, group)
        figure, _, _ = plot_channel_data(self._channels_for_main_plot(), data_slice, self.channels_to_record, is_table=False, channel_map=self.channel_map)
        pdf = self.create_pdf(unique_path, False)
        insert_plot_and_logo(figure, pdf, False)
        return unique_path

    def _generate_single_page_report(self, breakout_values, breakout_indices):
        """Generates a single-page report for a small number of cycles."""
        unique_path = self._build_output_path(self.test_metadata)
        figure, axes, axis_map = plot_channel_data(self._channels_for_main_plot(), self.cleaned_data, self.channels_to_record, is_table=True, channel_map=self.channel_map)
        if self.channels_to_record.at[self.channel_map[config.TORQUE_CHANNEL], 1]:
            plot_crosses(df=breakout_indices, channel=config.TORQUE_CHANNEL, data=self.cleaned_data, ax=axes[axis_map[config.TORQUE_AXIS]])
        pdf = self.create_pdf(unique_path, True)
        draw_table(pdf_canvas=pdf, dataframe=breakout_values)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path

    def _cycle_str(self, cycles: List[int]) -> str:
        """Returns a string representation of a list of cycles."""
        if not cycles:
            return ""
        return str(cycles[0]) if len(cycles) == 1 else f"{cycles[0]}-{cycles[-1]}"

    def _slice_data(self, data: pd.DataFrame, cycle_ranges: pd.DataFrame, cycles: List[int]) -> pd.DataFrame:
        """Slices the main dataframe to include only the specified cycles."""
        start_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[0], 'Start Index'].iat[0]
        end_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[-1], 'End Index'].iat[0]
        return data.loc[start_idx:end_idx]