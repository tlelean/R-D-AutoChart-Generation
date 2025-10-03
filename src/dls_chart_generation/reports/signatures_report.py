# -*- coding: utf-8 -*-
"""
Defines the strategy for generating reports for signature tests.
"""

from typing import List
from pathlib import Path
import math
import pandas as pd

from .. import config
from .breakouts_report import BreakoutsReportGenerator
from ..graph_plotter import plot_channel_data, plot_crosses
from ..pdf_helpers import draw_table, insert_plot_and_logo
from ..additional_info_functions import locate_signature_key_points, find_cycle_breakpoints

class SignaturesReportGenerator(BreakoutsReportGenerator):
    """
    Generates reports for signature tests. Inherits from BreakoutsReportGenerator
    and overrides the main `generate` method to use signature-specific data.
    """
    def generate(self) -> List[Path]:
        """
        Generates a multi-page report for signature tests, identifying key
        points in the data and plotting them across various cycles.
        """
        torque_values, torque_indices, actuator_values, actuator_indices = locate_signature_key_points(self.channels_to_record, self.raw_data, self.channel_map)
        cycle_ranges, max_cycle = find_cycle_breakpoints(self.raw_data, self.channels_to_record, self.channel_map)
        base_section = self.test_metadata.at['Test Name', 1]
        all_cycles = list(range(1, max_cycle + 1))
        generated_paths = []

        if self.channels_to_record.at[self.channel_map[config.TORQUE_CHANNEL], 1]:
            values_df, indices_df, plot_channel, axis_key = torque_values, torque_indices, self.channel_map[config.TORQUE_CHANNEL], config.TORQUE_AXIS
        else:
            values_df, indices_df, plot_channel, axis_key = actuator_values, actuator_indices, self.channel_map[config.ACTUATOR_CHANNEL], config.PRESSURE_AXIS

        if max_cycle >= config.BREAKOUTS_MAX_CYCLES_FOR_SINGLE_PAGE + 1:
            middle_start = max(0, (max_cycle // 2) - 1)
            first_cycles = all_cycles[:config.BREAKOUTS_GROUP_SIZE]
            middle_cycles = all_cycles[middle_start:middle_start + config.BREAKOUTS_GROUP_SIZE]
            last_cycles = all_cycles[-config.BREAKOUTS_GROUP_SIZE:]
            grouped_cycles = sorted(set(first_cycles + middle_cycles + last_cycles))
            pre_middle_cycles = all_cycles[config.BREAKOUTS_GROUP_SIZE:middle_start]
            post_middle_cycles = all_cycles[middle_start + config.BREAKOUTS_GROUP_SIZE:-config.BREAKOUTS_GROUP_SIZE]
            remaining_segments = [seg for seg in (pre_middle_cycles, post_middle_cycles) if seg]
            total_pages = len(grouped_cycles) + sum(math.ceil(len(seg) / config.BREAKOUTS_MULTI_CYCLE_PAGE_SIZE) for seg in remaining_segments)
            page_idx = 1
            for cycle in grouped_cycles:
                path = self._generate_single_cycle_page(cycle, page_idx, total_pages, base_section, cycle_ranges, values_df, indices_df, plot_channel, axis_key)
                generated_paths.append(path)
                page_idx += 1
            for seg in remaining_segments:
                for i in range(0, len(seg), config.BREAKOUTS_MULTI_CYCLE_PAGE_SIZE):
                    group = seg[i:i + config.BREAKOUTS_MULTI_CYCLE_PAGE_SIZE]
                    path = self._generate_multi_cycle_page(group, page_idx, total_pages, base_section, cycle_ranges)
                    generated_paths.append(path)
                    page_idx += 1
        else:
            path = self._generate_single_page_report(values_df, indices_df, plot_channel, axis_key)
            generated_paths.append(path)

        return generated_paths

    def _generate_single_cycle_page(self, cycle, page_idx, total_pages, base_section, cycle_ranges, values_df, indices_df, plot_channel, axis_key):
        """Generates a single page for a specific cycle with detailed annotations."""
        meta = self.test_metadata.copy()
        meta.at['Test Section Number', 1] = f"{base_section} Cycle {self._cycle_str([cycle])}"
        meta.at['Test Name', 1] = f"{base_section} Cycle {self._cycle_str([cycle])} (Page {page_idx} of {total_pages})"
        unique_path = self._build_output_path(meta)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, [cycle])
        result_slice = values_df[values_df['Cycle'] == cycle]
        index_slice = indices_df[indices_df['Cycle'] == cycle]
        figure, axes, axis_map = plot_channel_data(self._channels_for_main_plot(), data_slice, self.channels_to_record, is_table=True, channel_map=self.channel_map)
        plot_crosses(df=index_slice, channel=plot_channel, data=data_slice, ax=axes[axis_map[axis_key]])
        pdf = self.create_pdf(unique_path, True)
        draw_table(pdf_canvas=pdf, dataframe=result_slice)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path

    def _generate_single_page_report(self, values_df, indices_df, plot_channel, axis_key):
        """Generates a single-page report for a small number of signature cycles."""
        unique_path = self._build_output_path(self.test_metadata)
        figure, axes, axis_map = plot_channel_data(
            self._channels_for_main_plot(), self.cleaned_data, self.channels_to_record, is_table=True, channel_map=self.channel_map
        )
        if self.channels_to_record.at[self.channel_map[config.TORQUE_CHANNEL], 1]:
            plot_crosses(df=indices_df, channel=plot_channel, data=self.cleaned_data, ax=axes[axis_map[axis_key]])
        pdf = self.create_pdf(unique_path, True)
        draw_table(pdf_canvas=pdf, dataframe=values_df)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path