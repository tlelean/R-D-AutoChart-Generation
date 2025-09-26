"""Program specific handlers for creating PDFs."""

from pathlib import Path
from typing import Any, Callable, Dict, List
import math
import pandas as pd

from graph_plotter import (
    plot_channel_data,
    plot_crosses,
)
from plotter_info import CHANNEL_AXIS_NAMES_MAP
from pdf_helpers import (
    draw_table,
    draw_test_details,
    insert_plot_and_logo,
)
from additional_info_functions import (
    locate_bto_btc_rows,
    locate_key_time_rows,
    locate_signature_key_points,
    find_cycle_breakpoints,
    locate_calibration_points,
    calculate_succesful_calibration,
)

class BaseReportGenerator:
    def __init__(self, **kwargs):
        self.program_name = kwargs.get("program_name")
        self.pdf_output_path = kwargs.get("pdf_output_path")
        self.test_metadata = kwargs.get("test_metadata")
        self.transducer_details = kwargs.get("transducer_details")
        self.active_channels = kwargs.get("active_channels")
        self.cleaned_data = kwargs.get("cleaned_data")
        self.raw_data = kwargs.get("raw_data")
        self.additional_info = kwargs.get("additional_info")
        self.part_windows = kwargs.get("part_windows")
        self.channels_to_record = kwargs.get("channels_to_record")
        self.channel_map = kwargs.get("channel_map")

    def build_output_path(self, test_metadata) -> Path:
        """Construct the output PDF path from metadata."""
        return self.pdf_output_path / (
            f"{test_metadata.at['Test Section Number', 1]} "
            f"{test_metadata.at['Test Name', 1]}_"
            f"{test_metadata.at['Date Time', 1]}.pdf"
        )

    def generate(self) -> Path:
        """Generate the report."""
        raise NotImplementedError

class GenericReportGenerator(BaseReportGenerator):
    def generate(self) -> Path:
        is_table = False
        unique_path = self.build_output_path(self.test_metadata)
        figure, _, _ = plot_channel_data(
            active_channels=self.active_channels,
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
        )
        pdf = draw_test_details(
            test_metadata=self.test_metadata,
            transducer_details=self.transducer_details,
            active_channels=self.active_channels,
            cleaned_data=self.cleaned_data,
            pdf_output_path=unique_path,
            is_table=is_table,
            raw_data=self.raw_data,
        )
        insert_plot_and_logo(figure, pdf, is_table)
        return unique_path

class HoldsReportGenerator(BaseReportGenerator):
    """Generate reports for hold tests.

    The ``additional_info`` DataFrame is expected to contain a single header
    row followed by groups of three data rows describing individual holds.
    """

    def generate(self) -> Path:
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
        return self.build_output_path(self.test_metadata)

    def _generate_single_hold_report(self, title_prefix, info_df, group_idx=None):
        is_table = True
        if group_idx is not None:
            self.test_metadata.at['Test Section Number', 1] = (
                f"{title_prefix}.{group_idx + 1}"
            )
        single_info = info_df

        unique_path = self.build_output_path(self.test_metadata)
        holds_indices, display_table = locate_key_time_rows(self.cleaned_data, single_info)

        figure, axes, axis_map = plot_channel_data(
            active_channels=self.active_channels,
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
        )
        plot_crosses(
            df=holds_indices,
            channel=single_info.iloc[0, 2],
            data=self.cleaned_data,
            ax=axes[axis_map["Pressure"]],
        )
        pdf = draw_test_details(
            self.test_metadata, self.transducer_details, self.active_channels,
            self.cleaned_data, unique_path, is_table, self.raw_data
        )
        draw_table(pdf_canvas=pdf, dataframe=display_table)
        insert_plot_and_logo(figure, pdf, is_table)

class BreakoutsReportGenerator(BaseReportGenerator):
    def generate(self) -> List[Path]:
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

        if max_cycle >= 10:
            first_cycles = all_cycles[:3]
            middle_start = max(0, (max_cycle // 2) - 1)
            middle_cycles = all_cycles[middle_start:middle_start + 3]
            last_cycles = all_cycles[-3:]
            grouped_cycles = [first_cycles, middle_cycles, last_cycles]
            pre_middle_cycles = all_cycles[3:middle_start]
            post_middle_cycles = all_cycles[middle_start + 3:-3]
            remaining_segments = [seg for seg in (pre_middle_cycles, post_middle_cycles) if seg]

            total_group_pages = sum(1 for grp in grouped_cycles if grp)
            total_remaining_pages = sum(math.ceil(len(seg) / 40) for seg in remaining_segments)
            total_pages = total_group_pages + total_remaining_pages

            page_idx = 1
            for group in grouped_cycles:
                if not group:
                    continue
                path = self._generate_grouped_cycle_page(
                    group, page_idx, total_pages, base_section, cycle_ranges, breakout_values, breakout_indices
                )
                generated_paths.append(path)
                page_idx += 1

            for seg in remaining_segments:
                for i in range(0, len(seg), 40):
                    group = seg[i:i + 40]
                    path = self._generate_multi_cycle_page(
                        group, page_idx, total_pages, base_section, cycle_ranges
                    )
                    generated_paths.append(path)
                    page_idx += 1
        else:
            path = self._generate_single_page_report(breakout_values, breakout_indices)
            generated_paths.append(path)

        return generated_paths

    def _generate_grouped_cycle_page(self, group, page_idx, total_pages, base_section, cycle_ranges, breakout_values, breakout_indices):
        meta = self.test_metadata.copy()
        meta.at['Test Name', 1] = f"{base_section} Cycles {self._cycle_str(group)} (Page {page_idx} of {total_pages})"
        unique_path = self.build_output_path(meta)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, group)
        result_slice = breakout_values[breakout_values['Cycle'].isin(group)]
        index_slice = breakout_indices[breakout_indices['Cycle'].isin(group)]

        figure, axes, axis_map = plot_channel_data(
            self.active_channels, data_slice, self.channels_to_record, is_table=True, channel_map=self.channel_map
        )
        plot_crosses(df=index_slice, channel='Torque', data=data_slice, ax=axes[axis_map['Torque']])
        pdf = draw_test_details(
            meta, self.transducer_details, self.active_channels, data_slice, unique_path, True, self.raw_data, has_breakout_table=True
        )
        draw_table(pdf_canvas=pdf, dataframe=result_slice)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path

    def _generate_multi_cycle_page(self, group, page_idx, total_pages, base_section, cycle_ranges):
        meta = self.test_metadata.copy()
        meta.at['Test Name', 1] = f"{base_section} Cycles {self._cycle_str(group)} (Page {page_idx} of {total_pages})"
        unique_path = self.build_output_path(meta)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, group)

        figure, _, _ = plot_channel_data(
            self.active_channels, data_slice, self.channels_to_record, is_table=False, channel_map=self.channel_map
        )
        pdf = draw_test_details(
            meta, self.transducer_details, self.active_channels, data_slice, unique_path, False, self.raw_data
        )
        insert_plot_and_logo(figure, pdf, False)
        return unique_path

    def _generate_single_page_report(self, breakout_values, breakout_indices):
        unique_path = self.build_output_path(self.test_metadata)
        figure, axes, axis_map = plot_channel_data(
            self.active_channels, self.cleaned_data, self.channels_to_record, is_table=True, channel_map=self.channel_map
        )
        plot_crosses(df=breakout_indices, channel='Torque', data=self.cleaned_data, ax=axes[axis_map['Torque']])
        pdf = draw_test_details(
            self.test_metadata, self.transducer_details, self.active_channels,
            self.cleaned_data, unique_path, True, self.raw_data, has_breakout_table=True
        )
        draw_table(pdf_canvas=pdf, dataframe=breakout_values)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path

    def _cycle_str(self, cycles):
        if isinstance(cycles, int): return str(cycles)
        return str(cycles[0]) if len(cycles) == 1 else f"{cycles[0]}-{cycles[-1]}"

    def _slice_data(self, data, cycle_ranges, cycles):
        start_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[0], 'Start Index'].iat[0]
        end_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[-1], 'End Index'].iat[0]
        return data.loc[start_idx:end_idx]

class SignaturesReportGenerator(BreakoutsReportGenerator):
    def generate(self) -> List[Path]:
        torque_values, torque_indices, actuator_values, actuator_indices = locate_signature_key_points(
            self.channels_to_record, self.raw_data, self.channel_map
        )
        cycle_ranges, max_cycle = find_cycle_breakpoints(self.raw_data, self.channels_to_record, self.channel_map)
        base_section = self.test_metadata.at['Test Name', 1]
        all_cycles = list(range(1, max_cycle + 1))
        generated_paths = []

        if self.channels_to_record.at[self.channel_map['Torque'], 1]:
            values_df, indices_df, plot_channel, axis_key = torque_values, torque_indices, self.channel_map['Torque'], 'Torque'
        else:
            values_df, indices_df, plot_channel, axis_key = actuator_values, actuator_indices, self.channel_map['Actuator'], 'Pressure'

        if max_cycle >= 10:
            middle_start = max(0, (max_cycle // 2) - 1)
            first_cycles = all_cycles[:3]
            middle_cycles = all_cycles[middle_start:middle_start + 3]
            last_cycles = all_cycles[-3:]
            grouped_cycles = sorted(set(first_cycles + middle_cycles + last_cycles))

            pre_middle_cycles = all_cycles[3:middle_start]
            post_middle_cycles = all_cycles[middle_start + 3:-3]
            remaining_segments = [seg for seg in (pre_middle_cycles, post_middle_cycles) if seg]

            total_pages = len(grouped_cycles) + sum(math.ceil(len(seg) / 40) for seg in remaining_segments)

            page_idx = 1
            for cycle in grouped_cycles:
                path = self._generate_single_cycle_page(
                    cycle, page_idx, total_pages, base_section, cycle_ranges, values_df, indices_df, plot_channel, axis_key
                )
                generated_paths.append(path)
                page_idx += 1

            for seg in remaining_segments:
                for i in range(0, len(seg), 40):
                    group = seg[i:i + 40]
                    path = self._generate_multi_cycle_page(group, page_idx, total_pages, base_section, cycle_ranges)
                    generated_paths.append(path)
                    page_idx += 1
        else:
            path = self._generate_single_page_report(values_df, indices_df, plot_channel, axis_key)
            generated_paths.append(path)

        return generated_paths

    def _generate_single_cycle_page(self, cycle, page_idx, total_pages, base_section, cycle_ranges, values_df, indices_df, plot_channel, axis_key):
        meta = self.test_metadata.copy()
        meta.at['Test Section Number', 1] = f"{base_section} Cycle {self._cycle_str([cycle])}"
        meta.at['Test Name', 1] = f"{base_section} Cycle {self._cycle_str([cycle])} (Page {page_idx} of {total_pages})"
        unique_path = self.build_output_path(meta)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, [cycle])
        result_slice = values_df[values_df['Cycle'] == cycle]
        index_slice = indices_df[indices_df['Cycle'] == cycle]

        figure, axes, axis_map = plot_channel_data(
            self.active_channels, data_slice, self.channels_to_record, is_table=True, channel_map=self.channel_map
        )
        plot_crosses(df=index_slice, channel=plot_channel, data=data_slice, ax=axes[axis_map[axis_key]])
        pdf = draw_test_details(
            meta, self.transducer_details, self.active_channels, data_slice, unique_path, True, self.raw_data, has_breakout_table=True
        )
        draw_table(pdf_canvas=pdf, dataframe=result_slice)
        insert_plot_and_logo(figure, pdf, True)
        return unique_path

class CalibrationReportGenerator(BaseReportGenerator):
    def generate(self) -> Path:
        is_table = True
        unique_path = self.build_output_path(self.test_metadata)
        calibration_indices, _ = locate_calibration_points(self.cleaned_data, self.additional_info)
        average_values = calculate_succesful_calibration(self.cleaned_data, calibration_indices, self.additional_info)

        figure, axes, axis_map = plot_channel_data(
            active_channels=self.active_channels,
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
            lock_temperature_axis=False,
        )
        custom_to_default_map = {v: k for k, v in self.channel_map.items()}
        for phase in calibration_indices.index:
            positions = calibration_indices.loc[phase].dropna().astype(int).tolist()
            times = self.cleaned_data["Datetime"].iloc[positions]
            channel_name = self.additional_info.at[1, 0]
            default_channel_name = custom_to_default_map.get(channel_name, channel_name)
            axis_type = CHANNEL_AXIS_NAMES_MAP.get(default_channel_name)
            axis_location = axis_map.get(axis_type, "left")
            axis = axes.get(axis_location, axes["left"])
            values = self.cleaned_data[channel_name].iloc[positions]
            axis.scatter(
                times, values, marker='x', s=50, color='black', label=f'calib_{phase}'
            )
        
        pdf = draw_test_details(
            self.test_metadata, self.transducer_details, self.active_channels,
            self.cleaned_data, unique_path, is_table, self.raw_data
        )
        draw_table(pdf_canvas=pdf, dataframe=average_values)
        insert_plot_and_logo(figure, pdf, is_table)
        return unique_path

class DoNothingReportGenerator(BaseReportGenerator):
    def generate(self) -> None:
        return None

HANDLERS: Dict[str, Callable[..., Any]] = {
    "Initial Cycle": GenericReportGenerator,
    "Atmospheric Breakouts": BreakoutsReportGenerator,
    "Atmospheric Cyclic": BreakoutsReportGenerator,
    "Dynamic Cycles PR2": BreakoutsReportGenerator,
    "Dynamic Cycles Petrobras": BreakoutsReportGenerator,
    "Pulse Cycles": GenericReportGenerator,
    "Signatures": SignaturesReportGenerator,
    "Holds-Seat": HoldsReportGenerator,
    "Holds-Body": HoldsReportGenerator,
    "Holds-Body onto Seat": HoldsReportGenerator,
    "Open-Close": BreakoutsReportGenerator,
    "Number Of Turns": DoNothingReportGenerator,
    "Calibration": CalibrationReportGenerator,
    "Data Logger": GenericReportGenerator,
}