"""Program specific handlers for creating PDFs."""

from pathlib import Path
from typing import Any, Callable, Dict, List
import math
import pandas as pd

from graph_plotter import (
    plot_channel_data,
    plot_crosses,
)
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
        self.channels_to_record = kwargs.get("channels_to_record")
        self.channel_map = kwargs.get("channel_map")

    def build_output_path(self, test_metadata) -> Path:
        """Construct the output PDF path from metadata."""
        return self.pdf_output_path / (
            f"{test_metadata.at['Test Section Number', 1]}_"
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
    def generate(self) -> Path:
        title_prefix = self.test_metadata.at['Test Section Number', 1]
        if len(self.additional_info) > 1:
            for index in self.additional_info.index:
                self._generate_single_hold_report(title_prefix, index)
        else:
            self._generate_single_hold_report(title_prefix)
        return self.build_output_path(self.test_metadata)

    def _generate_single_hold_report(self, title_prefix, index=None):
        is_table = True
        if index is not None:
            self.test_metadata.at['Test Section Number', 1] = f"{title_prefix}.{index + 1}"
            single_info = self.additional_info.loc[[index]]
        else:
            single_info = self.additional_info

        unique_path = self.build_output_path(self.test_metadata)
        holds_indices, holds_values = locate_key_time_rows(self.cleaned_data, single_info)

        figure, axes, axis_map = plot_channel_data(
            active_channels=self.active_channels,
            cleaned_data=self.cleaned_data,
            channels_to_record=self.channels_to_record,
            is_table=is_table,
            channel_map=self.channel_map,
        )
        plot_crosses(
            df=holds_indices,
            channel=holds_values.at[0, 2],
            data=self.cleaned_data,
            ax=axes[axis_map["Pressure"]],
        )
        pdf = draw_test_details(
            self.test_metadata, self.transducer_details, self.active_channels,
            self.cleaned_data, unique_path, is_table, self.raw_data
        )
        draw_table(pdf_canvas=pdf, dataframe=single_info)
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
            remaining_cycles = [c for c in all_cycles if c not in set().union(*grouped_cycles)]

            total_group_pages = sum(1 for grp in grouped_cycles if grp)
            total_remaining_pages = math.ceil(len(remaining_cycles) / 40)
            total_pages = total_group_pages + total_remaining_pages

            page_idx = 1
            for group in grouped_cycles:
                if not group: continue
                path = self._generate_grouped_cycle_page(
                    group, page_idx, total_pages, base_section, cycle_ranges, breakout_values, breakout_indices
                )
                generated_paths.append(path)
                page_idx += 1

            for i in range(0, len(remaining_cycles), 40):
                group = remaining_cycles[i:i + 40]
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
            meta, self.transducer_details, self.active_channels, data_slice, unique_path, True, self.raw_data
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
            self.cleaned_data, unique_path, True, self.raw_data
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
            first_cycles, middle_cycles, last_cycles = all_cycles[:3], all_cycles[max(0, (max_cycle // 2) - 1):max(0, (max_cycle // 2) - 1) + 3], all_cycles[-3:]
            selected_cycles = sorted(list(set(first_cycles + middle_cycles + last_cycles)))
            remaining_cycles = [c for c in all_cycles if c not in selected_cycles]

            total_pages = len(selected_cycles) + math.ceil(len(remaining_cycles) / 40)

            page_idx = 1
            for cycle in selected_cycles:
                path = self._generate_single_cycle_page(
                    cycle, page_idx, total_pages, base_section, cycle_ranges, values_df, indices_df, plot_channel, axis_key
                )
                generated_paths.append(path)
                page_idx += 1

            for i in range(0, len(remaining_cycles), 40):
                group = remaining_cycles[i:i + 40]
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
            meta, self.transducer_details, self.active_channels, data_slice, unique_path, True, self.raw_data
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
        )
        for phase in calibration_indices.index:
            positions = calibration_indices.loc[phase].dropna().astype(int).tolist()
            times = self.cleaned_data["Datetime"].iloc[positions]
            values = self.cleaned_data[self.additional_info.at[0, 0]].iloc[positions]
            axes[axis_map["Pressure"]].scatter(
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