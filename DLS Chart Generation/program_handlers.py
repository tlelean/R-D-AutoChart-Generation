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
    draw_regression_table,
    evaluate_calibration_thresholds,
    build_test_title,
    insert_plot_and_logo,
)
from additional_info_functions import (
    locate_bto_btc_rows,
    locate_key_time_rows,
    locate_signature_key_points,
    find_cycle_breakpoints,
    locate_calibration_points,
    calculate_succesful_calibration,
    calculate_calibration_regression,
    calculate_number_of_turns_table,
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

    def _channels_for_main_plot(self, include_mass_spec: bool = False) -> List[str]:
        """Return the active channels for the primary plot.

        Unless ``include_mass_spec`` is ``True`` the channel mapped from the
        default ``"Mass Spectrometer"`` entry is removed so it does not appear
        on standard report pages.
        """

        channels = list(self.active_channels or [])
        if include_mass_spec:
            return channels

        mass_spec_channel = None
        if self.channel_map:
            mass_spec_channel = self.channel_map.get("Mass Spectrometer")

        if mass_spec_channel and mass_spec_channel in channels:
            channels = [ch for ch in channels if ch != mass_spec_channel]

        return channels
    
    def _is_channel_recorded(self, default_channel: str) -> bool:
        """Return ``True`` when the default channel has recorded data."""

        if not isinstance(self.channels_to_record, pd.DataFrame):
            return False

        if not self.channel_map or default_channel not in self.channel_map:
            return False

        channel_name = self.channel_map[default_channel]
        if channel_name not in self.channels_to_record.index:
            return False

        return bool(self.channels_to_record.at[channel_name, 1])

    def build_output_path(self, test_metadata) -> Path:
        """Construct the output PDF path from metadata."""
        full_name = build_test_title(test_metadata)
        return self.pdf_output_path / (
            f"{full_name}_"
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
            active_channels=self._channels_for_main_plot(),
            cleaned_data=self.cleaned_data,
            test_metadata=self.test_metadata,
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
    
class NumberOfTurnsReportGenerator(BaseReportGenerator):
    def generate(self) -> Path:
        is_table = True
        unique_path = self.build_output_path(self.test_metadata)
        figure, _, _ = plot_channel_data(
            active_channels=self._channels_for_main_plot(),
            cleaned_data=self.cleaned_data,
            test_metadata=self.test_metadata,
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
        table = calculate_number_of_turns_table(
            raw_data=self.raw_data,
            channels_to_record=self.channels_to_record,
            channel_map=self.channel_map,
        )
        draw_table(pdf_canvas=pdf, dataframe=table)
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
            active_channels=self._channels_for_main_plot(),
            cleaned_data=self.cleaned_data,
            test_metadata=self.test_metadata,
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
        all_cycles = list(range(1, max_cycle + 1))
        generated_paths = []

        if breakout_values is None or breakout_indices is None:
            breakout_values = pd.DataFrame(columns=['Cycle'])
            breakout_indices = pd.DataFrame(columns=['Cycle'])

        show_breakout = bool(self.test_metadata.at[("Show Breakout Torque"), 1] == 'TRUE')

        if show_breakout:
            cycles_to_display = all_cycles[-3:] or all_cycles
        else:
            cycles_to_display = all_cycles

        if not cycles_to_display:
            return generated_paths

        path = self._generate_breakout_cycles_page(
            cycles=cycles_to_display,
            cycle_ranges=cycle_ranges,
            breakout_values=breakout_values,
            breakout_indices=breakout_indices,
            show_breakout=show_breakout,
        )
        generated_paths.append(path)

        return generated_paths
    
    def _generate_breakout_cycles_page(self, cycles, cycle_ranges, breakout_values, breakout_indices, show_breakout):
        ordered_cycles = sorted(cycles)
        unique_path = self.build_output_path(self.test_metadata)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, ordered_cycles)

        figure, axes, axis_map = plot_channel_data(
            self._channels_for_main_plot(),
            data_slice,
            self.test_metadata,
            is_table=show_breakout,
            channel_map=self.channel_map,
        )

        if show_breakout:
            result_slice = breakout_values[breakout_values['Cycle'].isin(ordered_cycles)]
            index_slice = breakout_indices[breakout_indices['Cycle'].isin(ordered_cycles)]

            header_row = result_slice.columns.tolist()
            result_slice = pd.concat(
                [pd.DataFrame([header_row], columns=header_row), result_slice],
                ignore_index=True,
            )

            torque_axis_location = axis_map.get('Torque')
            if torque_axis_location and self._is_channel_recorded('Torque'):
                plot_crosses(
                    df=index_slice,
                    channel=self.channel_map['Torque'],
                    data=data_slice,
                    ax=axes[torque_axis_location],
                )

            pdf = draw_test_details(
                self.test_metadata,
                self.transducer_details,
                self.active_channels,
                data_slice,
                unique_path,
                True,
                self.raw_data,
                has_breakout_table=True,
            )
            draw_table(pdf_canvas=pdf, dataframe=result_slice)
            insert_plot_and_logo(figure, pdf, True)
        else:
            pdf = draw_test_details(
                self.test_metadata,
                self.transducer_details,
                self.active_channels,
                data_slice,
                unique_path,
                False,
                self.raw_data,
            )
            insert_plot_and_logo(figure, pdf, False)

        return unique_path

    def _slice_data(self, data, cycle_ranges, cycles):
        start_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[0], 'Start Index'].iat[0]
        end_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[-1], 'End Index'].iat[0]
        return data.loc[start_idx:end_idx]

class SignaturesReportGenerator(BreakoutsReportGenerator):
    def generate(self) -> List[Path]:
        torque_values, torque_indices, actuator_values, actuator_indices = locate_signature_key_points(
            self.channels_to_record, self.raw_data, self.channel_map, self.test_metadata
        )

        cycle_ranges, max_cycle = find_cycle_breakpoints(self.raw_data, self.channels_to_record, self.channel_map)
        all_cycles = list(range(1, max_cycle + 1))
        generated_paths = []
        show_breakout = bool(self.test_metadata.at[("Show Breakout Torque"), 1] == 'TRUE')

        if self._is_channel_recorded('Torque'):
            values_df, indices_df, plot_channel, axis_key = torque_values, torque_indices, self.channel_map['Torque'], 'Torque'
        else:
            values_df, indices_df, plot_channel, axis_key = actuator_values, actuator_indices, self.channel_map['Actuator'], 'Pressure'

        if show_breakout:
            cycles_to_display = all_cycles[-3:] or all_cycles
        else:
            cycles_to_display = all_cycles

        if not cycles_to_display:
            return generated_paths

        path = self._generate_signature_cycles_page(
            cycles=cycles_to_display,
            cycle_ranges=cycle_ranges,
            values_df=values_df,
            indices_df=indices_df,
            plot_channel=plot_channel,
            axis_key=axis_key,
            show_breakout=show_breakout,
        )
        generated_paths.append(path)

        return generated_paths

    def _generate_signature_cycles_page(
        self,
        cycles,
        cycle_ranges,
        values_df,
        indices_df,
        plot_channel,
        axis_key,
        show_breakout,
    ):
        ordered_cycles = sorted(cycles)
        unique_path = self.build_output_path(self.test_metadata)
        data_slice = self._slice_data(self.cleaned_data, cycle_ranges, ordered_cycles)

        figure, axes, axis_map = plot_channel_data(
            self._channels_for_main_plot(),
            data_slice,
            self.test_metadata,
            is_table=show_breakout,
            channel_map=self.channel_map,
        )

        if show_breakout:
            axis_location = axis_map.get(axis_key)
            if axis_location is not None:
                index_slice = indices_df[indices_df['Cycle'].isin(ordered_cycles)]
                plot_crosses(
                    df=index_slice,
                    channel=plot_channel,
                    data=data_slice,
                    ax=axes[axis_location],
                )

            pdf = draw_test_details(
                self.test_metadata,
                self.transducer_details,
                self.active_channels,
                data_slice,
                unique_path,
                True,
                self.raw_data,
                has_breakout_table=True,
            )

            draw_table(pdf_canvas=pdf, dataframe=values_df)
            insert_plot_and_logo(figure, pdf, True)
        else:
            pdf = draw_test_details(
                self.test_metadata,
                self.transducer_details,
                self.active_channels,
                data_slice,
                unique_path,
                False,
                self.raw_data,
            )
            insert_plot_and_logo(figure, pdf, False)

        return unique_path
    
class CalibrationReportGenerator(BaseReportGenerator):
    def generate(self) -> Path:
        is_table = True
        unique_path = self.build_output_path(self.test_metadata)
        calibration_indices, _ = locate_calibration_points(self.cleaned_data, self.additional_info)
        (
            average_values,
            counts_series,
            expected_series,
            abs_errors,
        ) = calculate_succesful_calibration(
            self.cleaned_data,
            calibration_indices,
            self.additional_info,
        )

        breach_mask = evaluate_calibration_thresholds(average_values, abs_errors)
        has_breach = not breach_mask.empty and breach_mask.to_numpy().any()
        regression_coefficients = None
        if has_breach:
            regression_coefficients = calculate_calibration_regression(
                counts_series,
                expected_series,
            )

        figure, axes, axis_map = plot_channel_data(
            active_channels=self._channels_for_main_plot(include_mass_spec=True),
            cleaned_data=self.cleaned_data,
            test_metadata=self.test_metadata,
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
        if regression_coefficients is not None and not regression_coefficients.dropna().empty:
            draw_regression_table(pdf, regression_coefficients)
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
    "Number of Turns": NumberOfTurnsReportGenerator,
    "Calibration": CalibrationReportGenerator,
    "Data Logger": GenericReportGenerator,
}