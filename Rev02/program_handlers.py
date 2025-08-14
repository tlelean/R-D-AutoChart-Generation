"""Program specific handlers for creating PDFs."""

from pathlib import Path
from typing import Any, Callable, Dict
import math
import pandas as pd

from graph_plotter import (
    plot_channel_data,
    plot_crosses,
    #plot_cycle_lines,
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
    calculate_succesful_calibration
)

def build_output_path(base_path: Path, test_metadata) -> Path:
    """Construct the output PDF path from metadata."""
    return base_path / (
        f"{test_metadata.at['Test Section Number', 1]}_"
        f"{test_metadata.at['Test Name', 1]}_"
        f"{test_metadata.at['Date Time', 1]}.pdf"
    )

def handle_generic(
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    channels_to_record,
    is_gui: bool,
    **kwargs,
):
    """Default handler used by many programs."""
    is_table = False

    unique_path = build_output_path(pdf_output_path, test_metadata)

    figure, axes, axis_map = plot_channel_data(
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        channels_to_record = channels_to_record,
        is_table=is_table,
    )

    pdf = draw_test_details(
        test_metadata=test_metadata,
        transducer_details=transducer_details,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        pdf_output_path=unique_path,
        is_table=is_table,
        raw_data=raw_data
    )

    insert_plot_and_logo(figure, pdf, is_gui, is_table)
    return unique_path


def handle_holds(
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    channels_to_record,
    is_gui: bool,
    **kwargs,
):
    """Handler for Holds."""
    title_prefix = test_metadata.at['Test Section Number', 1]

    if len(additional_info) > 1:
        for index in additional_info.index:
            is_table = True
            test_metadata.at['Test Section Number', 1] = f"{title_prefix}.{index + 1}"
            unique_path = build_output_path(pdf_output_path, test_metadata)

            single_info = additional_info.loc[[index]]

            holds_indices, holds_values = locate_key_time_rows(cleaned_data, single_info)

            figure, axes, axis_map = plot_channel_data(
                active_channels=active_channels,
                cleaned_data=cleaned_data,
                channels_to_record = channels_to_record,
                is_table=is_table,
            )

            plot_crosses(
                df=holds_indices,
                channel=holds_values.at[0, 2],
                data=cleaned_data,
                ax=axes[axis_map["Pressure"]],
            )
            
            pdf = draw_test_details(
                test_metadata,
                transducer_details,
                active_channels,
                cleaned_data,
                unique_path,
                is_table,
                raw_data
            )

            draw_table(
                pdf_canvas=pdf,
                dataframe=single_info)

            insert_plot_and_logo(figure, pdf, is_gui, is_table)            
    else:
        is_table=True

        unique_path = build_output_path(pdf_output_path, test_metadata)

        single_info = additional_info

        holds_indices, holds_values = locate_key_time_rows(cleaned_data, additional_info)

        figure, axes, axis_map = plot_channel_data(
            active_channels=active_channels,
            cleaned_data=cleaned_data,
            channels_to_record = channels_to_record,
            is_table=is_table,
        )

        plot_crosses(
            df=holds_indices,
            channel=holds_values.at[0, 2],
            data=cleaned_data,
            ax=axes[axis_map["Pressure"]],
        )
        
        pdf = draw_test_details(
            test_metadata,
            transducer_details,
            active_channels,
            cleaned_data,
            unique_path,
            is_table,
            raw_data
        )

        draw_table(
            pdf_canvas=pdf,
            dataframe=single_info)

        insert_plot_and_logo(figure, pdf, is_gui, is_table)

    return unique_path

def handle_breakouts(
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    channels_to_record,
    is_gui: bool,
    **kwargs,
):
    """Create breakout PDFs – grouped single‑page plots for first, middle and last cycles
    (three cycles per graph), followed by multi‑cycle summary pages (40 cycles each)."""

    # ------------------------------------------------------------------
    # 0. Helper functions
    # ------------------------------------------------------------------
    def cycle_str(cycles):
        if isinstance(cycles, int):
            return str(cycles)
        return str(cycles[0]) if len(cycles) == 1 else f"{cycles[0]}‑{cycles[-1]}"

    def slice_data(cycles):
        start_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[0], 'Start Index'].iat[0]
        end_idx   = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[-1], 'End Index'].iat[0]
        return cleaned_data.loc[start_idx:end_idx]

    # ------------------------------------------------------------------
    # 1. Locate key rows & cycles
    # ------------------------------------------------------------------
    breakout_values, breakout_indices = locate_bto_btc_rows(
        raw_data, additional_info, channels_to_record
    )

    cycle_ranges, max_cycle = find_cycle_breakpoints(raw_data, channels_to_record)

    base_section   = test_metadata.at['Test Name', 1]
    all_cycles     = list(range(1, max_cycle + 1))
    generated_paths = []

    # Fallback if no breakout rows were found
    if breakout_values is None or breakout_indices is None:
        breakout_values  = pd.DataFrame(columns=['Cycle'])
        breakout_indices = pd.DataFrame(columns=['Cycle'])

    # ------------------------------------------------------------------
    # 2. Decide which cycles belong to each group (first, middle, last)
    # ------------------------------------------------------------------
    if max_cycle >= 10:
        first_cycles  = all_cycles[:3]
        middle_start  = max(0, (max_cycle // 2) - 1)
        middle_cycles = all_cycles[middle_start:middle_start + 3]
        last_cycles   = all_cycles[-3:]

        grouped_cycles = [first_cycles, middle_cycles, last_cycles]
        remaining_cycles = [c for c in all_cycles if c not in set().union(*grouped_cycles)]

        total_group_pages = sum(1 for grp in grouped_cycles if grp)  # three, unless dataset very small
        total_remaining_pages = math.ceil(len(remaining_cycles) / 40)
        total_pages = total_group_pages + total_remaining_pages

        # --------------------------------------------------------------
        # 2a. Group‑of‑three pages (first, middle, last)
        # --------------------------------------------------------------
        page_idx = 1
        for group in grouped_cycles:
            if not group:
                continue  # can happen if <3 cycles exist in a section

            meta = test_metadata.copy()
            meta.at['Test Name', 1] = (
                f"{base_section} Cycles {cycle_str(group)} "
                f"(Page {page_idx} of {total_pages})"
            )
            unique_path  = build_output_path(pdf_output_path, meta)
            data_slice   = slice_data(group)
            result_slice = breakout_values [ breakout_values ['Cycle'].isin(group) ]
            index_slice  = breakout_indices[ breakout_indices['Cycle'].isin(group) ]

            is_table = True
            figure, axes, axis_map = plot_channel_data(
                active_channels=active_channels,
                cleaned_data=data_slice,
                channels_to_record=channels_to_record,
                is_table=is_table,
            )

            plot_crosses(
                df=index_slice,
                channel='Torque',
                data=data_slice,
                ax=axes[axis_map['Torque']],
            )

            pdf = draw_test_details(
                meta,
                transducer_details,
                active_channels,
                data_slice,
                unique_path,
                is_table,
                raw_data,
            )

            draw_table(pdf_canvas=pdf, dataframe=result_slice)
            insert_plot_and_logo(figure, pdf, is_gui, is_table)
            generated_paths.append(unique_path)
            page_idx += 1

        # --------------------------------------------------------------
        # 2b. Multi‑cycle pages (40 cycles at a time, no tables)
        # --------------------------------------------------------------
        for i in range(0, len(remaining_cycles), 40):
            group     = remaining_cycles[i:i + 40]
            meta      = test_metadata.copy()
            meta.at['Test Name', 1] = (
                f"{base_section} Cycles {cycle_str(group)} "
                f"(Page {page_idx} of {total_pages})"
            )
            unique_path = build_output_path(pdf_output_path, meta)
            data_slice  = slice_data(group)

            is_table = False
            figure, axes, axis_map = plot_channel_data(
                active_channels=active_channels,
                cleaned_data=data_slice,
                channels_to_record=channels_to_record,
                is_table=is_table,
            )

            pdf = draw_test_details(
                meta,
                transducer_details,
                active_channels,
                data_slice,
                unique_path,
                is_table,
                raw_data,
            )
            insert_plot_and_logo(figure, pdf, is_gui, is_table)
            generated_paths.append(unique_path)
            page_idx += 1

    # ------------------------------------------------------------------
    # 3. Fewer than 10 cycles – one tabled page for everything
    # ------------------------------------------------------------------
    else:
        is_table   = True
        unique_path = build_output_path(pdf_output_path, test_metadata)

        figure, axes, axis_map = plot_channel_data(
            active_channels=active_channels,
            cleaned_data=cleaned_data,
            channels_to_record=channels_to_record,
            is_table=is_table,
        )

        plot_crosses(
            df=breakout_indices,
            channel='Torque',
            data=cleaned_data,
            ax=axes[axis_map['Torque']],
        )

        pdf = draw_test_details(
            test_metadata,
            transducer_details,
            active_channels,
            cleaned_data,
            unique_path,
            is_table,
            raw_data,
        )
        draw_table(pdf_canvas=pdf, dataframe=breakout_values)
        insert_plot_and_logo(figure, pdf, is_gui, is_table)
        generated_paths.append(unique_path)

    return generated_paths

def handle_signatures(
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    channels_to_record,
    is_gui: bool,
    **kwargs,
):
    """Handler for Signatures program."""

    (
        torque_signature_values,
        torque_signature_indices,
        actuator_signature_values,
        actuator_signature_indices,
    ) = locate_signature_key_points(channels_to_record, raw_data)

    cycle_ranges, max_cycle = find_cycle_breakpoints(raw_data, channels_to_record)
    base_section = test_metadata.at['Test Name', 1]
    all_cycles = list(range(1, max_cycle + 1))

    if channels_to_record.at["Torque", 1]:
        values_df = torque_signature_values
        indices_df = torque_signature_indices
        plot_channel = 'Torque'
        axis_key = 'Torque'
    else:
        values_df = actuator_signature_values
        indices_df = actuator_signature_indices
        plot_channel = 'Actuator'
        axis_key = 'Pressure'

    def cycle_str(cycles):
        return str(cycles[0]) if len(cycles) == 1 else f"{cycles[0]}-{cycles[-1]}"

    def slice_data(cycles):
        start_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[0], 'Start Index'].iloc[0]
        end_idx = cycle_ranges.loc[cycle_ranges['Cycle'] == cycles[-1], 'End Index'].iloc[0]
        return cleaned_data.loc[start_idx:end_idx]

    generated_paths = []

    if max_cycle >= 10:
        first_cycles = all_cycles[:3]
        middle_start = max(0, (max_cycle // 2) - 1)
        middle_cycles = all_cycles[middle_start:middle_start + 3]
        last_cycles = all_cycles[-3:]

        selected_cycles = []
        for seq in (first_cycles, middle_cycles, last_cycles):
            for c in seq:
                if c not in selected_cycles:
                    selected_cycles.append(c)

        remaining_cycles = [c for c in all_cycles if c not in selected_cycles]

        total_selected = len(selected_cycles)
        total_remaining = math.ceil(len(remaining_cycles) / 40)
        total_pages = total_selected + total_remaining

        # Single-cycle pages
        for page_idx, cycle in enumerate(selected_cycles, start=1):
            meta = test_metadata.copy()
            meta.at['Test Section Number', 1] = f"{base_section} Cycle {cycle_str([cycle])}"
            meta.at['Test Name', 1] = f"{base_section} Cycle {cycle_str([cycle])} (Page {page_idx} of {total_pages})"
            unique_path = build_output_path(pdf_output_path, meta)

            data_slice = slice_data([cycle])
            result_slice = values_df[values_df['Cycle'] == cycle]
            index_slice = indices_df[indices_df['Cycle'] == cycle]

            is_table=True

            figure, axes, axis_map = plot_channel_data(
                active_channels=active_channels,
                cleaned_data=data_slice,
                channels_to_record = channels_to_record,
                is_table=is_table,
            )

            plot_crosses(
                df=index_slice,
                channel=plot_channel,
                data=data_slice,
                ax=axes[axis_map[axis_key]],
            )

            # plot_cycle_lines(
            #     indices_df=cycle_ranges[cycle_ranges['Cycle'] == cycle],
            #     data=data_slice,
            #     axes=axes,
            # )

            pdf = draw_test_details(
                test_metadata=meta,
                transducer_details=transducer_details,
                active_channels=active_channels,
                cleaned_data=data_slice,
                pdf_output_path=unique_path,
                is_table=is_table,
                raw_data=raw_data
            )

            draw_table(pdf_canvas=pdf, dataframe=result_slice)

            insert_plot_and_logo(figure, pdf, is_gui, is_table)

            generated_paths.append(unique_path)

        # Multi-cycle pages (40 at a time)
        for i in range(0, len(remaining_cycles), 40):
            group = remaining_cycles[i:i + 40]
            page_idx = total_selected + (i // 40) + 1
            meta = test_metadata.copy()
            meta.at['Test Section Number', 1] = f"{base_section} Cycles {cycle_str(group)}"
            meta.at['Test Name', 1] = f"{base_section} Cycles {cycle_str(group)} (Page {page_idx} of {total_pages})"
            unique_path = build_output_path(pdf_output_path, meta)

            data_slice = slice_data(group)

            is_table=False

            figure, axes, axis_map = plot_channel_data(
                active_channels=active_channels,
                cleaned_data=data_slice,
                channels_to_record = channels_to_record,
                is_table=is_table,
            )

            pdf = draw_test_details(
                test_metadata=meta,
                transducer_details=transducer_details,
                active_channels=active_channels,
                cleaned_data=data_slice,
                pdf_output_path=unique_path,
                is_table=is_table,
                raw_data=raw_data
            )

            insert_plot_and_logo(figure, pdf, is_gui, is_table)

            generated_paths.append(unique_path)
    else:
        is_table = True

        unique_path = build_output_path(pdf_output_path, test_metadata)

        if channels_to_record.at["Torque", 1]:
            signature_key_points = torque_signature_values
            signature_indices = torque_signature_indices
        else:
            signature_key_points = actuator_signature_values
            signature_indices = actuator_signature_indices

        figure, axes, axis_map = plot_channel_data(
            active_channels=active_channels,
            cleaned_data=cleaned_data,
            channels_to_record = channels_to_record,
            is_table=is_table,
        )

        if channels_to_record.at["Torque", 1]:
            ax = axes[axis_map["Torque"]]
        else:
            ax = axes[axis_map["Pressure"]]

        plot_crosses(
            df=signature_indices,
            channel=plot_channel,
            data=cleaned_data,
            ax=ax,
        )

        # plot_cycle_lines(
        #     indices_df=cycle_ranges,
        #     data=cleaned_data,
        #     axes=axes,
        # )
        
        pdf = draw_test_details(
            test_metadata=test_metadata,
            transducer_details=transducer_details,
            active_channels=active_channels,
            cleaned_data=cleaned_data,
            pdf_output_path=unique_path,
            is_table=is_table,
            raw_data=raw_data
        )

        draw_table(
            pdf_canvas=pdf,
            dataframe=signature_key_points)
        
        insert_plot_and_logo(figure, pdf, is_gui, is_table)

        generated_paths.append(unique_path)

    return generated_paths

def handle_calibration(
    program_name: str,
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    channels_to_record,
    is_gui: bool,
    **kwargs,
):
    
    is_table=True

    unique_path = build_output_path(pdf_output_path, test_metadata)

    calibration_indices, _ = locate_calibration_points(cleaned_data, additional_info)

    average_values = calculate_succesful_calibration(cleaned_data, calibration_indices, additional_info)

    figure, axes, axis_map = plot_channel_data(
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        channels_to_record = channels_to_record,
        is_table=is_table,
    )

    for phase in calibration_indices.index:
        # grab the non‑NaN integer positions for this phase
        positions = (
            calibration_indices
            .loc[phase]
            .dropna()
            .astype(int)
            .tolist()
        )
        # now pull out the timestamps & values by position
        times  = cleaned_data["Datetime"].iloc[positions]
        values = cleaned_data[additional_info.at[0, 0]].iloc[positions]

        # scatter them all at once
        axes[axis_map["Pressure"]].scatter(
            times,
            values,
            marker='x',
            s=50,
            color='black',
            label=f'calib_{phase}'
        )
    
    pdf = draw_test_details(
        test_metadata,
        transducer_details,
        active_channels,
        cleaned_data,
        unique_path,
        is_table,
        raw_data
    )

    draw_table(
        pdf_canvas=pdf,
        dataframe=average_values)

    insert_plot_and_logo(figure, pdf, is_gui, is_table)

    return unique_path

HANDLERS: Dict[str, Callable[..., Any]] = {
    "Initial Cycle": handle_generic,
    "Data Logger": handle_generic,
    "Atmospheric Breakouts": handle_breakouts,
    "Atmospheric Cyclic": handle_breakouts,
    "Dynamic Cycles PR2": handle_breakouts,
    "Dynamic Cycles Petrobras": handle_breakouts,
    "Pulse Cycles": handle_generic,
    "Signatures": handle_signatures,
    "Holds-Seat": handle_holds,
    "Holds-Body": handle_holds,
    "Holds-Body onto Seat": handle_holds,
    "Open-Close": handle_breakouts,
    "Number Of Turns": lambda *a, **k: None,
    "Calibration": handle_calibration,
}
}
