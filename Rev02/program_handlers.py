"""Program specific handlers for creating PDFs."""

from pathlib import Path
from typing import Any, Callable, Dict

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
)

def build_output_path(base_path: Path, test_metadata) -> Path:
    """Construct the output PDF path from metadata."""
    return base_path / (
        f"{test_metadata.at['Test Section Number', 1]}_"
        f"{test_metadata.at['Test Name', 1]}_"
        f"{test_metadata.at['Date Time', 1]}.pdf"
    )

def handle_generic(
    program_name: str,
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    is_gui: bool,
    **kwargs,
):
    """Default handler used by many programs."""
    unique_path = build_output_path(pdf_output_path, test_metadata)

    figure, axes, axis_map = plot_channel_data(
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        test_metadata=test_metadata,
        results_df=None,
    )

    pdf = draw_test_details(
        test_metadata=test_metadata,
        transducer_details=transducer_details,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        pdf_output_path=unique_path,
    )
    insert_plot_and_logo(figure, pdf, is_gui)
    return unique_path


def handle_holds(
    program_name: str,
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    is_gui: bool,
    **kwargs,
):
    """Handler for Holds."""
    title_prefix = test_metadata.at['Test Section Number', 1]

    if len(additional_info) > 1:
        for index in additional_info.index:
            test_metadata.at['Test Section Number', 1] = f"{title_prefix}.{index + 1}"
            unique_path = build_output_path(pdf_output_path, test_metadata)

            single_info = additional_info.loc[[index]]

            # locate_key_time_rows needs to to produce another dataframe which is the data needed to be drawn underneath the plot

            key_time_indices = locate_key_time_rows(cleaned_data, additional_info)

            figure, axes, axis_map = plot_channel_data(
                active_channels=active_channels,
                cleaned_data=cleaned_data,
                test_metadata=test_metadata,
                results_df=single_info,
            )

            plot_crosses(
                df=key_time_indices,
                channel=key_time_indices.iloc[0]['Main Channel'],
                data=cleaned_data,
                ax=axes[axis_map["Pressure"]],
            )
            
            pdf = draw_test_details(
                test_metadata,
                transducer_details,
                active_channels,
                cleaned_data,
                unique_path,
            )

            draw_table(
                pdf_canvas=pdf,
                dataframe=single_info)

            insert_plot_and_logo(figure, pdf, is_gui)            
    else:
        unique_path = build_output_path(pdf_output_path, test_metadata)

        single_info = additional_info

        # locate_key_time_rows needs to to produce another dataframe which is the data needed to be drawn underneath the plot

        key_time_indices = locate_key_time_rows(cleaned_data, additional_info)

        figure, axes, axis_map = plot_channel_data(
            active_channels=active_channels,
            cleaned_data=cleaned_data,
            test_metadata=test_metadata,
            results_df=single_info,
        )

        plot_crosses(
            df=key_time_indices,
            channel=key_time_indices.iloc[0]['Main Channel'],
            data=cleaned_data,
            ax=axes[axis_map["Pressure"]],
        )
        
        pdf = draw_test_details(
            test_metadata,
            transducer_details,
            active_channels,
            cleaned_data,
            unique_path,
        )

        draw_table(
            pdf_canvas=pdf,
            dataframe=single_info)

        insert_plot_and_logo(figure, pdf, is_gui)

    return unique_path


def handle_breakouts(
    program_name: str,
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    is_gui: bool,
    **kwargs,
):
    """Handler for breakout programs."""
    unique_path = build_output_path(pdf_output_path, test_metadata)

    breakout_values, breakout_indices = locate_bto_btc_rows(raw_data)

    figure, axes, axis_map = plot_channel_data(
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        test_metadata=test_metadata,
        results_df=breakout_values
    )

    plot_crosses(
        df=breakout_indices,
        channel="Torque",
        data=cleaned_data,
        ax=axes[axis_map["Torque"]],
    )
    
    pdf = draw_test_details(
        test_metadata,
        transducer_details,
        active_channels,
        cleaned_data,
        unique_path,
    )

    draw_table(
        pdf_canvas=pdf,
        dataframe=breakout_values)

    insert_plot_and_logo(figure, pdf, is_gui)

    return unique_path

def handle_signatures(
    program_name: str,
    pdf_output_path: Path,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    raw_data,
    additional_info,
    is_gui: bool,
    **kwargs,
):
    """Handler for Signatures program."""
    unique_path = build_output_path(pdf_output_path, test_metadata)

    (
        torque_signature_values,
        torque_signature_indices,
        actuator_signature_values,
        actuator_signature_indices,
    ) = locate_signature_key_points(
        transducer_details,
        raw_data
    )

    if transducer_details.at["Torque", 2] is True:
        signature_key_points = torque_signature_values
        signature_indices = torque_signature_indices
        channel = raw_data["Torque"]
        ax = axes[axis_map["Torque"]]
    else:
        signature_key_points = actuator_signature_values
        signature_indices = actuator_signature_indices
        channel = raw_data["Actuator"]
        ax = axes[axis_map["Pressure"]]

    figure, axes, axis_map = plot_channel_data(
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        test_metadata=test_metadata,
        results_df=signature_key_points
    )

    plot_crosses(
        df=signature_indices,
        channel=channel,
        data=cleaned_data,
        ax=ax,
    )
    
    pdf = draw_test_details(
        test_metadata=test_metadata,
        transducer_details=transducer_details,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        pdf_output_path=unique_path,
    )

    draw_table(
        pdf_canvas=pdf,
        dataframe=signature_key_points)
    
    insert_plot_and_logo(figure, pdf, is_gui)
    
    return unique_path


HANDLERS: Dict[str, Callable[..., Any]] = {
    "Initial Cycle": handle_generic,
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
}
