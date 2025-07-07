from pathlib import Path
from typing import Callable, Dict, Any
from graph_plotter import plot_channel_data, annotate_holds, annotate_breakouts
from pdf_helpers import draw_test_details, insert_plot_and_logo, locate_key_time_rows, locate_bto_btc_rows

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
    figure = plot_channel_data(
        active_channels=active_channels,
        program_name=program_name,
        cleaned_data=cleaned_data,
        raw_data=raw_data,
        additional_info=additional_info,
        test_metadata=test_metadata,
    )
    pdf = draw_test_details(
        test_metadata=test_metadata,
        transducer_details=transducer_details,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        unique_path=unique_path,
        additional_info=additional_info,
        program_name=program_name,
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
    """Handler for Holds-Seat and Holds-Body."""
    title_prefix = test_metadata.at['Test Section Number', 1]

    if len(additional_info) > 1:
        for index in additional_info.index:
            test_metadata.at['Test Section Number', 1] = f"{title_prefix}.{index + 1}"
            unique_path = build_output_path(pdf_output_path, test_metadata)
            single_info = additional_info.loc[[index]]
            figure = plot_channel_data(
                active_channels,
                program_name,
                cleaned_data,
                raw_data,
                single_info,
                test_metadata,
            )
            key_time_indices = locate_key_time_rows(cleaned_data, additional_info)

            annotate_holds(
                axes=axes, 
                cleaned_data=cleaned_data, 
                key_time_indices=key_time_indices)
            
            pdf = draw_test_details(
                test_metadata,
                transducer_details,
                active_channels,
                cleaned_data,
                unique_path,
                single_info,
                program_name,
            )
            insert_plot_and_logo(figure, pdf, is_gui)
    else:
        unique_path = build_output_path(pdf_output_path, test_metadata)
        single_info = additional_info
        figure = plot_channel_data(
            active_channels,
            program_name,
            cleaned_data,
            raw_data,
            single_info,
            test_metadata,
        )
        key_time_indices = locate_key_time_rows(cleaned_data, additional_info)

        annotate_holds(
            axes=axes, 
            cleaned_data=cleaned_data, 
            key_time_indices=key_time_indices)
        
        pdf = draw_test_details(
            test_metadata,
            transducer_details,
            active_channels,
            cleaned_data,
            unique_path,
            single_info,
            program_name,
        )
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
    figure = plot_channel_data(
        active_channels=active_channels,
        program_name=program_name,
        cleaned_data=cleaned_data,
        raw_data=raw_data,
        additional_info=additional_info,
        test_metadata=test_metadata,
    )

    additional_info, bto_indicies, btc_indicies = locate_bto_btc_rows(raw_data, additional_info)

    annotate_breakouts(
        axes=axes, 
        cleaned_data=cleaned_data, 
        bto_indicies=bto_indicies,
        btc_indicies=btc_indicies,
    )
    
    pdf = draw_test_details(
        test_metadata,
        transducer_details,
        active_channels,
        cleaned_data,
        unique_path,
        additional_info,
        program_name,
    )
    insert_plot_and_logo(figure, pdf, is_gui)
    return unique_path


HANDLERS: Dict[str, Callable[..., Any]] = {
    "Initial Cycle": handle_generic,
    "Atmospheric Breakouts": handle_breakouts,
    "Atmospheric Cyclic": handle_breakouts,
    "Dynamic Cycles PR2": handle_breakouts,
    "Dynamic Cycles Petrobras": handle_breakouts,
    "Pulse Cycles": handle_generic,
    "Signatures": handle_generic,
    "Holds-Seat": handle_holds,
    "Holds-Body": handle_holds,
    "Open-Close": handle_breakouts,
    "Number Of Turns": lambda *a, **k: None,
}
