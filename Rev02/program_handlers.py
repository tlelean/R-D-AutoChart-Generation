from pathlib import Path
from typing import Callable, Optional

from graph_plotter import plot_channel_data
from pdf_helpers import draw_test_details, insert_plot_and_logo


def build_pdf_path(test_metadata, base_path: Path) -> Path:
    """Construct the PDF output path using metadata."""
    return base_path / (
        f"{test_metadata.at['Test Section Number', 1]}_"
        f"{test_metadata.at['Test Name', 1]}_"
        f"{test_metadata.at['Date Time', 1]}.pdf"
    )


def generate_pdf(
    program_name: str,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    additional_info,
    base_path: Path,
    is_gui: bool,
) -> Path:
    """Generate a single PDF for the given program."""
    output_path = build_pdf_path(test_metadata, base_path)
    figure, key_time_indices = plot_channel_data(
        active_channels, program_name, cleaned_data, additional_info
    )
    pdf = draw_test_details(
        test_metadata,
        transducer_details,
        active_channels,
        cleaned_data,
        output_path,
        key_time_indices,
        additional_info,
        program_name,
    )
    insert_plot_and_logo(figure, pdf, is_gui)
    return output_path


def handle_holds(
    program_name: str,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    additional_info,
    base_path: Path,
    is_gui: bool,
) -> Optional[Path]:
    """Handle the Holds program which may produce multiple PDFs."""
    test_title_prefix = test_metadata.at['Test Section Number', 1]
    last_path: Optional[Path] = None
    for idx in additional_info.index:
        test_metadata.at['Test Section Number', 1] = f"{test_title_prefix}.{idx + 1}"
        single_info = additional_info.loc[[idx]]
        last_path = generate_pdf(
            program_name,
            test_metadata,
            transducer_details,
            active_channels,
            cleaned_data,
            single_info,
            base_path,
            is_gui,
        )
    test_metadata.at['Test Section Number', 1] = test_title_prefix
    return last_path


def handle_generic(
    program_name: str,
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    additional_info,
    base_path: Path,
    is_gui: bool,
) -> Path:
    return generate_pdf(
        program_name,
        test_metadata,
        transducer_details,
        active_channels,
        cleaned_data,
        additional_info,
        base_path,
        is_gui,
    )


PROGRAM_HANDLERS: dict[str, Callable[..., Optional[Path]]] = {
    "Holds": handle_holds,
    "Atmospheric Breakouts": handle_generic,
    "Atmospheric Cyclic": handle_generic,
    "Dynamic Cycles PR2": handle_generic,
    "Dynamic Cycles Petrobras": handle_generic,
    "Pulse Cycles": handle_generic,
    "Signatures": handle_generic,
    "Open-Close": handle_generic,
    "Number Of Turns": handle_generic,
}
