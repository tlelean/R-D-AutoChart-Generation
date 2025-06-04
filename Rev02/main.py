import argparse
from pathlib import Path
import fitz

from data_loading import (
    get_file_paths,
    load_test_information,
    prepare_primary_data,
)
from program_handlers import PROGRAM_HANDLERS


def generate_program_pdf(
    primary_data_path: str,
    test_details_path: str,
    output_pdf_path: str | Path,
    is_gui: bool,
) -> Path | None:
    """Generate a PDF for a single program and return the file path."""
    primary_data_file, test_details_file, pdf_output_path = get_file_paths(
        primary_data_path,
        test_details_path,
        output_pdf_path,
    )

    test_metadata, transducer_details, channels_to_record, additional_info = (
        load_test_information(test_details_file)
    )

    cleaned_data, active_channels = prepare_primary_data(
        primary_data_file,
        channels_to_record,
    )

    program_name = test_metadata.at["Program Name", 1]
    handler = PROGRAM_HANDLERS.get(program_name)
    if handler is None:
        raise NotImplementedError(f"Program '{program_name}' is not supported")

    return handler(
        program_name,
        test_metadata,
        transducer_details,
        active_channels,
        cleaned_data,
        additional_info,
        pdf_output_path,
        is_gui,
    )


def main() -> None:
    """Entry point for generating program PDFs."""
    try:
        parser = argparse.ArgumentParser(description="Process file paths.")
        parser.add_argument(
            "file_path1", type=str, help="Path to the primary data CSV file"
        )
        parser.add_argument(
            "file_path2", type=str, help="Path to the test details CSV file"
        )
        parser.add_argument(
            "file_path3", type=str, help="Path to the PDF Save Location"
        )
        parser.add_argument("is_gui", type=bool, help="GUI or not")
        args = parser.parse_args()

        last_output = generate_program_pdf(
            args.file_path1,
            args.file_path2,
            args.file_path3,
            args.is_gui,
        )

        if not args.is_gui and last_output:
            doc = fitz.open(last_output)
            page = doc.load_page(0)
            zoom_factor = 3
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat)
            pix.save("/var/opt/codesys/PlcLogic/visu/pdf.png")
            doc.close()

        print("Done")
    except Exception as exc:
        print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
