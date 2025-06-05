import argparse
from pathlib import Path
import fitz


def str2bool(value: str) -> bool:
    """Convert a string to boolean for argument parsing."""
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}

from data_loading import (
    get_file_paths,
    load_test_information,
    prepare_primary_data,
)
from program_handlers import PROGRAM_HANDLERS


def main() -> None:
    """Entry point for generating program PDFs."""
    try:
        parser = argparse.ArgumentParser(description="Process file paths.")
        parser.add_argument(
            "file_path1",
            nargs="?",
            type=str,
            help="Path to the primary data CSV file",
        )
        parser.add_argument(
            "file_path2",
            nargs="?",
            type=str,
            help="Path to the test details CSV file",
        )
        parser.add_argument(
            "file_path3",
            nargs="?",
            type=str,
            help="Path to the PDF save location",
        )
        parser.add_argument(
            "is_gui",
            nargs="?",
            type=str,
            default="False",
            help="GUI or not",
        )
        parser.add_argument(
            "--example",
            type=str,
            help=(
                "Name of example dataset to load from 'Example Data/Hydraulic'. "
                "When provided, file path arguments are ignored."
            ),
        )
        args = parser.parse_args()

        is_gui = str2bool(args.is_gui)

        if args.example:
            example_dir = (
                Path(__file__).resolve().parent.parent
                / "Example Data"
                / "Hydraulic"
                / args.example
            )
            primary_data_file = example_dir / "primary_data.csv"
            test_details_file = example_dir / "test_details.csv"
            pdf_output_path = example_dir
            is_gui = True
        else:
            if not all([args.file_path1, args.file_path2, args.file_path3]):
                parser.error(
                    "file_path1, file_path2 and file_path3 are required unless --example is used"
                )
            primary_data_file, test_details_file, pdf_output_path = get_file_paths(
                args.file_path1, args.file_path2, args.file_path3
            )

        # # For testing purposes, hardcode the file paths
        # primary_data_file = "V:/Userdoc/R & D/DAQ_Station/jbradley/Attempt /CSV/4.0/4.0_Data_2-6-2025_9-47-6.csv"
        # test_details_file = "V:/Userdoc/R & D/DAQ_Station/jbradley/Attempt /CSV/4.0/4.0_Test_Details_2-6-2025_9-47-6.csv"
        # pdf_output_path = Path("V:/Userdoc/R & D/DAQ_Station/jbradley/Attempt /CSV/4.0")

        # is_gui = True

        test_metadata, transducer_details, channels_to_record, additional_info = load_test_information(
            test_details_file
        )

        cleaned_data, active_channels = prepare_primary_data(
            primary_data_file, channels_to_record
        )

        program_name = test_metadata.at["Program Name", 1]

        handler = PROGRAM_HANDLERS.get(program_name)
        if handler is None:
            raise NotImplementedError(f"Program '{program_name}' is not supported")

        last_output = handler(
            program_name,
            test_metadata,
            transducer_details,
            active_channels,
            cleaned_data,
            additional_info,
            pdf_output_path,
            is_gui,
        )

        if not is_gui and last_output:
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
