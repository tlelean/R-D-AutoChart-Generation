"""Entry point for generating R&D test reports."""

import fitz
from pathlib import Path
import argparse
import os

from data_loading import (
    load_test_information,
    prepare_primary_data,
)
from program_handlers import HANDLERS

def process_files_and_generate_report(primary_data_file, test_details_file, pdf_output_path, run_tests):
    """
    Processes a single set of data files to generate a PDF report.
    """
    (
        test_metadata,
        transducer_details,
        channels_to_record,
        additional_info,
        program_name,
    ) = load_test_information(test_details_file)

    cleaned_data, active_channels, raw_data = prepare_primary_data(
        primary_data_file,
        channels_to_record,
    )

    program_name = test_metadata.at['Program Name', 1]
    handler_class = HANDLERS.get(program_name)
    if handler_class is None:
        raise ValueError(f"Unsupported program: {program_name}")

    handler_instance = handler_class(
        program_name=program_name,
        pdf_output_path=pdf_output_path,
        test_metadata=test_metadata,
        transducer_details=transducer_details,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        raw_data=raw_data,
        additional_info=additional_info,
        channels_to_record=channels_to_record,
    )
    unique_pdf_output_path = handler_instance.generate()

    # Extra copy if not GUI
    if not run_tests:
        output_dir = "/var/opt/codesys/PlcLogic/visu"
        output_path = os.path.join(output_dir, "pdf.png")

        if os.path.isdir(output_dir):
            paths = (
                unique_pdf_output_path
                if isinstance(unique_pdf_output_path, list)
                else [unique_pdf_output_path]
            )
            # This currently only saves the first page of the first PDF.
            # This matches the original logic.
            if paths:
                doc = fitz.open(paths[0])
                page = doc.load_page(0)
                zoom_factor = 3
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                pix.save(output_path)
                doc.close()
                print(f"Successfully generated PNG preview at {output_path}")

def main():
    """
    Main function to parse command-line arguments, process CSV data,
    generate a plot, and export a PDF report combining text + images.
    Can run in two modes:
    1. --run-tests: Runs through a list of test cases in test_config.py
    2. Standard: Takes command-line arguments for a single run.
    """
    parser = argparse.ArgumentParser(description="Generate PDF reports from CSV data.")
    parser.add_argument(
        '--run-tests',
        action='store_true',
        help="Run in test mode, processing all files specified in test_config.py"
    )
    parser.add_argument("primary_data_file", type=str, nargs='?', default=None, help="Path to the primary data CSV file")
    parser.add_argument("test_details_file", type=str, nargs='?', default=None, help="Path to the test details CSV file")
    parser.add_argument("pdf_output_path", type=str, nargs='?', default=None, help="Path to the PDF Save Location")

    args = parser.parse_args()

    try:
        if args.run_tests:
            from test_config import TEST_CASES
            print("Running in test mode...")
            for i, test_case in enumerate(TEST_CASES):
                print(f"--- Running Test Case {i+1} ---")
                process_files_and_generate_report(
                    primary_data_file=test_case["primary_data_file"],
                    test_details_file=test_case["test_details_file"],
                    pdf_output_path=Path(test_case["pdf_output_path"]),
                    run_tests=True,
                )
            print("--- Test run complete ---")
        else:
            if not all([args.primary_data_file, args.test_details_file]):
                parser.error("In standard mode, primary_data_file and test_details_file are required.")
            process_files_and_generate_report(
                primary_data_file=args.primary_data_file,
                test_details_file=args.test_details_file,
                pdf_output_path=Path(args.pdf_output_path),
                run_tests=False,
            )

        print("Done")

    except Exception as exc:
        print(f"An error occurred: {exc}")

if __name__ == "__main__":
    main()