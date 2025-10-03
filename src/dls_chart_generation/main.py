"""Entry point for generating R&D test reports."""

import sys
import os
from pathlib import Path
import fitz
import argparse

# Allow the script to be run directly by adding the project's source directory to the path
if __name__ == "__main__" and __package__ is None:
    src_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_dir))

from dls_chart_generation.data_loading import DataLoader
from dls_chart_generation.program_handlers import ReportGeneratorFactory
from dls_chart_generation.mass_spec_report import generate_mass_spec_reports
from dls_chart_generation import config


def process_files_and_generate_report(primary_data_file, test_details_file, pdf_output_path, run_tests):
    """
    Processes a single set of data files to generate a PDF report.
    """
    loader = DataLoader(
        primary_data_path=primary_data_file,
        test_details_path=test_details_file
    )
    (
        test_metadata,
        transducer_details,
        channels_to_record,
        part_windows,
        additional_info,
        program_name,
        default_to_custom_map,
        cleaned_data,
        active_channels,
        raw_data,
    ) = loader.load_and_process_data()

    # Use the factory to get the correct report generation strategy
    strategy = ReportGeneratorFactory.get_strategy(
        program_name=program_name,
        pdf_output_path=pdf_output_path,
        test_metadata=test_metadata,
        transducer_details=transducer_details,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        raw_data=raw_data,
        additional_info=additional_info,
        part_windows=part_windows,
        channels_to_record=channels_to_record,
        channel_map=default_to_custom_map,
    )
    unique_pdf_output_path = strategy.generate()

    mass_spec_channel = default_to_custom_map.get(config.MASS_SPECTROMETER_CHANNEL)
    if (
        (part_windows[["Start", "Stop"]].notna().sum().sum() != 0)
        and mass_spec_channel in cleaned_data.columns
    ):
        generate_mass_spec_reports(
            cleaned_data=cleaned_data,
            part_windows=part_windows,
            mass_spec_channel=mass_spec_channel,
            test_metadata=test_metadata,
            transducer_details=transducer_details,
            pdf_output_path=pdf_output_path,
            channels_to_record=channels_to_record,
            channel_map=default_to_custom_map,
            raw_data=raw_data,
        )

    # Extra copy if not GUI
    if not run_tests:
        output_path = os.path.join(config.VISU_OUTPUT_DIR, config.VISU_OUTPUT_FILENAME)

        if os.path.isdir(config.VISU_OUTPUT_DIR):
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
                mat = fitz.Matrix(config.PDF_IMAGE_ZOOM_FACTOR, config.PDF_IMAGE_ZOOM_FACTOR)
                pix = page.get_pixmap(matrix=mat)
                pix.save(output_path)
                doc.close()

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
            from dls_chart_generation.test_config import TEST_CASES
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