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
from mass_spec_report import generate_mass_spec_reports

def process_files_and_generate_report(primary_data_file, test_details_file, pdf_output_path, run_tests):
    """
    Processes a single set of data files to generate a PDF report.
    """
    (
        test_metadata,
        transducers_codes,
        gauge_codes,
        channel_visibility,
        mass_spec_timings,
        holds,
        cycles,
        calibration,
        default_to_custom_map
    ) = load_test_information(test_details_file)

    cleaned_data, active_channels, raw_data = prepare_primary_data(
        primary_data_file,
        channel_visibility,
    )

    program_name = test_metadata["Program Name"]
    handler_class = HANDLERS.get(program_name)
    if handler_class is None:
        raise ValueError(f"Unsupported program: {program_name}")

    handler_instance = handler_class(
        program_name=program_name,
        pdf_output_path=pdf_output_path,
        test_metadata=test_metadata,
        transducer_codes=transducers_codes,
        gauge_codes=gauge_codes,
        channel_visibility=channel_visibility,
        mass_spec_timings=mass_spec_timings,
        holds=holds,
        cycles=cycles,
        calibration=calibration,
        active_channels=active_channels,
        cleaned_data=cleaned_data,
        raw_data=raw_data,
        channel_map=default_to_custom_map,
    )
    unique_pdf_output_path = handler_instance.generate()

    mass_spec_channel = default_to_custom_map.get("Mass Spectrometer")
    if (
        (mass_spec_timings[["start", "stop"]].notna().sum().sum() != 0)
        and mass_spec_channel in cleaned_data.columns
        ):
        generate_mass_spec_reports(
            cleaned_data=cleaned_data,
            mass_spec_timings=mass_spec_timings,
            mass_spec_channel=mass_spec_channel,
            test_metadata=test_metadata,
            transducer_codes=transducers_codes,
            gauge_codes=gauge_codes,
            pdf_output_path=pdf_output_path,
            channel_visibility=channel_visibility,
            channel_map=default_to_custom_map,
            raw_data=raw_data,
        )

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
                with fitz.open(paths[0]) as doc:
                    page = doc.load_page(0)
                    zoom_factor = 3
                    mat = fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=mat)
                    # Write the PNG bytes directly and fsync to guarantee the data is
                    # persisted before the file is consumed elsewhere.
                    with open(output_path, "wb") as image_file:
                        image_file.write(pix.tobytes("png"))
                        image_file.flush()
                        os.fsync(image_file.fileno())

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
        
    except Exception as exc:
        print(f"An error occurred: {exc}")

    finally:
        print("Done")

if __name__ == "__main__":
    main()