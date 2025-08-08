"""Entry point for generating R&D test reports."""

import fitz
from pathlib import Path
import argparse

from data_loading import (
    get_file_paths,
    load_test_information,
    prepare_primary_data,
)
from program_handlers import HANDLERS

def main():
    """
    Main function to parse command-line arguments, process CSV data, 
    generate a plot, and export a PDF report combining text + images.
    """
    try:
        # # Comment out to test
        # parser = argparse.ArgumentParser(description="Process file paths.")
        # parser.add_argument("file_path1", type=str, help="Path to the primary data CSV file")
        # parser.add_argument("file_path2", type=str, help="Path to the test details CSV file")
        # parser.add_argument("file_path3", type=str, help="Path to the PDF Save Location")
        # parser.add_argument("is_gui", type=bool, help="GUI or not")
        # args = parser.parse_args()

        # is_gui = args.is_gui

        # # Gather file paths
        # primary_data_file, test_details_file, pdf_output_path = get_file_paths(args.file_path1, args.file_path2, args.file_path3)

        # For testing purposes, hardcode the file paths
        primary_data_file = (
            "V:/Userdoc/R & D/DAQ_Station/tlelean/250752/60910-283/Attempt 1/CSV/X.X/X.X._Data_7-8-2025_16-46-30.csv"
        )
        test_details_file = (
            "V:/Userdoc/R & D/DAQ_Station/tlelean/250752/60910-283/Attempt 1/CSV/X.X/X.X._Test_Details_7-8-2025_16-46-30.csv"
        )
        pdf_output_path = Path(
            "V:/Userdoc/R & D/DAQ_Station/tlelean/250752/60910-283/Attempt 1/PDF"
        )

        is_gui = True

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

        handler = HANDLERS.get(program_name)
        if handler is None:
            raise ValueError(f"Unsupported program: {program_name}")

        unique_pdf_output_path = handler(
            program_name=program_name,
            pdf_output_path=pdf_output_path,
            test_metadata=test_metadata,
            transducer_details=transducer_details,
            active_channels=active_channels,
            cleaned_data=cleaned_data,
            raw_data=raw_data,
            additional_info=additional_info,
            channels_to_record=channels_to_record,
            is_gui=is_gui
        )

        # Extra copy if not GUI
        if not is_gui:
            paths = (
                unique_pdf_output_path
                if isinstance(unique_pdf_output_path, list)
                else [unique_pdf_output_path]
            )
            for idx, pdf_path in enumerate(paths, start=1):
                doc = fitz.open(pdf_path)
                page = doc.load_page(0)       # or doc[0]
                zoom_factor = 3
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)       # get the rasterised page
                pix.save(f"/var/opt/codesys/PlcLogic/visu/pdf.png")
                doc.close()

        print("Done")

    except Exception as exc:
        print(f"An error occurred: {exc}")

if __name__ == "__main__":
    main()