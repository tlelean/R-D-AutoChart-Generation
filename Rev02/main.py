import fitz
import argparse
from pathlib import Path
from data_loading import get_file_paths, load_test_information, prepare_primary_data
from pdf_helpers import draw_test_details, insert_plot_and_logo
from graph_plotter import plot_channel_data

def main():
    """
    Main function to parse command-line arguments, process CSV data, 
    generate a plot, and export a PDF report combining text + images.
    """
    try:
        # Comment out to test
        parser = argparse.ArgumentParser(description="Process file paths.")
        parser.add_argument("file_path1", type=str, help="Path to the primary data CSV file")
        parser.add_argument("file_path2", type=str, help="Path to the test details CSV file")
        parser.add_argument("file_path3", type=str, help="Path to the PDF Save Location")
        parser.add_argument("is_gui", type=bool, help="GUI or not")
        args = parser.parse_args()

        is_gui = args.is_gui

        # Gather file paths
        primary_data_file, test_details_file, pdf_output_path = get_file_paths(args.file_path1, args.file_path2, args.file_path3)

        # # For testing purposes, hardcode the file paths
        # primary_data_file = "V:/Userdoc/R & D/DAQ_Station/jbradley///Attempt /CSV/12.0/12.0_Data_3-7-2025_11-58-51.csv"
        # test_details_file = "V:/Userdoc/R & D/DAQ_Station/jbradley///Attempt /CSV/12.0/12.0_Test_Details_3-7-2025_11-58-51.csv"
        # pdf_output_path = Path("V:/Userdoc/R & D/DAQ_Station/jbradley///Attempt /PDF")

        # is_gui = True

        # Load test details + transducer info
        (test_metadata, transducer_details, channels_to_record, additional_info) = load_test_information(test_details_file)

        # Prepare the primary data
        cleaned_data, active_channels = prepare_primary_data(primary_data_file, channels_to_record)

        program_name = test_metadata.at['Program Name', 1]

        #------------------------------------------------------------------------------
        # Program = Initial Cycle
        #------------------------------------------------------------------------------

        if program_name == "Initial Cycle":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)   

        #------------------------------------------------------------------------------
        # Program = Holds-Seat or Holds-Body
        #------------------------------------------------------------------------------  

        elif program_name == "Holds-Seat" or program_name == "Holds-Body":

            test_title_prefix = test_metadata.at['Test Section Number', 1]

            if len(additional_info) > 1:
                for index in additional_info.index:

                    # Update the test metadata with the current test section number
                    test_metadata.at['Test Section Number', 1] = f"{test_title_prefix}.{index + 1}"

                    # Build the final PDF path using metadata
                    unique_pdf_output_path = pdf_output_path / (
                        f"{test_metadata.at['Test Section Number', 1]}_"
                        f"{test_metadata.at['Test Name', 1]}_"
                        f"{test_metadata.at['Date Time', 1]}.pdf"
                        )

                    # Filter the key time points to the current row (as a DataFrame)
                    single_key_time_point = additional_info.loc[[index]]

                    # Create a plot of pressures + temperatures
                    figure, key_time_indicies = plot_channel_data(data=cleaned_data, active_channels=active_channels, program_name=program_name, cleaned_data=cleaned_data, key_time_points=single_key_time_point)

                    # Create the PDF and draw the test details
                    pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, single_key_time_point, program_name)

                    # Add a PNG of the plot to the PDF
                    insert_plot_and_logo(figure, pdf, is_gui)
            else:
                    
                    # Build the final PDF path using metadata
                    unique_pdf_output_path = pdf_output_path / (
                        f"{test_metadata.at['Test Section Number', 1]}_"
                        f"{test_metadata.at['Test Name', 1]}_"
                        f"{test_metadata.at['Date Time', 1]}.pdf"
                        )

                    # Filter the key time points to the current row (as a DataFrame)
                    single_key_time_point = additional_info

                    # Create a plot of pressures + temperatures
                    figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, single_key_time_point)

                    # Create the PDF and draw the test details
                    pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, single_key_time_point, program_name)

                    # Add a PNG of the plot to the PDF
                    insert_plot_and_logo(figure, pdf, is_gui)    

        #------------------------------------------------------------------------------
        # Program = Atmospheric Breakouts
        #------------------------------------------------------------------------------

        elif program_name == "Atmospheric Breakouts":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)   

        #------------------------------------------------------------------------------
        # Program = Atmospheric Cyclic
        #------------------------------------------------------------------------------

        elif program_name == "Atmospheric Cyclic":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)   

        #------------------------------------------------------------------------------
        # Program = Dynamic Cycles PR2
        #------------------------------------------------------------------------------

        elif program_name == "Dynamic Cycles PR2":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)   

        #------------------------------------------------------------------------------
        # Program = Dynamic Cycles Petrobras
        #------------------------------------------------------------------------------

        elif program_name == "Dynamic Cycles Petrobras":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)   

        #------------------------------------------------------------------------------
        # Program = Pulse Cycles
        #------------------------------------------------------------------------------

        elif program_name == "Pulse Cycles":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)  

        #------------------------------------------------------------------------------
        # Program = Signatures
        #------------------------------------------------------------------------------

        elif program_name == "Signatures":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)  

        #------------------------------------------------------------------------------
        # Program = Open-Close
        #------------------------------------------------------------------------------

        elif program_name == "Open-Close":

            # Build the final PDF path using metadata
            unique_pdf_output_path = pdf_output_path / (
                f"{test_metadata.at['Test Section Number', 1]}_"
                f"{test_metadata.at['Test Name', 1]}_"
                f"{test_metadata.at['Date Time', 1]}.pdf"
                )
            
            breakouts = additional_info

            # Create a plot of pressures + temperatures
            figure, key_time_indicies = plot_channel_data(active_channels, program_name, cleaned_data, breakouts)

            # Create the PDF and draw the test details
            pdf = draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, unique_pdf_output_path, key_time_indicies, breakouts, program_name)

            # Add a PNG of the plot to the PDF
            insert_plot_and_logo(figure, pdf, is_gui)   

        #------------------------------------------------------------------------------
        # Program = Number of Turns
        #------------------------------------------------------------------------------

        elif program_name == "Number Of Turns":
            pass

        # Extra copy if not GUI
        if not is_gui:
            doc = fitz.open(unique_pdf_output_path)
            page = doc.load_page(0)       # or doc[0]
            zoom_factor = 3
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat)       # get the rasterised page
            pix.save("/var/opt/codesys/PlcLogic/visu/pdf.png")
            doc.close()

        print("Done")     

    except Exception as exc:
        print(f"An error occurred: {exc}")

if __name__ == "__main__":
    main()