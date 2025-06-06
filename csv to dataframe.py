import argparse
import io
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import fitz

KEY_OR_BREAKOUT = None

def get_file_paths(primary_data_path, test_details_path, output_pdf_path):
    """
    Return validated file paths for the primary data, test details, 
    and the output PDF location.

    Parameters:
        primary_data_path (str): Path to the primary data CSV file.
        test_details_path (str): Path to the test details CSV file.
        output_pdf_path (str): Directory or file path for the output PDF.

    Returns:
        tuple: (str, str, Path)
            - Path to primary data file. 'tlelean/Job Number/Valve Drawing Number/CSV/2.1/Test Description_Data_9-1-2025_8-29-47.csv'
            - Path to test details file. 'tlelean/Job Number/Valve Drawing Number/CSV/2.1/Test Description_Test_Details_9-1-2025_8-29-47.csv'
            - Path object for the output PDF. 'tlelean/Job Number/Valve Drawing Number/PDF'
    """
    return (
        primary_data_path,
        test_details_path,
        Path(output_pdf_path)
    )


def load_csv_file(file_path, **kwargs):
    """
    Load a CSV file using pandas, adding error handling to catch 
    common issues (file missing, empty file, parse errors, etc.).

    Parameters:
        file_path (str or Path): The CSV file path to read.
        **kwargs: Additional arguments to pass to pd.read_csv.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
        Exception: For any other reading/formatting errors.
    """
    try:
        return pd.read_csv(file_path, **kwargs, dayfirst=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {file_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File is empty: {file_path}") from exc
    except Exception as exc:
        raise Exception(f"Error reading file {file_path}: {exc}") from exc


def load_test_information(test_details_path, pdf_output_path):
    global KEY_OR_BREAKOUT
    """
    Load test details and channel/transducer information from a CSV.
    Also builds the final PDF path using 'Test Description' and 'Test Title'.

    Parameters:
        test_details_path (str or Path): CSV containing test details.
        pdf_output_path (Path): Base path or directory for saving the PDF.

    Returns:
        tuple: 
            - pd.DataFrame: DataFrame of test details (metadata).
            - pd.DataFrame: Transducer info for each channel.
            - pd.DataFrame: Indicates channels to be recorded (True/False).
            - pd.DataFrame: Key points (start/hold/end).
            - Path: Updated PDF path (including final filename).
    """
    # Load top sections (test metadata and transducer data)
    test_metadata = (
        load_csv_file(
            test_details_path, 
            header=None, 
            index_col=0, 
            usecols=[0, 1], 
            nrows=16)
            .fillna('')
    )

    transducer_details = (
        load_csv_file(
            test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1, 2],
            skiprows=16,
            nrows=21)
            .fillna('')
    )

    channels_to_record = (
        load_csv_file(
            test_details_path,
            header=None,
            usecols=[0, 3],
            skiprows=16,
            nrows=21)
            .fillna('')
    )

    channels_to_record.columns = [0, 1]
    channels_to_record.set_index(0, inplace=True)
    channels_to_record.fillna('', inplace=True)

    breakout_or_key_points = (
        load_csv_file(
            test_details_path,
            header=None,
            usecols=[0],
            skiprows=37,
            nrows=1)
            .fillna('')
    )

    if breakout_or_key_points.iloc[0, 0] == 'Breakouts':
        breakout_values = (
            load_csv_file(
                test_details_path,
                header=None,
                usecols=[0, 1],
                skiprows=38)
                .dropna(how='all')
                .fillna('')
        )
        key_time_points = None

        KEY_OR_BREAKOUT = 'Breakouts'

    elif breakout_or_key_points.iloc[0, 0] == 'Key Points':
        key_time_points = (
            load_csv_file(
                test_details_path,
                usecols=[0, 1, 2, 3],
                skiprows=38)
                .dropna(how='all')
                .fillna('')
        )
        breakout_values = None

        KEY_OR_BREAKOUT = 'Key Points'

    # Build the final PDF path using metadata
    pdf_output_path = pdf_output_path / (
        f"{test_metadata.at['Test Section Number', 1]}_"
        f"{test_metadata.at['Test Name', 1]}_"
        f"{test_metadata.at['Date Time', 1]}.pdf"
    )
    return (
        test_metadata,
        transducer_details,
        channels_to_record,
        key_time_points,
        breakout_values,
        pdf_output_path
    )


def prepare_primary_data(primary_data_path, channels_to_record):
    """
    Prepare and clean the primary CSV data, which now contains a single
    'Datetime' column in dd/mm/yyyy hh:mm:ss.000 format (with milliseconds).

    Assumptions:
        - 'Datetime' is dd/mm/yyyy hh:mm:ss.000.
        - Some channels are flagged as True in 'channels_to_record'
          to indicate relevance.

    Parameters:
        primary_data_path (str): File path to the main data CSV.
        channels_to_record (pd.DataFrame): DataFrame indicating which
            channels are active (True/False).

    Returns:
        tuple:
            - pd.DataFrame: Filtered DataFrame with a parsed 'Datetime' column.
            - list: Names of the channels actually recorded (True).
            - pd.DataFrame: Original loaded data (for reference).
    """
    # Load raw data (assumes the CSV now has Datetime as its first column,
    # followed by the channels in order)
    raw_data = load_csv_file(primary_data_path, header=None, parse_dates=[0]).iloc[:-1]

    # Prepare a list of all expected columns: 'Datetime' + channel names
    date_time_columns = ['Datetime']
    channel_names = channels_to_record.index.tolist()
    all_headers = date_time_columns + channel_names

    # Rename the columns in raw_data
    raw_data.columns = all_headers

    # Identify which channels are actually recorded
    active_channels = channels_to_record[channels_to_record[1] == True].index.tolist()
    required_columns = ['Datetime'] + active_channels

    # Extract only the required columns
    data_subset = raw_data[required_columns].copy()

    # Convert the single 'Datetime' column to a proper datetime type
    # (assuming format dd/mm/yyyy hh:mm:ss.000)
    data_subset['Datetime'] = pd.to_datetime(
        data_subset['Datetime'],
        format='%d/%m/%Y %H:%M:%S.%f',
        errors='coerce',               # in case of any parse issues
    )

    # Ensure 'Datetime' is the first column
    columns_ordered = ['Datetime'] + [col for col in data_subset.columns if col != 'Datetime']
    data_subset = data_subset[columns_ordered]

    return data_subset, active_channels


def locate_key_time_rows(cleaned_data, key_time_points):
    time_columns = ["Start of Stabilisation", "Start of Hold", "End of Hold"]
    key_time_indicies = key_time_points.copy()
    date_time_index = cleaned_data.set_index('Datetime')

    for col in time_columns:
        if (key_time_points.at[0, col]) == '':
            key_time_indicies.at[0, col] = ''
        else:
            key_time_points.at[0, col] = pd.to_datetime(key_time_points.at[0, col], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce', dayfirst=True)
            key_time_indicies.at[0, col] = date_time_index.index.get_indexer([key_time_points.at[0, col]], method='nearest')[0]
    return key_time_indicies

def plot_pressure_and_temperature(cleaned_data, key_time_indicies, key_time_points):
    """
    Plot pressure on the left y-axis and temperature on the right y-axis.
    Also overlay markers for key time points (start/hold/end).

    Parameters:
        cleaned_data (pd.DataFrame): Data containing 'Datetime' plus 
            one or more pressure/temperature channels.
        key_time_points (pd.DataFrame): Contains main channel name 
            and three key times (stabilisation, hold, end).

    Returns:
        matplotlib.figure.Figure: The figure object for further processing.
    """
    data_for_plot = cleaned_data.copy()
    data_for_plot['Datetime'] = pd.to_datetime(data_for_plot['Datetime'], format='%d/%m/%Y %H:%M:%S.%f')

    # Colour mapping for known channels
    CHANNEL_COLOUR_MAP = {
        'Upstream': 'red',
        'Downstream': 'green',
        'Body': 'orange',
        'Actuator': 'black',
        'Hyperbaric': 'lightblue',
        'Backseat': 'yellow',
        'Spring Chamber': 'gold',
        'Primary Stem Seal': 'darkcyan',
        'Secondary Stem Seal': 'darkblue',
        'Relief Port': 'hotpink',
        'BX Port': 'brown',
        'Torque Digital Input': 'teal',
        'Number Of Turns': 'cyan',
        'Motor Speed': 'purple',
        'LVDT': 'limegreen',
        'Torque': 'grey',
        'Ambient Temperature': 'blue',
        'Body Temperature': 'violet',
        'Actuator Temperature': 'darkorange',
        'Chamber Temperature': 'magenta',
        'Hyperbaric Water Temperature': 'turquoise'
    }

    # Identify columns that are temperatures
    TEMPERATURE_CHANNELS = [
        "Ambient Temperature",
        "Body Temperature",
        "Actuator Temperature",
        "Chamber Temperature",
        "Hyperbaric Water Temperature"
    ]

    # Separate columns
    y_columns = [col for col in data_for_plot.columns if col != 'Datetime']

    # Create the main figure and axes
    fig, ax_pressure = plt.subplots(figsize=(11.96, 8.49))
    ax_pressure.set_ylabel('Pressure (psi)', color='red')
    ax_pressure.tick_params(axis='y', colors='red')
    ax_pressure.margins(x=0)
    ax_pressure.spines['top'].set_visible(False)
    ax_pressure.spines['right'].set_visible(False)
    ax_pressure.spines['left'].set_edgecolor('red')
    ax_pressure.spines['left'].set_linewidth(0.5)
    ax_pressure.spines['bottom'].set_edgecolor('black')
    ax_pressure.spines['bottom'].set_linewidth(0.5)
    ax_pressure.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S'))

    # Plot each non-temperature channel on the pressure axis
    for col in y_columns:
        if col not in TEMPERATURE_CHANNELS:
            ax_pressure.plot(
                data_for_plot['Datetime'],
                data_for_plot[col],
                label=col,
                linewidth=1,
                color=CHANNEL_COLOUR_MAP.get(col, 'black')
            )

    # Create a secondary axis for temperature
    ax_temp = ax_pressure.twinx()
    ax_temp.set_ylabel('Temperature (°C)', color='blue')
    ax_temp.tick_params(axis='y', colors='blue')
    ax_temp.set_xlim(ax_pressure.get_xlim())
    ax_temp.set_ylim(-60, 260)
    ax_temp.yaxis.set_major_locator(MultipleLocator(10))
    ax_temp.spines['top'].set_visible(False)
    ax_temp.spines['left'].set_visible(False)
    ax_temp.spines['bottom'].set_visible(False)
    ax_temp.spines['right'].set_edgecolor('blue')
    ax_temp.spines['right'].set_linewidth(0.5)

    # Plot temperature data
    for col in TEMPERATURE_CHANNELS:
        if col in data_for_plot.columns:
            ax_temp.plot(
                data_for_plot['Datetime'],
                data_for_plot[col],
                label=col,
                linewidth=1,
                color=CHANNEL_COLOUR_MAP.get(col, 'blue')
            )

    # Adjust the lower bound of the pressure axis to zero
    y_min, y_max = ax_pressure.get_ylim()
    ax_pressure.set_ylim(0, y_max)

    # Define time range and tick spacing
    x_min, x_max = data_for_plot['Datetime'].min(), data_for_plot['Datetime'].max()
    x_ticks = pd.date_range(start=x_min, end=x_max, periods=10)
    ax_pressure.set_xticks(x_ticks)
    ax_pressure.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S'))

    if KEY_OR_BREAKOUT == 'Key Points':
        # Overlay key time points
        time_columns = ["Start of Stabilisation", "Start of Hold", "End of Hold"]
        key_labels = ['SOS', 'SOH', 'EOH']
        for col in time_columns:
            if key_time_indicies.iloc[0][col] == '':
                continue
            x = cleaned_data['Datetime'].loc[key_time_indicies.iloc[0][col]]
            y = cleaned_data[key_time_points.iloc[0]['Main Channel']].loc[key_time_indicies.iloc[0][col]]      
            ax_pressure.plot(x, y, marker='x', color='black', markersize=10)
            ax_pressure.text(
                x,
                y + (y_max - y_min) * 0.03,
                f" {key_labels[time_columns.index(col)]}",
                color='black',
                fontsize=10,
                ha='center',
                va='bottom'
            )

    # Create a combined legend at the bottom
    lines1, labels1 = ax_pressure.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='lower center',
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def convert_figure_to_bytes(figure):
    """
    Convert a matplotlib Figure to a BytesIO stream (PNG format),
    for later embedding into a PDF.

    Parameters:
        figure (matplotlib.figure.Figure): The matplotlib figure to convert.

    Returns:
        io.BytesIO: A binary stream containing the rendered PNG.
    """
    buffer_stream = io.BytesIO()
    figure.savefig(buffer_stream, format='PNG', bbox_inches='tight', dpi=100)
    buffer_stream.seek(0)
    plt.close(figure)
    return buffer_stream


def draw_bounding_box(pdf_canvas, x, y, width, height):
    """
    Draw a rectangular bounding box on the PDF canvas.

    Parameters:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas.
        x (float): X-coordinate of the lower-left corner.
        y (float): Y-coordinate of the lower-left corner.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
    """
    pdf_canvas.setLineWidth(0.5)
    pdf_canvas.rect(x, y, width, height)


def draw_text_on_pdf(pdf_canvas, text, x, y, font="Helvetica",
                     colour='black', size=10, left_aligned=False, replace_empty=False):
    """
    Draw text onto the PDF canvas, with options for alignment (left or centre).
    
    If the text is a Timestamp, it can optionally preserve only the date (dd/mm/yyyy) 
    or include milliseconds precision to 3 decimal places.

    Parameters:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas to draw upon.
        text (str or pd.Timestamp): The text (or datetime) to be printed.
        x (float): Horizontal reference (centre or left).
        y (float): Vertical centre position.
        font (str): Font name (default: Helvetica).
        colour (str): Text colour (default: black).
        size (int): Font size (default: 10).
        left_aligned (bool): If True, text is left-aligned; otherwise it is centred.
        date_only (bool): If True, only the date (dd/mm/yyyy) is shown for datetime inputs.
    """

    # Convert text to string or set to "" if None
    text = "" if text is None else str(text)

    if replace_empty:
        # If replace_empty is True, replace if empty/whitespace
        text = "N/A" if not text.strip() else text

    pdf_canvas.setFont(font, size)
    text_width = pdf_canvas.stringWidth(text, font, size)
    text_height = size * 0.7  # Approx. 70% of the font size as a typical measure

    if left_aligned:
        draw_x = x
        draw_y = y - (text_height / 2)
    else:
        draw_x = x - (text_width / 2)
        draw_y = y - (text_height / 2)

    pdf_canvas.setFillColor(colour if colour else colors.black)
    pdf_canvas.drawString(draw_x, draw_y, text)
    pdf_canvas.setFillColor(colors.black)  # Reset to black afterwards


def generate_pdf_report(
    pdf_output_path,
    test_metadata,
    active_channels,
    transducer_details,
    key_time_points,
    figure_bytes,
    cleaned_data,
    key_time_indicies,
    breakout_values,
    is_gui
):
    """
    Generate a PDF report including plot images and textual details.
    """
    pdf = canvas.Canvas(str(pdf_output_path), pagesize=landscape(A4))
    pdf.setStrokeColor(colors.black)

    # Define layout boxes
    PDF_LAYOUT_BOXES = [
        (15, 515, 600, 65),     # Info Top Left
        (15, 66.5, 600, 418.5), # Graph
        (15, 15, 600, 51.5),    # Graph Index
        (630, 268.75, 197, 17.5),    # Test Pressures
        (630, 220, 197, 35),    # Breakout Torque
        (630, 35, 197, 180),    # 3rd Party Stamp
        (630, 300, 197, 185)    # Info Right
    ]
    for box in PDF_LAYOUT_BOXES:
        draw_bounding_box(pdf, *box)

    # Colours
    light_blue = Color(0.325, 0.529, 0.761)
    black = Color(0, 0, 0)

    # Title and headers
    draw_text_on_pdf(pdf,
        f"{test_metadata.at['Test Section Number', 1]} {test_metadata.at['Test Name', 1]}",
        315, 500, font="Helvetica-Bold", size=16)
    draw_text_on_pdf(
        pdf,
        cleaned_data.at[0, 'Datetime'].strftime('%d/%m/%Y'),
        487.5, 539.375,
        colour=light_blue,
        left_aligned=True
    )
    draw_text_on_pdf(pdf, "Data Recording Equipment Used", 728.5, 475, "Helvetica-Bold", size=12)
    draw_text_on_pdf(pdf, "3rd Party Stamp and Date", 728.5, 45, "Helvetica-Bold", size=12)

    # Prepare transducers DataFrame
    empty_rows = pd.DataFrame([['', '']] * 14)
    used_transducers = transducer_details.loc[active_channels].reset_index(drop=True)
    used_transducers.columns = [0, 1]
    used_transducers = pd.concat([used_transducers, empty_rows], ignore_index=True)

    if KEY_OR_BREAKOUT != 'Key Points':  # i.e. for Breakouts
        # Check breakout_values for indices 0 to 2 (i.e. first three rows)
        breakout_vals = [str(breakout_values.iat[i, 1]).strip() for i in range(3)]
        if all(val in ['', '0', '0.0'] for val in breakout_vals):
            breakout_display = "N/A"
        else:
            breakout_display = "See Below Graph"
    else:
        breakout_display = None

    # Build static text positions
    pdf_text_positions = [
        # Left column
        (20, 571.875, "Test Procedure Reference", black, False),
        (140, 571.875, test_metadata.at['Test Procedure Reference', 1], light_blue, True),
        (20, 555.625, "Unique No.", black, False),
        (140, 555.625, test_metadata.at['Unique Number', 1], light_blue, True),
        (20, 539.375, "R&D Reference", black, False),
        (140, 539.375, test_metadata.at['R&D Reference', 1], light_blue, True),
        (20, 523.125, "Valve Description", black, False),
        (140, 523.125, test_metadata.at['Valve Description', 1], light_blue, True),

        # Right column
        (402.5, 571.875, "Job No.", black, False),
        (487.5, 571.875, test_metadata.at['Job Number', 1], light_blue, True),
        (402.5, 555.625, "Test Description", black, False),
        (487.5, 555.625, test_metadata.at['Test Section Number', 1], light_blue, True),
        (402.5, 539.375, "Test Date", black, False),
        (402.5, 523.125, "Valve Drawing No.", black, False),
        (487.5, 523.125, test_metadata.at['Valve Drawing Number', 1], light_blue, True),

        # Pressures & torques
        (635, 277.5, "Test Pressure", black, False),
        (725, 277.5, f"{test_metadata.at['Test Pressure', 1]} psi", light_blue, True),
        (635, 246.25, "Breakout Torque", black, False),
        (725, 246.25,
            (f"{test_metadata.at['Breakout Torque', 1]} ft.lbs"
             if KEY_OR_BREAKOUT == 'Key Points' and test_metadata.at['Breakout Torque', 1] not in ['0.0','0'] 
             else breakout_display
            ),
            light_blue, True
        ),
        (635, 228.75, "Running Torque", black, False),
        (725, 228.75,
            f"{test_metadata.at['Running Torque', 1]} ft.lbs"
            if test_metadata.at['Running Torque', 1] not in ['0.0','0'] else "N/A",
            light_blue, True
        ),
        (635, 457.5, "Data Logger", black, False),
        (725, 457.5, test_metadata.at['Data Logger', 1], light_blue, True),
        (635, 442.5, "Serial No.", black, False),
        (725, 442.5, test_metadata.at['Serial Number', 1], light_blue, True),

        (635, 427.5, "Transducers", black, False),
        (635, 367.5, "Gauges", black, False),
    ]

    # Transducers list (first 15 entries)
    for i in range(15):
        x = 635 + (i % 5) * 39.375
        y = 412.5 - (i // 5) * 15
        pdf_text_positions.append((x, y, used_transducers.iat[i, 0], light_blue, False))

    # Gauges list (first 12 entries)
    for i in range(12):
        x = 635 + (i % 4) * 50
        y = 352.5 - (i // 4) * 15
        pdf_text_positions.append((x, y, used_transducers.iat[i, 1], light_blue, False))

    # Torque transducer and stamp
    pdf_text_positions.extend([
        (635, 307.5, "Torque Transducer", black, False),
        (725, 307.5, transducer_details.at['Torque', 1], light_blue, True),
        (635, 22.5, "Operative:", black, False),
        (685, 22.5, test_metadata.at['Operative', 1], light_blue, False),
    ])

    # Conditional key points / breakouts
    if KEY_OR_BREAKOUT == 'Key Points':
        indices = key_time_indicies.iloc[0]
        main_ch = key_time_points.iloc[0]['Main Channel']
        for label, ypos, col in [
            ("Start of Stabilisation", 56.5, 'Start of Stabilisation'),
            ("Start of Hold", 41.25, 'Start of Hold'),
            ("End of Hold", 25, 'End of Hold')
        ]:
            idx = indices[col]
            if idx != '':
                time = cleaned_data.at[idx, 'Datetime'].strftime('%d/%m/%Y %H:%M:%S')
                psi  = int(cleaned_data.at[idx, main_ch])
                temp = cleaned_data.at[idx, 'Ambient Temperature']
                pdf_text_positions.extend([
                    (20, ypos, label, black, False),
                    (120, ypos,
                        f"{time}   {psi} psi   {temp}\u00B0C",
                        light_blue, False
                    )
                ])
    elif KEY_OR_BREAKOUT == 'Breakouts':
        for i in range(3):
            val = breakout_values.iat[i, 1]
            if val not in ['', 0, 0.0]:
                pdf_text_positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                pdf_text_positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    # Draw all text
    for x, y, text, colour, replace_empty in pdf_text_positions:
        draw_text_on_pdf(pdf, text, x, y, colour=colour, size=10, left_aligned=True, replace_empty=replace_empty)

    # Insert plot and logo
    fig_img = ImageReader(figure_bytes)
    pdf.drawImage(fig_img, 16, 67.5, 598, 416.5, preserveAspectRatio=False, mask='auto')
    image_path = 'V:/Userdoc/R & D/Logos/R&D_Page_2.png' if is_gui else '/var/opt/codesys/PlcLogic/R&D_Page_2.png'
    pdf.drawImage(image_path, 629, 515, 197, 65, preserveAspectRatio=True, mask='auto')
    pdf.save()

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
        primary_data_file, test_details_file, pdf_output_path = get_file_paths(
            args.file_path1, args.file_path2, args.file_path3
        )

        # # For testing
        # primary_data_file = "V:/Userdoc/R & D/DAQ_Station/tlelean/Job Number/Valve Drawing Number/Attempt Attempt/CSV/1.0/1.0_Data_30-4-2025_13-51-1.csv"
        # test_details_file = "V:/Userdoc/R & D/DAQ_Station/tlelean/Job Number/Valve Drawing Number/Attempt Attempt/CSV/1.0/1.0_Test_Details_30-4-2025_13-51-1.csv"
        # pdf_output_path = Path('V:/Userdoc/R & D/DAQ_Station/tlelean/Job Number/Valve Drawing Number/Attempt Attempt/CSV/1.0')
        # is_gui = True

        # Load test details + transducer info
        (
            test_metadata,
            transducer_details,
            channels_to_record,
            key_time_points,
            breakout_values,
            pdf_output_path
        ) = load_test_information(test_details_file, pdf_output_path)

        # Prepare the primary data
        cleaned_data, active_channels = prepare_primary_data(
            primary_data_file, 
            channels_to_record
        )

        if KEY_OR_BREAKOUT == 'Key Points' and len(key_time_points) > 1:

            test_title_prefix = test_metadata.at['Test Section Number', 1]

            for index, row in key_time_points.iterrows():
                # Filter the key time points to the current row (as a DataFrame)
                single_key_time_point = key_time_points.loc[[index]]

                # Identify row indexes for each key time in the original data
                key_time_indicies = locate_key_time_rows(cleaned_data, single_key_time_point)

                # Create a plot of pressures + temperatures
                figure = plot_pressure_and_temperature(cleaned_data, key_time_indicies, single_key_time_point)

                # Convert figure to an in-memory bytes stream
                figure_stream = convert_figure_to_bytes(figure)

                # Generate a unique filename for the PDF
                unique_pdf_output_path = pdf_output_path.with_name(
                    f"{pdf_output_path.stem.strip()}.{index + 1}{pdf_output_path.suffix}"
                )

                test_metadata.at['Test Section Number', 1] = f"{test_title_prefix}.{index + 1}"

                # Generate the final PDF report
                generate_pdf_report(
                    unique_pdf_output_path,
                    test_metadata,
                    active_channels,
                    transducer_details,
                    single_key_time_point,
                    figure_stream,
                    cleaned_data,
                    key_time_indicies,
                    breakout_values,
                    is_gui
                )

                # Extra copy if not GUI
                if not is_gui:
                    doc = fitz.open(pdf_output_path)
                    page = doc.load_page(0)       # or doc[0]
                    zoom_factor = 3
                    mat = fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=mat)       # get the rasterised page
                    pix.save("/var/opt/codesys/PlcLogic/visu/pdf.png")
                    doc.close()

        elif KEY_OR_BREAKOUT == 'Breakouts':

            # Identify row indexes for each key time in the original data
            key_time_indicies = key_time_points

            # Create a plot of pressures + temperatures
            figure = plot_pressure_and_temperature(cleaned_data, key_time_indicies, key_time_points)

            # Convert figure to an in-memory bytes stream
            figure_stream = convert_figure_to_bytes(figure)
            
            # Generate the final PDF report
            generate_pdf_report(
                pdf_output_path,
                test_metadata,
                active_channels,
                transducer_details,
                key_time_points,
                figure_stream,
                cleaned_data,
                key_time_indicies,
                breakout_values,
                is_gui
            )

            # Extra copy if not GUI
            if not is_gui:
                doc = fitz.open(pdf_output_path)
                page = doc.load_page(0)       # or doc[0]
                zoom_factor = 3
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)       # get the rasterised page
                pix.save("/var/opt/codesys/PlcLogic/visu/pdf.png")
                doc.close()

        elif KEY_OR_BREAKOUT == 'Key Points':
            # Identify row indexes for each key time in the original data
            key_time_indicies = locate_key_time_rows(cleaned_data, key_time_points)

            # Create a plot of pressures + temperatures
            figure = plot_pressure_and_temperature(cleaned_data, key_time_indicies, key_time_points)

            # Convert figure to an in-memory bytes stream
            figure_stream = convert_figure_to_bytes(figure)
            
            # Generate the final PDF report
            generate_pdf_report(
                pdf_output_path,
                test_metadata,
                active_channels,
                transducer_details,
                key_time_points,
                figure_stream,
                cleaned_data,
                key_time_indicies,
                breakout_values,
                is_gui
            )

            # Extra copy if not GUI
            if not is_gui:
                doc = fitz.open(pdf_output_path)
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