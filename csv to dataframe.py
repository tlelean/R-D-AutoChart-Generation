import argparse
import io
from datetime import datetime
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
            - Path to primary data file.
            - Path to test details file.
            - Path object for the output PDF.
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
        return pd.read_csv(file_path, **kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {file_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File is empty: {file_path}") from exc
    except Exception as exc:
        raise Exception(f"Error reading file {file_path}: {exc}") from exc


def load_test_information(test_details_path, pdf_output_path):
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
        load_csv_file(test_details_path, header=None, index_col=0, usecols=[0, 1], nrows=12)
        .fillna('')
    )
    transducer_details = (
        load_csv_file(
            test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1, 2],
            skiprows=12,
            nrows=21
        ).fillna('')
    )
    channels_to_record = load_csv_file(
        test_details_path,
        header=None,
        usecols=[0, 3],
        skiprows=12,
        nrows=21
    )
    channels_to_record.columns = [0, 1]
    channels_to_record.set_index(0, inplace=True)
    channels_to_record.fillna('', inplace=True)

    # Load key time points (start of stabilisation, hold, etc.)
    key_time_points = pd.read_csv(
    test_details_path,
    skiprows=33,
    parse_dates=["Start of Stabalisation", "Start of Hold", "End of Hold"],  # Replace with the actual column name
    dayfirst=True,  # Indicates dd/mm/yyyy format
    infer_datetime_format=True  # Optimizes datetime parsing
    )

    print(key_time_points)
    key_time_points.columns = ["Main Channel", "Start of Stabalisation", "Start of Hold", "End of Hold"]

    # Build the final PDF path using metadata
    pdf_output_path = pdf_output_path / (
        f"{test_metadata.at['Test Description', 1]} "
        f"{test_metadata.at['Test Title', 1]}.pdf"
    )
    return (
        test_metadata,
        transducer_details,
        channels_to_record,
        key_time_points,
        pdf_output_path
    )


def prepare_primary_data(primary_data_path, channels_to_record):
    """
    Prepare and clean the primary CSV data, creating a combined
    DateTime column in dd/mm/yyyy hh:mm:ss format (with milliseconds).

    Assumptions:
        - 'Date' is dd/mm/yyyy.
        - 'Time' is hh:mm:ss.
        - 'Milliseconds' can be fractional seconds (e.g. 123).
        - Some channels are flagged as True in 'channels_to_record' 
          to indicate relevance.

    Parameters:
        primary_data_path (str): File path to the main data CSV.
        channels_to_record (pd.DataFrame): DataFrame indicating which 
            channels are active (True/False).

    Returns:
        tuple:
            - pd.DataFrame: Filtered DataFrame with a combined 'Datetime' column.
            - list: Names of the channels actually recorded (True).
            - pd.DataFrame: Original loaded data with headers for reference.
    """
    raw_data = load_csv_file(primary_data_path, header=None)

    # Construct full headers: 3 date/time columns + all channels
    date_time_columns = ['Date', 'Time', 'Milliseconds']
    channel_names = channels_to_record.index.tolist()
    all_headers = date_time_columns + channel_names
    raw_data.columns = all_headers

    # Identify which channels are actually recorded
    active_channels = channels_to_record[channels_to_record[1] == True].index.tolist()
    required_columns = date_time_columns + active_channels

    # Extract only the required columns
    data_subset = raw_data.loc[:, raw_data.columns.isin(required_columns)].copy()

    # Combine date/time into a single datetime column
    data_subset['Datetime'] = pd.to_datetime(
        data_subset['Date'].astype(str) + ' ' +
        data_subset['Time'].astype(str) + '.' +
        data_subset['Milliseconds'].astype(str),
        format='%d/%m/%Y %H:%M:%S.%f'
    )

    # Drop original date/time columns
    data_subset.drop(columns=['Date', 'Time', 'Milliseconds'], inplace=True)

    # Ensure 'Datetime' is the first column
    columns_ordered = ['Datetime'] + [col for col in data_subset.columns if col != 'Datetime']
    data_subset = data_subset[columns_ordered]

    return data_subset, active_channels, raw_data


def locate_key_time_rows(cleaned_data, key_time_points):
    """
    Locate the row indexes in 'cleaned_data' that match each 
    of the three key times from 'key_time_points'.

    Key times are expected to be strings in dd/mm/yyyy hh:mm:ss 
    (with optional .xxx milliseconds) format.

    Parameters:
        cleaned_data (pd.DataFrame): DataFrame with a 'Datetime' column 
            containing parsed datetime values.
        key_time_points (pd.DataFrame): DataFrame with columns for 
            Start of Stabalisation, Start of Hold, End of Hold.

    Returns:
        list: The row indexes in 'cleaned_data' corresponding to 
              each key time.
    """
    time_columns = ["Start of Stabalisation", "Start of Hold", "End of Hold"]
    row_indices = []

    for col in time_columns:
        # Only parse if a time is present
        if pd.notnull(key_time_points.at[0, col]):
            parsed_time = pd.to_datetime(
                key_time_points.at[0, col],
                format='%d/%m/%Y %H:%M:%S.%f',
                errors='coerce'
            )
            if not pd.isnull(parsed_time):
                # Find matching rows in the dataset
                matching = cleaned_data.index[cleaned_data['Datetime'] == parsed_time].tolist()
                row_indices.extend(matching)

    return row_indices


def plot_pressure_and_temperature(cleaned_data, key_time_points):
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
    data_for_plot['Datetime'] = pd.to_datetime(data_for_plot['Datetime'])

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

    # Overlay key time points
    key_columns = ["Start of Stabalisation", "Start of Hold", "End of Hold"]
    key_labels = ['SOS', 'SOH', 'EOH']
    for _, row in key_time_points.iterrows():
        main_channel = row['Main Channel']
        if main_channel in data_for_plot.columns:
            # For each key time
            time_label_pairs = []
            for col, lbl in zip(key_columns, key_labels):
                if pd.notnull(row[col]):
                    dt_val = pd.to_datetime(
                        str(row[col]),
                        format='%d/%m/%Y %H:%M:%S.%f',
                        errors='coerce'
                    )
                    time_label_pairs.append((dt_val, lbl))
                else:
                    time_label_pairs.append((None, lbl))

            # Plot each time if it's valid
            for dt_val, lbl in time_label_pairs:
                if dt_val is not None and dt_val in data_for_plot['Datetime'].values:
                    channel_val = data_for_plot.loc[data_for_plot['Datetime'] == dt_val, main_channel].values[0]
                    ax_pressure.scatter(dt_val, channel_val, color='black', marker='x')
                    ax_pressure.text(
                        dt_val,
                        channel_val + (y_max - y_min) * 0.05,
                        f" {lbl}",
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
    plt.show()
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
    figure.savefig(buffer_stream, format='PNG', bbox_inches='tight', dpi=1000)
    buffer_stream.seek(0)
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
                     colour='black', size=10, left_aligned=False):
    """
    Draw text onto the PDF canvas, with options for alignment (left or centre).

    If the text is a Timestamp, it preserves milliseconds precision 
    to 3 decimal places.

    Parameters:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas to draw upon.
        text (str or pd.Timestamp): The text (or datetime) to be printed.
        x (float): Horizontal reference (centre or left).
        y (float): Vertical centre position.
        font (str): Font name (default: Helvetica).
        colour (str): Text colour (default: black).
        size (int): Font size (default: 10).
        left_aligned (bool): If True, text is left-aligned; otherwise it is centred.
    """
    if isinstance(text, (pd.Timestamp, datetime)):
        # Format to dd/mm/yyyy HH:MM:SS.xxx
        text = text.strftime('%d/%m/%Y  %H:%M:%S.%f')[:-3]
    else:
        text = str(text) if text is not None else ""

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
    raw_data,
    key_point_rows
):
    """
    Generate a PDF report including plot images and textual details.

    Parameters:
        pdf_output_path (Path): Final path for saving the PDF report.
        test_metadata (pd.DataFrame): DataFrame containing general test info.
        active_channels (list): Channels that were recorded (True in CSV).
        transducer_details (pd.DataFrame): Mapping of channels to transducers.
        key_time_points (pd.DataFrame): DataFrame of main channel + key times.
        figure_bytes (io.BytesIO): In-memory stream of the PNG figure.
        raw_data (pd.DataFrame): Original CSV data (including Date/Time columns).
        key_point_rows (list): Row indexes that match each key time in raw_data.
    """
    pdf = canvas.Canvas(str(pdf_output_path), pagesize=landscape(A4))
    pdf.setStrokeColor(colors.black)

    # Define typical layout boxes on the PDF page
    PDF_LAYOUT_BOXES = [
        (15, 515, 600, 65),     # Info Top Left
        (15, 66.5, 600, 418.5), # Graph
        (15, 15, 600, 51.5),    # Graph Index
        (630, 240, 197, 35),    # Test Pressures
        (630, 35, 197, 180),    # 3rd Party Stamp
        (630, 300, 197, 185)    # Info Right
    ]

    # Draw each bounding box
    for box in PDF_LAYOUT_BOXES:
        draw_bounding_box(pdf, *box)

    # Primary colours used in the PDF
    light_blue = Color(0.325, 0.529, 0.761)
    black = Color(0, 0, 0)

    # Add a title using the test description and test title
    draw_text_on_pdf(
        pdf,
        f"{test_metadata.at['Test Description', 1]} {test_metadata.at['Test Title', 1]}",
        315,
        500,
        font="Helvetica-Bold",
        size=16
    )
    draw_text_on_pdf(pdf, "Data Recording Equipment Used", 728.5, 475, "Helvetica-Bold", size=12)
    draw_text_on_pdf(pdf, "3rd Party Stamp and Date", 728.5, 45, "Helvetica-Bold", size=12)

    # Prepare a DataFrame of transducers that were actually used
    empty_rows = pd.DataFrame([[''] * 2 for _ in range(14)])
    used_transducers = transducer_details.loc[active_channels].reset_index(drop=True)
    used_transducers.columns = range(used_transducers.shape[1])
    used_transducers = pd.concat([used_transducers, empty_rows], ignore_index=True)

    # Retrieve time data from raw_data for the key points
    time_0 = raw_data.at[key_point_rows[0], 'Time'] if len(key_point_rows) > 0 else ''
    ms_0 = raw_data.at[key_point_rows[0], 'Milliseconds'] if len(key_point_rows) > 0 else ''
    time_1 = raw_data.at[key_point_rows[1], 'Time'] if len(key_point_rows) > 1 else ''
    ms_1 = raw_data.at[key_point_rows[1], 'Milliseconds'] if len(key_point_rows) > 1 else ''
    time_2 = raw_data.at[key_point_rows[2], 'Time'] if len(key_point_rows) > 2 else ''
    ms_2 = raw_data.at[key_point_rows[2], 'Milliseconds'] if len(key_point_rows) > 2 else ''

    # Build static and dynamic text for the PDF
    pdf_text_positions = [
        # Left column data
        (20, 571.875, "Test Procedure Reference", black),
        (140, 571.875, test_metadata.at['Test Procedure Reference', 1], light_blue),
        (20, 555.625, "Unique No.", black),
        (140, 555.625, test_metadata.at['Unique Number', 1], light_blue),
        (20, 539.375, "R&D Reference", black),
        (140, 539.375, test_metadata.at['R&D Reference', 1], light_blue),
        (20, 523.125, "Valve Description", black),
        (140, 523.125, test_metadata.at['Valve Description', 1], light_blue),

        # Right column data (top)
        (402.5, 571.875, "Job No.", black),
        (487.5, 571.875, test_metadata.at['Job Number', 1], light_blue),
        (402.5, 555.625, "Test Description", black),
        (487.5, 555.625, test_metadata.at['Test Description', 1], light_blue),
        (402.5, 539.375, "Test Date", black),
        (487.5, 539.375, raw_data.at[0, 'Date'], light_blue),
        (402.5, 523.125, "Valve Drawing No.", black),
        (487.5, 523.125, test_metadata.at['Valve Drawing Number', 1], light_blue),

        # Pressures & other details
        (635, 266.25, "Test Pressure", black),
        (725, 266.25, test_metadata.at['Test Pressure', 1], light_blue),
        (635, 248.75, "Actuator Pressure", black),
        (725, 248.75, test_metadata.at['Actuator Pressure', 1], light_blue),

        # Data Logger info
        (635, 457.5, "Data Logger", black),
        (725, 457.5, test_metadata.at['Data Logger', 1], light_blue),
        (635, 442.5, "Serial No.", black),
        (725, 442.5, test_metadata.at['Serial Number', 1], light_blue),

        # Transducers used
        (635, 427.5, "Transducers", black),
        (635, 412.5, used_transducers.at[0, 0], light_blue),
        (674.375, 412.5, used_transducers.at[1, 0], light_blue),
        (713.75, 412.5, used_transducers.at[2, 0], light_blue),
        (753.125, 412.5, used_transducers.at[3, 0], light_blue),
        (792.5, 412.5, used_transducers.at[4, 0], light_blue),
        (635, 397.5, used_transducers.at[5, 0], light_blue),
        (674.375, 397.5, used_transducers.at[6, 0], light_blue),
        (713.75, 397.5, used_transducers.at[7, 0], light_blue),
        (753.125, 397.5, used_transducers.at[8, 0], light_blue),
        (792.5, 397.5, used_transducers.at[9, 0], light_blue),
        (635, 382.5, used_transducers.at[10, 0], light_blue),
        (674.375, 382.5, used_transducers.at[11, 0], light_blue),
        (713.75, 382.5, used_transducers.at[12, 0], light_blue),
        (753.125, 382.5, used_transducers.at[13, 0], light_blue),
        (792.5, 382.5, used_transducers.at[14, 0], light_blue),

        # Gauges
        (635, 367.5, "Gauges", black),
        (635, 352.5, used_transducers.at[0, 1], light_blue),
        (685, 352.5, used_transducers.at[1, 1], light_blue),
        (735, 352.5, used_transducers.at[2, 1], light_blue),
        (785, 352.5, used_transducers.at[3, 1], light_blue),
        (635, 337.5, used_transducers.at[4, 1], light_blue),
        (685, 337.5, used_transducers.at[5, 1], light_blue),
        (735, 337.5, used_transducers.at[6, 1], light_blue),
        (785, 337.5, used_transducers.at[7, 1], light_blue),
        (635, 322.5, used_transducers.at[8, 1], light_blue),
        (685, 322.5, used_transducers.at[9, 1], light_blue),
        (735, 322.5, used_transducers.at[10, 1], light_blue),
        (785, 322.5, used_transducers.at[11, 1], light_blue),

        # Torque Transducer
        (635, 307.5, "Torque Transducer", black),
        (725, 307.5, transducer_details.at['Torque', 1], light_blue),

        # Bottom-left stamp
        (635, 22.5, "Operative:", black),
        (725, 22.5, "Operative", light_blue),

        # Key points at the bottom
        (20, 56.5, "Start of Stabalisation", black),
        (120, 56.5,
         f"{raw_data.at[key_point_rows[0], 'Date'] if len(key_point_rows) > 0 else ''}  "
         f"{time_0}.{ms_0}  "
         f"{float(raw_data.at[key_point_rows[0], key_time_points.at[0, 'Main Channel']]) if len(key_point_rows) > 0 else 0:.0f} psi  "
         f"{raw_data.at[key_point_rows[0], 'Ambient Temperature'] if len(key_point_rows) > 0 else ''}\u00B0C",
         light_blue),
        (20, 41.25, "Start of Hold", black),
        (120, 41.25,
         f"{raw_data.at[key_point_rows[1], 'Date'] if len(key_point_rows) > 1 else ''}  "
         f"{time_1}.{ms_1}  "
         f"{float(raw_data.at[key_point_rows[1], key_time_points.at[0, 'Main Channel']]) if len(key_point_rows) > 1 else 0:.0f} psi  "
         f"{raw_data.at[key_point_rows[1], 'Ambient Temperature'] if len(key_point_rows) > 1 else ''}\u00B0C",
         light_blue),
        (20, 25, "End of Hold", black),
        (120, 25,
         f"{raw_data.at[key_point_rows[2], 'Date'] if len(key_point_rows) > 2 else ''}  "
         f"{time_2}.{ms_2}  "
         f"{float(raw_data.at[key_point_rows[2], key_time_points.at[0, 'Main Channel']]) if len(key_point_rows) > 2 else 0:.0f} psi  "
         f"{raw_data.at[key_point_rows[2], 'Ambient Temperature'] if len(key_point_rows) > 2 else ''}\u00B0C",
         light_blue)
    ]

    # Draw text fields on the PDF
    for x_coord, y_coord, text_value, text_colour in pdf_text_positions:
        draw_text_on_pdf(pdf, text_value, x_coord, y_coord, colour=text_colour, size=10, left_aligned=True)

    # Insert the figure
    figure_image = ImageReader(figure_bytes)
    pdf.drawImage(figure_image, 16, 67.5, 598, 416.5, preserveAspectRatio=False, mask='auto')

    # Example overlay image (remove or replace if not required)
    pdf.drawImage('/var/opt/codesys/PlcLogic/R&D_Page_2.png',
                  629, 515, 197, 65, preserveAspectRatio=True, mask='auto')

    pdf.save()


def main():
    """
    Main function to parse command-line arguments, process CSV data, 
    generate a plot, and export a PDF report combining text + images.
    """
    try:
        print('Starting...')
        parser = argparse.ArgumentParser(description="Process file paths.")
        parser.add_argument("file_path1", type=str, help="Path to the primary data CSV file")
        parser.add_argument("file_path2", type=str, help="Path to the test details CSV file")
        parser.add_argument("file_path3", type=str, help="Path to the PDF Save Location")
        args = parser.parse_args()

        # Gather file paths
        primary_data_file, test_details_file, pdf_output_path = get_file_paths(
            args.file_path1, args.file_path2, args.file_path3
        )

        # Load test details + transducer info
        (
            test_metadata,
            transducer_details,
            channels_to_record,
            key_time_points,
            pdf_output_path
        ) = load_test_information(test_details_file, pdf_output_path)

        # Prepare the primary data
        cleaned_data, active_channels, raw_data = prepare_primary_data(
            primary_data_file, 
            channels_to_record
        )

        # Create a plot of pressures + temperatures
        figure = plot_pressure_and_temperature(cleaned_data, key_time_points)

        # Convert figure to an in-memory bytes stream
        figure_stream = convert_figure_to_bytes(figure)

        # Identify row indexes for each key time in the original data
        key_point_rows = locate_key_time_rows(cleaned_data, key_time_points)

        # Generate the final PDF report
        generate_pdf_report(
            pdf_output_path,
            test_metadata,
            active_channels,
            transducer_details,
            key_time_points,
            figure_stream,
            raw_data,
            key_point_rows
        )

    except Exception as exc:
        print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()