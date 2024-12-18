import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import lines
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from pathlib import Path
import io
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

# ------------------------
# Configuration
# ------------------------

def define_file_paths():
    """Define file paths for the CSV files."""
    base_dir = Path(r"V:/Userdoc/R & D/DAQ_Station/tlelean/Job Number/Valve Drawing Number/CSV/1.1")
    data_file = base_dir / "Test Description_Data_18-12-2024_16-33-5.csv"
    test_details_file = base_dir / "Test Description_Test_Details_18-12-2024_16-33-5.csv"
    output_pdf_path = base_dir / "output.pdf"
    return data_file, test_details_file, output_pdf_path

# ------------------------
# Functions
# ------------------------

def read_csv_safely(filepath, **kwargs):
    """Read a CSV file with error handling."""
    try:
        return pd.read_csv(filepath, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {e}")

def load_test_details(filepath):
    """Load test details and split into key sections."""
    # Load and process the DataFrames
    test_details = read_csv_safely(filepath, header=None, index_col=0, usecols=[0, 1], nrows=13).fillna('')
    channel_transducers = read_csv_safely(filepath, header=None, index_col=0, usecols=[0, 1, 2], skiprows=13, nrows=21).fillna('')
    channels_recorded = read_csv_safely(filepath, header=None, usecols=[0, 3], skiprows=13, nrows=21)
    channels_recorded.columns = [0, 1]
    channels_recorded.set_index(0, inplace=True)
    channels_recorded.fillna('', inplace=True)

    key_points = read_csv_safely(filepath, usecols=["Main Channel", "Start of Stabalisation", "Start of Hold", "End of Hold"], skiprows=34).fillna('')
    return test_details, channel_transducers, channels_recorded, key_points

def process_primary_data(filepath, channels_recorded):
    """Process primary data, generate headers, and filter columns."""
    data = read_csv_safely(filepath, header=None)

    # Define headers
    date_time_headers = ['Date', 'Time', 'Milliseconds']
    channel_names = channels_recorded.index.tolist()
    data_headers = date_time_headers + channel_names
    data.columns = data_headers

    # Identify relevant columns
    true_columns = channels_recorded[channels_recorded[1] == True].index.tolist()
    required_columns = ['Date', 'Time', 'Milliseconds'] + true_columns

    # Explicitly make a copy to avoid slice issues
    data_recorded = data.loc[:, data.columns.isin(required_columns)].copy()

    # Combine 'Date', 'Time', and 'Milliseconds' into 'Datetime' column
    data_recorded.loc[:, 'Datetime'] = pd.to_datetime(
        data_recorded['Date'].astype(str) + ' ' +
        data_recorded['Time'].astype(str) + '.' +
        data_recorded['Milliseconds'].astype(str),
        format='%d/%m/%Y %H-%M-%S.%f'
    )

    # Drop unnecessary columns
    data_recorded = data_recorded.drop(columns=['Date', 'Time', 'Milliseconds'])

    # Reorder columns
    columns = ['Datetime'] + [col for col in data_recorded.columns if col != 'Datetime']
    data_recorded = data_recorded[columns]
    return data_recorded, true_columns

def plot_data_with_dual_axes(data_recorded, key_points):
    """
    Plots a DataFrame with a dual y-axis setup:
    - Left Y-axis (red): For pressure or primary data.
    - Right Y-axis (blue): For temperature or secondary data.
    - Overlay key points as markers on the specified channels.

    Parameters:
    - data_recorded: pd.DataFrame with a 'datetime' column and numeric columns for plotting.
    - key_points: pd.DataFrame containing main channels and key time points.
    """
    # Ensure datetime is in datetime format
    data_recorded = data_recorded.copy()
    data_recorded['Datetime'] = pd.to_datetime(data_recorded['Datetime'])

    # Define color mapping for all channels
    color_mapping = {
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

    # Extract column names for plotting
    y_columns = [col for col in data_recorded.columns if col != 'Datetime']

    # Initialize figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 7.5))
    ax1.set_ylabel('Pressure (psi)', color='red')
    ax1.tick_params(axis='y', colors='red')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)   
    ax1.spines['left'].set_edgecolor('red')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_edgecolor('black')
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S'))
    plt.xticks(rotation=0)

    # Plot primary axis data
    for col in y_columns:
        if col not in ["Ambient Temperature", "Body Temperature", "Actuator Temperature", "Chamber Temperature", "Hyperbaric Water Temperature"]:
            ax1.plot(data_recorded['Datetime'], data_recorded[col],
                     label=col, linewidth=1, color=color_mapping.get(col, 'black'))

    # Secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (°C)', color='blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_ylim(-60, 260)
    ax2.yaxis.set_major_locator(MultipleLocator(10))    
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_edgecolor('blue')
    ax2.spines['right'].set_linewidth(0.5)

    # Plot temperature data
    for col in ["Ambient Temperature", "Body Temperature", "Actuator Temperature", "Chamber Temperature", "Hyperbaric Water Temperature"]:
        if col in data_recorded.columns:
            ax2.plot(data_recorded['Datetime'], data_recorded[col],
                     label=col, linewidth=1, color=color_mapping.get(col, 'blue'))

    # Plot key points as markers
    for _, row in key_points.iterrows():
        channel = row['Main Channel']
        if channel in data_recorded.columns:
            # Assuming the date is the same as the first date in data_recorded
            base_date = data_recorded['Datetime'].iloc[0].date()
            times = [
                pd.to_datetime(f"{base_date} {row['Start of Stabalisation']}", format='%Y-%m-%d %H-%M-%S.%f'),
                pd.to_datetime(f"{base_date} {row['Start of Hold']}", format='%Y-%m-%d %H-%M-%S.%f'),
                pd.to_datetime(f"{base_date} {row['End of Hold']}", format='%Y-%m-%d %H-%M-%S.%f')
            ]
            labels = ['SOS', 'SOH', 'EOH']
            values = data_recorded.set_index('Datetime').loc[times, channel].values
            for time, value, label in zip(times, values, labels):
                ax1.scatter(time, value, color='black', marker='x')
                ax1.text(time, value + 0.001, f" {label}", color='black', fontsize=10, ha='center', va='bottom')

    # Custom legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    return fig

def convert_fig_to_image_stream(fig):
    """Converts a matplotlib figure to a BytesIO stream."""
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', dpi = 1000)  # Save the figure with tight bounding box
    buf.seek(0)  # Reset the stream's position to the beginning
    return buf

def draw_rectangle(c, x, y, width, height):
    """Draws a rectangle with the given dimensions."""
    c.rect(x, y, width, height)
    
def draw_centered_text(c, text, x, y, font="Helvetica", color='black', size=None, left_aligned=False):
    """
    Draws text with options for centering or left alignment in both the x and y directions
    with the specified font, color, and size.

    Parameters:
        c: Canvas object for drawing.
        text: Text to be drawn.
        x: X-coordinate of the center or left alignment.
        y: Y-coordinate of the center.
        font: Font of the text.
        color: Color of the text.
        size: Font size of the text.
        left_aligned: If True, aligns text to the left instead of centering.
    """
    c.setFont(font, size)  # Set the font and size
    width = c.stringWidth(text, font, size)  # Get the width of the text

    # Approximate text height based on the font size
    text_height = size * 0.7  # A typical approximation (70% of the font size)

    if left_aligned:
        # For left-aligned text, use x as is
        aligned_x = x
        aligned_y = y - (text_height / 2)
    else:
        # Adjust x and y to draw the text centered
        aligned_x = x - (width / 2)
        aligned_y = y - (text_height / 2)

    # Set the color and draw the string
    c.setFillColor(color if color else colors.black)  # Set text color
    c.drawString(aligned_x, aligned_y, str(text))  # Convert text to string if needed !!!!!
    c.setFillColor(colors.black)  # Reset color to black for future text


def create_pdf_with_figures(output_pdf_path, test_details, true_columns, channel_transducers, figures):
    c = canvas.Canvas(str(output_pdf_path), pagesize=landscape(A4))
    c.setStrokeColor(colors.black)

    # Define layout rectangles (x, y, width, height)
    layout_rectangles = [
        (15, 515, 600, 65),  # Info Top Left
        (15, 110, 600, 375),  # Graph
        (15, 15, 600, 95),  # Graph Index
        (630, 240, 197, 35), # Test Pressures
        (630, 15, 197, 200),  # 3rd Party Stamp
        (630, 300, 197, 185)  # Info Right
    ]

    for rect in layout_rectangles:
        draw_rectangle(c, *rect)

    light_blue = Color(0.325, 0.529, 0.761)
    black = Color(0, 0, 0)

    # Add title
    draw_centered_text(c,f"{test_details.at['Test Description', 1]} {test_details.at['Test Title', 1]}", 315, 500, font="Helvetica-Bold", size=16)    
    draw_centered_text(c, "Data Recording Equipment Used", 728.5, 475, "Helvetica-Bold", size=12)
    draw_centered_text(c, "3rd Party Stamp and Date", 728.5, 25, "Helvetica-Bold", size=12)

    empty = pd.DataFrame([[''] * 2 for _ in range(14)])
    transducers_present = channel_transducers.loc[true_columns]
    transducers_present = transducers_present.reset_index(drop=True)
    transducers_present.columns = range(transducers_present.shape[1])
    transducers_present = pd.concat([transducers_present, empty], ignore_index=True)
    print(transducers_present)

    # Add placeholders with dynamic content
    text_positions = [
        (20, 571.875, "Test Procedure Reference", black),
        (140, 571.875, test_details.at['Test Procedure Reference', 1], light_blue),
        (20, 555.625, "Unique No.", black),
        (140, 555.625, test_details.at['Unique Number', 1], light_blue),
        (20, 539.375, "R&D Reference", black),
        (140, 539.375, test_details.at['R&D Reference', 1], light_blue),
        (20, 523.125, "Valve Description", black),
        (140, 523.125, test_details.at['Valve Description', 1], light_blue),
        (402.5, 571.875, "Job No.", black),
        (487.5, 571.875, test_details.at['Job Number', 1], light_blue),
        (402.5, 555.625, "Test Description", black),
        (487.5, 555.625, test_details.at['Test Description', 1], light_blue),
        (402.5, 539.375, "Test Date", black),
        (487.5, 539.375, test_details.at['Test Date', 1], light_blue),
        (402.5, 523.125, "Valve Drawing No.", black),
        (487.5, 523.125, test_details.at['Valve Drawing Number', 1], light_blue),
        (635, 266.25, "Test Pressure", black),
        (725, 266.25, test_details.at['Test Pressure', 1], light_blue),
        (635, 248.75, "Actuator Pressure", black),
        (725, 248.75, test_details.at['Actuator Pressure', 1], light_blue),
        (635, 457.5, "Data Logger", black),
        (725, 457.5, test_details.at['Data Logger', 1], light_blue),
        (635, 442.5, "Serial No.", black),
        (725, 442.5, test_details.at['Serial Number', 1], light_blue),
        (635, 427.5, "Transducers", black),
        (635, 412.5, transducers_present.at[0, 0], light_blue),
        (674.375, 412.5, transducers_present.at[1, 0], light_blue),
        (713.75, 412.5, transducers_present.at[2, 0], light_blue),
        (753.125, 412.5, transducers_present.at[3, 0], light_blue),
        (792.5, 412.5, transducers_present.at[4, 0], light_blue),
        (635, 397.5, transducers_present.at[5, 0], light_blue),
        (674.375, 397.5, transducers_present.at[6, 0], light_blue),
        (713.75, 397.5, transducers_present.at[7, 0], light_blue),
        (753.125, 397.5, transducers_present.at[8, 0], light_blue),
        (792.5, 397.5, transducers_present.at[9, 0], light_blue),
        (635, 382.5, transducers_present.at[10, 0], light_blue),
        (674.375, 382.5, transducers_present.at[11, 0], light_blue),
        (713.75, 382.5, transducers_present.at[12, 0], light_blue),
        (753.125, 382.5, transducers_present.at[13, 0], light_blue),
        (792.5, 382.5, transducers_present.at[14, 0], light_blue),
        (635, 367.5, "Gauges", black),
        (635, 352.5, transducers_present.at[0, 1], light_blue),
        (685, 352.5, transducers_present.at[1, 1], light_blue),
        (735, 352.5, transducers_present.at[2, 1], light_blue),
        (785, 352.5, transducers_present.at[3, 1], light_blue),
        (635, 337.5, transducers_present.at[4, 1], light_blue),
        (685, 337.5, transducers_present.at[5, 1], light_blue),
        (735, 337.5, transducers_present.at[6, 1], light_blue),
        (785, 337.5, transducers_present.at[7, 1], light_blue),
        (635, 322.5, transducers_present.at[8, 1], light_blue),
        (685, 322.5, transducers_present.at[9, 1], light_blue),
        (735, 322.5, transducers_present.at[10, 1], light_blue),
        (785, 322.5, transducers_present.at[11, 1], light_blue),
        (635, 307.5, "Torque Transducer", black),
        (725, 307.5, channel_transducers.at['Torque', 1], light_blue)
    ]

    for x, y, text, color in text_positions:
        draw_centered_text(c, text, x, y, color=color, size=10, left_aligned=True)

    img = ImageReader(figures)  # Create an ImageReader object
    c.drawImage(img, 16, 111, 598, 373, preserveAspectRatio=False, mask='auto')  # Use ImageReader object (Graph)
    
    c.drawImage('V:/Userdoc/R & D/Logos/Address logo with address.jpg', 629, 514, 180, 67, preserveAspectRatio=True, mask='auto')

    c.save()

# ------------------------
# Main Execution
# ------------------------

def main():
    """Main function to load and process CSV data."""
    try:
        # Define file paths
        data_file, test_details_file, output_pdf_path = define_file_paths()

        # Load test details
        test_details, channel_transducers, channels_recorded, key_points = load_test_details(test_details_file)

        # Process primary data
        data_recorded, true_columns = process_primary_data(data_file, channels_recorded)

        # Plot the data
        figure = plot_data_with_dual_axes(data_recorded, key_points)

        # Plot to Figure
        figure = convert_fig_to_image_stream(figure)

        create_pdf_with_figures(output_pdf_path, test_details, true_columns, channel_transducers, figure)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()