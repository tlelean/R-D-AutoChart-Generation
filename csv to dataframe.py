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
    base_dir = Path(r"V:/Userdoc/R & D/DAQ_Station/tlelean/Job Number/Valve Drawing Number/CSV/0.2")
    data_file = base_dir / "Test Description_Data_16-12-2024_16-14-26.csv"
    test_details_file = base_dir / "Test Description_Test_Details_16-12-2024_16-14-26.csv"
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
    test_details = read_csv_safely(filepath, header=None, index_col=0, usecols=[0, 1], nrows=13)
    channel_transducers = read_csv_safely(filepath, header=None, index_col=0, usecols=[0, 1], skiprows=13, nrows=21)
    channels_recorded = read_csv_safely(filepath, header=None, usecols=[0, 2], skiprows=13, nrows=21)
    channels_recorded.columns = [0, 1]
    channels_recorded.set_index(0, inplace=True)
    key_points = read_csv_safely(filepath, usecols=["Main Channel", "Start of Stabalisation", "Start of Hold", "End of Hold"], skiprows=34)
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
    return data_recorded

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
    fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
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

def add_text(c, text, x, y, font="Helvetica", size=12, color=None):
    """Adds text at the specified location, with specified colour. Defaults to black if no colour is specified."""
    c.setFont(font, size)
    c.setFillColor(color if color else colors.black)  # Set color to black if none is specified
    c.drawString(x, y, text)
    c.setFillColor(colors.black)  # Reset color to black for future text
    
def draw_centered_text(c, text, x, y, font="Helvetica", size=12):
    """Draws centered text with the specified font and size."""
    c.setFont(font, size)  # Set the font and size
    width = c.stringWidth(text, font, size)  # Get the width of the text
    c.drawCentredString(x, y, text)  # Draw centered text

def draw_grid_with_labels(c, x, y, width, height, rows, cols, headers, row_labels):
    """Draws a grid with headers and row labels at the specified location."""
    cell_width = width / cols
    cell_height = height / rows

    # Draw horizontal lines
    for i in range(rows + 1):
        c.line(x, y + i * cell_height, x + width, y + i * cell_height)
    
    # Draw vertical lines
    for j in range(cols + 1):
        c.line(x + j * cell_width, y, x + j * cell_width, y + height)
    
     # Draw headers (centered)
    for j, header in enumerate(headers):
        text_width = c.stringWidth(header, "Helvetica", 12)
        text_height = 12  # Approximate height of the text
        c.drawString(x + j * cell_width + (cell_width - text_width) / 2, y + (rows - 1) * cell_height + (cell_height - text_height) / 2, header)
    
    # Draw row labels (centered inside the first column cells)
    for i, row_label in enumerate(row_labels):
        text_width = c.stringWidth(row_label, "Helvetica", 12)
        c.drawString(x + (cell_width - text_width) / 2, y + (rows - i - 1) * cell_height + (cell_height - 12) / 2, row_label)

def create_pdf_with_figures(output_pdf_path, test_details, figures):
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
    draw_centered_text(c, test_details.at['Test Title', 1], 315, 492.5, font="Helvetica-Bold", size=20)    
    add_text(c, "Data Recording Equipment Used", 635, 470, "Helvetica-Bold", 12)
    add_text(c, "3rd Party Stamp and Date", 645, 20, "Helvetica-Bold", 14)

    # Add placeholders with dynamic content
    text_positions = [
        (20, 565, f"Test Procedure Reference", black),
        (165, 565, test_details.at['Test Procedure Reference', 1], light_blue),
        (20, 550, f"Unique No.", black),
        (165, 550, test_details.at['Unique Number', 1], light_blue),
        (20, 535, f"R&D Reference", black),
        (165, 535, test_details.at['R&D Reference', 1], light_blue),
        (20, 520, f"Valve Description", black),
        (165, 520, test_details.at['Valve Description', 1], light_blue),
        (325, 565, f"Job No.", black),
        (455, 565, test_details.at['Job Number', 1], light_blue),
        (325, 550, f"Test Description", black),
        (455, 550, test_details.at['Test Description', 1], light_blue),
        (325, 535, f"Test Date", black),
        (455, 535, test_details.at['Test Date', 1], light_blue),
        (325, 520, f"Valve Drawing No.", black),
        (455, 520, test_details.at['Valve Drawing Number', 1], light_blue),
        (635, 260, f"Test Pressure", black),
        (745, 260, test_details.at['Test Pressure', 1], light_blue),
        (635, 245, f"Actuator Pressure", black),
        (745, 245, test_details.at['Actuator Pressure', 1], light_blue),
        (635, 455, f"Data Logger", black),
        (745, 455, test_details.at['Data Logger', 1], light_blue),
        (635, 440, f"Serial No.", black),
        (745, 440, test_details.at['Serial Number', 1], light_blue),
        (635, 425, f"Transducers", black),
        # (745, 425, f"{entries['transducer_1'].get()}", light_blue),
        # (745, 410, f"{entries['transducer_2'].get()}", light_blue),
        # (745, 395, f"{entries['transducer_3'].get()}", light_blue),
        # (745, 380, f"{entries['transducer_4'].get()}", light_blue),
        (635, 365, f"Gauges", black),
        # (745, 365, f"{entries['gauge_1'].get()}", light_blue),
        # (745, 350, f"{entries['gauge_2'].get()}", light_blue),
        # (745, 335, f"{entries['gauge_3'].get()}", light_blue),
        # (745, 320, f"{entries['gauge_4'].get()}", light_blue),
        (635, 305, f"Torque Transducer", black)
        # (745, 305, f"{entries['torque_transducer'].get()}", light_blue)
    ]

    for x, y, text, color in text_positions:
        add_text(c, text, x, y, color=color)

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
        data_recorded = process_primary_data(data_file, channels_recorded)

        # Plot the data
        figure = plot_data_with_dual_axes(data_recorded, key_points)

        # Plot to Figure
        figure = convert_fig_to_image_stream(figure)

        create_pdf_with_figures(output_pdf_path, test_details, figure)

        print(test_details)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()