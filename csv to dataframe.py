import pandas as pd
from pathlib import Path

# ------------------------
# Configuration
# ------------------------

def define_file_paths():
    """Define file paths for the CSV files."""
    base_dir = Path(r"V:/Userdoc/R & D/DAQ_Station/tlelean/Job Number/Valve Drawing Number/CSV/0.2")
    data_file = base_dir / "Test Description_Data_16-12-2024_16-14-26.csv"
    test_details_file = base_dir / "Test Description_Test_Details_16-12-2024_16-14-26.csv"
    return data_file, test_details_file

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
    key_points = read_csv_safely(filepath, usecols=[0, 1, 2, 3], skiprows=34)

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
        format='%d-%m-%Y %H-%M-%S.%f'
    )

    # Drop unnecessary columns
    data_recorded = data_recorded.drop(columns=['Date', 'Time', 'Milliseconds'])

    # Reorder columns
    columns = ['Datetime'] + [col for col in data_recorded.columns if col != 'Datetime']
    data_recorded = data_recorded[columns]
    return data_recorded

def display_results(data_recorded):
    """Display the processed data."""
    print("\nProcessed Data Recorded:")
    print(data_recorded)

# ------------------------
# Main Execution
# ------------------------

def main():
    """Main function to load and process CSV data."""
    try:
        # Define file paths
        data_file, test_details_file = define_file_paths()

        # Load test details
        test_details, channel_transducers, channels_recorded, key_points = load_test_details(test_details_file)

        # Process primary data
        data_recorded = process_primary_data(data_file, channels_recorded)

        # Display results
        display_results(data_recorded)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
