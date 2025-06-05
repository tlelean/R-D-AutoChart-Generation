import pandas as pd
from pathlib import Path

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


def load_test_information(test_details_path):
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
            nrows=19)
            .fillna('')
    )

    transducer_details = (
        load_csv_file(
            test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1, 2],
            skiprows=19,
            nrows=21)
            .fillna('')
    )

    channels_to_record = (
        load_csv_file(
            test_details_path,
            header=None,
            usecols=[0, 3],
            skiprows=19,
            nrows=21)
            .fillna('')
    )

    channels_to_record.columns = [0, 1]
    channels_to_record.set_index(0, inplace=True)
    channels_to_record.fillna('', inplace=True)

    program_name = test_metadata.at['Program Name', 1]
   
    # Handler functions for each program type
    def handle_holds():
        return load_csv_file(test_details_path, header=0, skiprows=40) \
            .dropna(how='all') \
            .dropna(axis=1, how='all') \
            .fillna('') \
            .reset_index(drop=True)
    
    def handle_breakouts():
        return load_csv_file(test_details_path, header=None, skiprows=40) \
            .dropna(how='all') \
            .dropna(axis=1, how='all') \
            .fillna('') \
            .reset_index(drop=True)

    # Default handler for unimplemented programs
    def handle_default():
        return None

    # Map program names to handler functions
    program_handlers = {
        "Holds-Body": handle_holds,
        "Holds-Seat": handle_holds,
        "Atmospheric Breakouts": handle_breakouts,
        "Atmospheric Cyclic": handle_breakouts,
        "Dynamic Cycles PR2": handle_breakouts,
        "Dynamic Cycles Petrobras": handle_breakouts,
        "Pulse Cycles": handle_breakouts,
        "Signatures": handle_breakouts,
        "Open-Close": handle_breakouts,
        "Number Of Turns": handle_breakouts,
    }

    # Call the appropriate handler
    additional_info = program_handlers.get(program_name, handle_default)()

    return (
        test_metadata,
        transducer_details,
        channels_to_record,
        additional_info
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