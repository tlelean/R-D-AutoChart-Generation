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
    Returns metadata, transducer info, channels to record, and program name.
    (Does NOT require cleaned_data.)

    Parameters:
        test_details_path (str or Path): CSV containing test details.

    Returns:
        tuple: 
            - pd.DataFrame: DataFrame of test details (metadata).
            - pd.DataFrame: Transducer info for each channel.
            - pd.DataFrame: Indicates channels to be recorded (True/False).
            - str: Program name.
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
            nrows=26)
            .fillna('')
    )

    channels_to_record = (
        load_csv_file(
            test_details_path,
            header=None,
            usecols=[0, 3],
            skiprows=19,
            nrows=26)
            .fillna('')
    )

    channels_to_record.columns = [0, 1]
    channels_to_record.set_index(0, inplace=True)
    channels_to_record.fillna('', inplace=True)

    program_name = test_metadata.at['Program Name', 1]

    return (
        test_metadata,
        transducer_details,
        channels_to_record,
        program_name
    )

def load_additional_info(test_details_path, program_name, raw_data):
    """
    Loads additional info (holds, breakouts, etc.) depending on program type.
    Requires cleaned_data for breakouts.

    Returns:
        pd.DataFrame or None
    """
    def handle_holds():
        df = load_csv_file(test_details_path, header=0, skiprows=45)
        return df.dropna(how='all').dropna(axis=1, how='all').fillna('').reset_index(drop=True)

    def handle_breakouts():
        additional_info = load_csv_file(
            test_details_path,
            header=None,
            usecols=[0, 1, 2],
            skiprows=45,
        ).reset_index(drop=True)

        bto_indicies: list[int] = []
        btc_indicies: list[int] = []

        # Early‑exit if data seem to be missing --------------------------------
        if additional_info.iloc[1, 0] == "NaN":
            return additional_info, bto_indicies, btc_indicies

        torque_data = raw_data["Torque"]
        cycle_count_data = raw_data["Cycle Count"]

        # ── Process one cycle at a time ───────────────────────────────────────
        for i, cycle_num in enumerate(sorted(cycle_count_data.unique())):
            mask = cycle_count_data == cycle_num
            torque_cycle = torque_data[mask]

            n_points = len(torque_cycle)
            if n_points == 0:
                # Skip empty cycles (shouldn't happen, but better to be safe)
                continue

            # Compute slice boundaries — each third is as equal as integer division allows
            third_len = max(1, n_points // 3)
            first_slice  = slice(0, third_len)
            middle_slice = slice(third_len, 2 * third_len)

            torque_first_third  = torque_cycle.iloc[first_slice]
            torque_middle_third = torque_cycle.iloc[middle_slice]

            # ── Determine BTO and BTC values ──────────────────────────────────
            bto = torque_first_third.min().round(1)
            btc = torque_middle_third.max().round(1)

            # Record the row indices at which these extremes occur -------------
            bto_indicies.append(int(torque_first_third.idxmin()))
            btc_indicies.append(int(torque_middle_third.idxmax()))

            # Write results back into *additional_info* (row offset by +1) -----
            additional_info.iloc[i + 1, 1] = bto
            additional_info.iloc[i + 1, 2] = btc

        return additional_info, bto_indicies, btc_indicies

    def handle_default():
        return None

    program_handlers = {
        "Holds-Seat": handle_holds,
        "Holds-Body": handle_holds,
        "Atmospheric Breakouts": handle_breakouts,
        "Atmospheric Cyclic": handle_breakouts,
        "Dynamic Cycles PR2": handle_breakouts,
        "Dynamic Cycles Petrobras": handle_breakouts,
        "Pulse Cycles": handle_default,
        "Signatures": handle_default,
        "Open-Close": handle_breakouts,
        "Number Of Turns": handle_default,
    }
    handler = program_handlers.get(program_name, handle_default)
    return handler()

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

    return data_subset, active_channels, raw_data