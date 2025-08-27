"""Helpers for loading CSV test data."""

from pathlib import Path
import pandas as pd
from channel_mapping import create_channel_name_mapping


def get_file_paths(primary_data_path: str, test_details_path: str, output_pdf_path: str):
    """Return standardised file paths used throughout the program."""
    return (
        primary_data_path,
        test_details_path,
        Path(output_pdf_path),
    )


def load_csv_file(file_path: str, **kwargs) -> pd.DataFrame:
    """Wrapper around :func:`pandas.read_csv` with friendly errors."""
    try:
        return pd.read_csv(file_path, **kwargs, dayfirst=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {file_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File is empty: {file_path}") from exc
    except Exception as exc:  # pragma: no cover - unexpected read error
        raise Exception(f"Error reading file {file_path}: {exc}") from exc


def load_test_information(test_details_path: str):
    """Load all test information from the test details CSV file."""

    # Load top sections (test metadata and transducer data)
    test_metadata = (
        load_csv_file(
            test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1],
            nrows=19,
        ).fillna("")
    )

    test_section_number = str(test_metadata.at["Test Section Number", 1]).strip()
    test_name = str(test_metadata.at["Test Name", 1]).strip()
    prefix = f"{test_section_number} "
    if test_section_number and test_name.startswith(prefix):
        test_metadata.at["Test Name", 1] = test_name[len(prefix):]

    transducer_details = (
        load_csv_file(
            test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1, 2],
            skiprows=19,
            nrows=26,
        ).fillna("")
    )

    channels_to_record = (
        load_csv_file(
            test_details_path,
            header=None,
            usecols=[0, 3],
            skiprows=19,
            nrows=26,
        ).fillna("")
    )

    channels_to_record.columns = [0, 1]
    channels_to_record.set_index(0, inplace=True)
    channels_to_record.fillna('', inplace=True)

    # Create the channel name mapping
    custom_channel_names = channels_to_record.index.tolist()
    default_to_custom_map = create_channel_name_mapping(custom_channel_names)

    with open(test_details_path, 'r', encoding='utf-8') as f:
        row_count = sum(1 for _ in f)

    if row_count > 45:
        additional_info = (
            load_csv_file(
                test_details_path,
                header=None,
                skiprows=45,
            ).reset_index(drop=True)
        )
    else:
        additional_info = pd.DataFrame()

    program_name = test_metadata.at["Program Name", 1]

    return (
        test_metadata,
        transducer_details,
        channels_to_record,
        additional_info,
        program_name,
        default_to_custom_map,
    )

def prepare_primary_data(primary_data_path: str, channels_to_record: pd.DataFrame):
    """Load the primary data CSV and return a cleaned subset."""

    # Load raw data (assumes the CSV has Datetime followed by channels)
    raw_data = load_csv_file(
        primary_data_path,
        header=0,
        parse_dates=["Datetime"],
    ).iloc[:-1]

    # Identify which channels are actually recorded
    active_channels = channels_to_record[channels_to_record[1] == True].index.tolist()
    required_columns = ["Datetime"] + active_channels

    # Extract only the required columns
    data_subset = raw_data[required_columns].copy()

    # Convert the single 'Datetime' column to a proper datetime type
    # (assuming format dd/mm/yyyy hh:mm:ss.000)
    data_subset["Datetime"] = pd.to_datetime(
        data_subset["Datetime"],
        format="%d/%m/%Y %H:%M:%S.%f",
        errors="coerce",  # in case of any parse issues
    )

    # Ensure 'Datetime' is the first column
    columns_ordered = ["Datetime"] + [col for col in data_subset.columns if col != "Datetime"]
    data_subset = data_subset[columns_ordered]

    return data_subset, active_channels, raw_data