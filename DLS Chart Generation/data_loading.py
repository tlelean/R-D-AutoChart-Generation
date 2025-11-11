"""Helpers for loading CSV test data."""

from pathlib import Path
import pandas as pd
from channel_mapping import create_channel_name_mapping
import json


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


    root = json.load(open(test_details_path))
    test_metadata = (root["metadata"])
    channel_info = pd.DataFrame(root["channel_info"])
    mass_spec_timings = pd.DataFrame(root["mass_spec_timings"])
    holds = pd.DataFrame(root["holds"])
    cycles = pd.DataFrame(root["cycles"])
    calibration = root["calibration"]

    test_section_number = str(test_metadata["Test Section Number"]).strip()
    test_name = str(test_metadata["Test Name"]).strip()
    prefix = f"{test_section_number} "
    if test_section_number and test_name.startswith(prefix):
        test_metadata["Test Name"] = test_name[len(prefix):]

    transducers_codes = channel_info[["channel", "transducer"]].fillna("")
    gauge_codes = channel_info[["channel", "gauge"]].fillna("")
    channel_visibility = channel_info.set_index('channel')[['visible']]

    # Create the channel name mapping
    custom_channel_names = channel_info["channel"].tolist()
    default_to_custom_map = create_channel_name_mapping(custom_channel_names)

    return (
        test_metadata,
        transducers_codes,
        gauge_codes,
        channel_visibility,
        mass_spec_timings,
        holds,
        cycles,
        calibration,
        default_to_custom_map
    )

def prepare_primary_data(primary_data_path: str, channels_to_record: pd.DataFrame):
    """Load the primary data CSV and return a cleaned subset."""

    # Load raw data (assumes the CSV has Datetime followed by channels)
    raw_data = load_csv_file(
        primary_data_path,
        header=0,
    ).iloc[:-1]

    # Identify which channels are actually recorded
    active_channels = channels_to_record[channels_to_record['visible'] == True].index.tolist()
    required_columns = ["Datetime"] + active_channels

    # Extract only the required columns
    data_subset = raw_data[required_columns].copy()

    # Drop duplicate timestamps so downstream consumers always see unique
    # Datetime values (while preserving the first occurrence).
    dedupe_mask = ~data_subset["Datetime"].duplicated(keep="first")
    if not dedupe_mask.all():
        data_subset = data_subset.loc[dedupe_mask].copy()
        raw_data = raw_data.loc[dedupe_mask].copy()

    # Convert the single 'Datetime' column to a proper datetime type
    # (assuming format dd/mm/yyyy hh:mm:ss.000)
    data_subset["Datetime"] = pd.to_datetime(
        data_subset["Datetime"],
        format="%d/%m/%Y %H:%M:%S.%f",
        errors="coerce",
        dayfirst=True,
    )

    # Ensure 'Datetime' is the first column
    columns_ordered = ["Datetime"] + [col for col in data_subset.columns if col != "Datetime"]
    data_subset = data_subset[columns_ordered]

    return data_subset, active_channels, raw_data