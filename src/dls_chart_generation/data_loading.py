# -*- coding: utf-8 -*-
"""
Encapsulates data loading and validation for the DLS Chart Generation tool.

This module provides the `DataLoader` class, which is responsible for
reading, validating, and processing data from primary and test details CSV files.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict, Any

from . import config
from .channel_mapping import create_channel_name_mapping

class DataLoader:
    """
    Handles loading, validation, and preparation of test data from CSV files.

    This class orchestrates the entire data ingestion process, from reading
    raw CSV files to producing cleaned and validated DataFrames ready for
    report generation.

    Attributes:
        primary_data_path (str): The file path for the primary data CSV.
        test_details_path (str): The file path for the test details CSV.
    """

    def __init__(self, primary_data_path: str, test_details_path: str):
        """
        Initializes the DataLoader with paths to the data files.

        Args:
            primary_data_path: Path to the primary data CSV file.
            test_details_path: Path to the test details CSV file.
        """
        self.primary_data_path = primary_data_path
        self.test_details_path = test_details_path

    def load_and_process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Dict[str, str], pd.DataFrame, List[str], pd.DataFrame]:
        """
        Orchestrates the loading, validation, and processing of all data.

        This is the main public method of the class. It calls helper methods
        to load test details, then loads and cleans the primary data based on
        the details found.

        Returns:
            A tuple containing all the processed DataFrames and metadata required
            for report generation:
            - test_metadata
            - transducer_details
            - channels_to_record
            - part_windows
            - additional_info
            - program_name
            - default_to_custom_map
            - cleaned_data
            - active_channels
            - raw_data
        """
        (
            test_metadata,
            transducer_details,
            channels_to_record,
            part_windows,
            additional_info,
            program_name,
            default_to_custom_map,
        ) = self._load_test_information()

        cleaned_data, active_channels, raw_data = self._prepare_primary_data(
            channels_to_record
        )

        return (
            test_metadata,
            transducer_details,
            channels_to_record,
            part_windows,
            additional_info,
            program_name,
            default_to_custom_map,
            cleaned_data,
            active_channels,
            raw_data,
        )

    def _load_csv(self, **kwargs) -> pd.DataFrame:
        """
        A wrapper for `pandas.read_csv` with standardized error handling.

        Args:
            **kwargs: Keyword arguments to pass directly to `pandas.read_csv`.

        Returns:
            A pandas DataFrame with the loaded data.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            ValueError: If the file is empty.
            IOError: For other pandas reading errors.
        """
        filepath = kwargs.get('filepath_or_buffer')
        try:
            return pd.read_csv(**kwargs, dayfirst=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {filepath}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File is empty: {filepath}") from exc
        except Exception as exc:
            raise IOError(f"Error reading file {filepath}: {exc}") from exc

    def _load_test_information(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Dict[str, str]]:
        """
        Loads and parses all data sections from the test details CSV file.

        This method reads the test details CSV in sections (metadata, transducers, etc.)
        based on the layout defined in the `config` module.

        Returns:
            A tuple containing the various DataFrames and metadata parsed from the file.
        """
        test_metadata = self._load_csv(
            filepath_or_buffer=self.test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1],
            nrows=config.METADATA_ROWS,
        ).fillna("")
        self._validate_metadata(test_metadata)

        transducer_details = self._load_csv(
            filepath_or_buffer=self.test_details_path,
            header=None,
            index_col=0,
            usecols=[0, 1, 2],
            skiprows=config.CHANNELS_TO_RECORD_SKIP_ROWS,
            nrows=config.CHANNELS_TO_RECORD_ROWS,
        ).fillna("")

        channels_to_record = self._load_csv(
            filepath_or_buffer=self.test_details_path,
            header=None,
            usecols=[0, 3],
            skiprows=config.CHANNELS_TO_RECORD_SKIP_ROWS,
            nrows=config.CHANNELS_TO_RECORD_ROWS,
        ).fillna("")
        channels_to_record.columns = [0, 1]
        channels_to_record.set_index(0, inplace=True)
        channels_to_record.fillna('', inplace=True)
        # Convert the 'Record' column to boolean
        channels_to_record[1] = channels_to_record[1].astype(str).str.upper() == 'TRUE'

        part_windows = self._load_csv(
            filepath_or_buffer=self.test_details_path,
            header=None,
            usecols=[0, 1, 2],
            skiprows=config.PART_WINDOW_SKIP_ROWS,
            nrows=config.PART_WINDOW_ROWS,
        ).fillna("")
        part_windows.columns = ["Part", "Start", "Stop"]

        with open(self.test_details_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f)

        additional_info = pd.DataFrame()
        if row_count > config.ADDITIONAL_INFO_SKIP_ROWS:
            additional_info = self._load_csv(
                filepath_or_buffer=self.test_details_path,
                header=None,
                skiprows=config.ADDITIONAL_INFO_SKIP_ROWS,
            ).reset_index(drop=True)

        test_section_number = str(test_metadata.at["Test Section Number", 1]).strip()
        test_name = str(test_metadata.at["Test Name", 1]).strip()
        prefix = f"{test_section_number} "
        if test_section_number and test_name.startswith(prefix):
            test_metadata.at["Test Name", 1] = test_name[len(prefix):]

        program_name = test_metadata.at["Program Name", 1]
        custom_channel_names = channels_to_record.index.tolist()
        default_to_custom_map = create_channel_name_mapping(custom_channel_names)

        return (
            test_metadata,
            transducer_details,
            channels_to_record,
            part_windows,
            additional_info,
            program_name,
            default_to_custom_map,
        )

    def _prepare_primary_data(self, channels_to_record: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Loads, validates, and cleans the primary data CSV.

        Args:
            channels_to_record: DataFrame indicating which channels to load.

        Returns:
            A tuple containing the cleaned data subset, a list of active channel
            names, and the raw, unprocessed data.
        """
        raw_data = self._load_csv(
            filepath_or_buffer=self.primary_data_path,
            header=0,
        )

        active_channels = channels_to_record[channels_to_record[1]].index.tolist()
        required_columns = ["Datetime"] + active_channels
        self._validate_primary_data_columns(raw_data.columns, required_columns)

        data_subset = raw_data[required_columns].copy()

        data_subset["Datetime"] = pd.to_datetime(
            data_subset["Datetime"],
            format=config.DATETIME_FORMAT,
            errors="coerce",
            dayfirst=True,
        )
        if data_subset["Datetime"].isnull().any():
            raise ValueError(f"Failed to parse one or more dates in 'Datetime' column using format {config.DATETIME_FORMAT}")

        columns_ordered = ["Datetime"] + [col for col in data_subset.columns if col != "Datetime"]
        return data_subset[columns_ordered], active_channels, raw_data

    def _validate_metadata(self, metadata: pd.DataFrame):
        """
        Validates that essential keys are present in the metadata DataFrame.

        Args:
            metadata: The DataFrame containing test metadata.

        Raises:
            ValueError: If any required metadata keys are missing.
        """
        required_keys = ["Test Section Number", "Test Name", "Date Time", "Program Name"]
        missing_keys = [key for key in required_keys if key not in metadata.index]
        if missing_keys:
            raise ValueError(f"Test details file is missing required metadata keys: {', '.join(missing_keys)}")

    def _validate_primary_data_columns(self, loaded_columns: pd.Index, required_columns: List[str]):
        """
        Validates that the primary data file contains all required channel columns.

        Args:
            loaded_columns: The columns present in the loaded primary data file.
            required_columns: The columns required based on the test details.

        Raises:
            ValueError: If any required columns are missing from the primary data file.
        """
        missing_cols = set(required_columns) - set(loaded_columns)
        if missing_cols:
            raise ValueError(f"Primary data file is missing required columns: {', '.join(missing_cols)}")