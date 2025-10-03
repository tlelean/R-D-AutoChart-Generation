# -*- coding: utf-8 -*-
"""
Tests for the DataLoader class.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.dls_chart_generation.data_loading import DataLoader

@pytest.fixture
def create_test_files(tmp_path):
    """Creates temporary test CSV files for data loading tests."""
    primary_data_content = (
        "Datetime,Channel1,Channel2\n"
        "01/01/2023 12:00:00.000,10,20\n"
        "01/01/2023 12:00:01.000,11,21\n"
    )
    test_details_content = (
        "Test Section Number,1.1\n"
        "Test Name,Sample Test\n"
        "Date Time,01/01/2023 12:00:00\n"
        "Program Name,Initial Cycle\n"
        "Operator,Jules\n"
        # Add 14 empty lines to simulate the full metadata section
        + "\n" * 14
        # Transducer section starts at row 20 (index 19)
        # This section needs a 4th column for the 'Record' flag
        + "Transducer,Type,Serial,Record\n"
        + "Channel1,Pressure,123,TRUE\n"
        + "Channel2,Temperature,456,TRUE\n"
        # Add 24 more empty lines to make a full 26-row channel section
        + ",,,\n" * 24
        # Padding to get to the part windows section
        + "\n" * 13
        + "Part,Start,Stop\n"
    )

    primary_data_path = tmp_path / "primary_data.csv"
    primary_data_path.write_text(primary_data_content)

    test_details_path = tmp_path / "test_details.csv"
    test_details_path.write_text(test_details_content)

    return primary_data_path, test_details_path

def test_data_loader_loads_data_successfully(create_test_files):
    """
    Tests that the DataLoader can successfully load and process valid data files.
    """
    primary_data_path, test_details_path = create_test_files

    loader = DataLoader(str(primary_data_path), str(test_details_path))
    (
        test_metadata,
        _, # transducer_details
        _, # channels_to_record
        _, # part_windows
        _, # additional_info
        program_name,
        _, # default_to_custom_map
        cleaned_data,
        active_channels,
        _, # raw_data
    ) = loader.load_and_process_data()

    assert not test_metadata.empty
    assert program_name == "Initial Cycle"
    assert "Channel1" in active_channels
    assert "Channel2" in active_channels
    assert "Datetime" in cleaned_data.columns
    assert len(cleaned_data) == 2

def test_data_loader_raises_error_on_missing_file():
    """
    Tests that the DataLoader raises a FileNotFoundError when a file is missing.
    """
    with pytest.raises(FileNotFoundError):
        loader = DataLoader("non_existent_file.csv", "another_non_existent_file.csv")
        loader.load_and_process_data()

def test_data_loader_raises_error_on_missing_metadata(tmp_path):
    """
    Tests that the DataLoader raises a ValueError if the metadata is incomplete.
    """
    primary_data_path = tmp_path / "primary.csv"
    primary_data_path.write_text("Datetime,Channel1\n01/01/2023 12:00:00.000,10")

    test_details_path = tmp_path / "details.csv"
    test_details_path.write_text("Invalid,Content\n") # Missing required keys

    with pytest.raises(ValueError, match="missing required metadata keys"):
        loader = DataLoader(str(primary_data_path), str(test_details_path))
        loader.load_and_process_data()

def test_data_loader_raises_error_on_missing_columns(tmp_path):
    """
    Tests that the DataLoader raises a ValueError if the primary data is missing columns.
    """
    primary_data_path = tmp_path / "primary.csv"
    primary_data_path.write_text("Datetime,WrongColumn\n01/01/2023 12:00:00.000,10")

    test_details_path = tmp_path / "details.csv"
    test_details_content = (
        "Test Section Number,1.1\n"
        "Test Name,Sample Test\n"
        "Date Time,01/01/2023 12:00:00\n"
        "Program Name,Initial Cycle\n"
        "Operator,Jules\n"
        + "\n" * 14
        + "Transducer,Type,Serial,Record\n"
        + "Channel1,Pressure,123,TRUE\n"
        # Add padding to ensure the file is long enough to read all sections
        + ",,,\n" * 40
    )
    test_details_path.write_text(test_details_content)

    with pytest.raises(ValueError, match="missing required columns"):
        loader = DataLoader(str(primary_data_path), str(test_details_path))
        loader.load_and_process_data()