"""Regression tests for Mass Spectrometer channel handling."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.dls_chart_generation.reports.calibration_report import CalibrationReportGenerator
from src.dls_chart_generation.reports.generic_report import GenericReportGenerator


@pytest.fixture()
def base_metadata():
    return pd.DataFrame(
        {
            1: {
                'Test Procedure Reference': 'Ref',
                'Unique Number': 'Unique',
                'R&D Reference': 'R&D',
                'Valve Description': 'Valve',
                'Job Number': 'Job',
                'Valve Drawing Number': 'Drawing',
                'Test Section Number': 'Section',
                'Test Name': 'Name',
                'Test Pressure': '0',
                'Breakout Torque': '0',
                'Running Torque': '0',
                'Data Logger': 'Logger',
                'Serial Number': 'SN',
                'Operative': 'Operator',
                'Date Time': '2025-01-01_00-00-00',
            }
        }
    )


@pytest.fixture()
def common_data():
    cleaned = pd.DataFrame(
        {
            'Datetime': pd.date_range('2025-01-01', periods=3, freq='s'),
            'Upstream Pressure': [100.0, 150.0, 200.0],
            'Mass Spec Custom': [1.0, 2.0, 3.0],
        }
    )
    channels_to_record = pd.DataFrame({0: ['Upstream'], 1: [True]}).set_index(0)
    channel_map = {
        'Upstream': 'Upstream Pressure',
        'Mass Spectrometer': 'Mass Spec Custom',
    }
    return cleaned, channels_to_record, channel_map


def _stub_base_report_dependencies(monkeypatch):
    """Replace heavy drawing functions from base_report with stubs."""
    base_report_module = 'src.dls_chart_generation.reports.base_report'

    def fake_draw_test_details(*args, **kwargs):
        return MagicMock()

    def fake_insert_plot_and_logo(*args, **kwargs):
        return None

    monkeypatch.setattr(f'{base_report_module}.draw_test_details', fake_draw_test_details)
    monkeypatch.setattr(f'{base_report_module}.insert_plot_and_logo', fake_insert_plot_and_logo)


def test_mass_spec_channel_filtered_for_standard_reports(monkeypatch, tmp_path, base_metadata, common_data):
    cleaned_data, channels_to_record, channel_map = common_data
    captured = []
    generic_report_module = 'src.dls_chart_generation.reports.generic_report'

    def fake_plot_channel_data(*, active_channels, **kwargs):
        captured.append(list(active_channels))
        axes = {'left': MagicMock(), 'right_1': MagicMock()}
        axis_map = {'Pressure': 'left', 'Mass Spectrometer': 'right_1'}
        return MagicMock(), axes, axis_map

    _stub_base_report_dependencies(monkeypatch)
    monkeypatch.setattr(f'{generic_report_module}.plot_channel_data', fake_plot_channel_data)

    generator = GenericReportGenerator(
        program_name='Generic',
        pdf_output_path=tmp_path,
        test_metadata=base_metadata.copy(),
        transducer_details=pd.DataFrame(),
        active_channels=['Upstream Pressure', 'Mass Spec Custom'],
        cleaned_data=cleaned_data,
        raw_data=cleaned_data,
        additional_info=pd.DataFrame(),
        part_windows=pd.DataFrame(),
        channels_to_record=channels_to_record,
        channel_map=channel_map,
    )

    generator.generate()

    assert captured, 'plot_channel_data should be invoked'
    assert captured[0] == ['Upstream Pressure']


def test_mass_spec_channel_retained_for_calibration(monkeypatch, tmp_path, base_metadata, common_data):
    cleaned_data, channels_to_record, channel_map = common_data
    captured = []
    calibration_report_module = 'src.dls_chart_generation.reports.calibration_report'

    def fake_plot_channel_data(*, active_channels, **kwargs):
        captured.append(list(active_channels))
        axes = {'left': MagicMock(), 'right_1': MagicMock()}
        axis_map = {'Pressure': 'left', 'Mass Spectrometer': 'right_1'}
        return MagicMock(), axes, axis_map

    _stub_base_report_dependencies(monkeypatch)
    monkeypatch.setattr(f'{calibration_report_module}.plot_channel_data', fake_plot_channel_data)
    monkeypatch.setattr(f'{calibration_report_module}.draw_table', lambda *a, **k: None)


    def fake_locate_calibration_points(*args, **kwargs):
        return pd.DataFrame({0: [0]} , index=['Phase']), None

    def fake_calculate_success(*args, **kwargs):
        average = pd.DataFrame({'Value': [1.0]})
        counts = pd.Series([1.0])
        expected = pd.Series([1.0])
        errors = pd.Series([0.0])
        return average, counts, expected, errors

    def fake_evaluate_thresholds(*args, **kwargs):
        return pd.Series(dtype=bool)

    monkeypatch.setattr(f'{calibration_report_module}.locate_calibration_points', fake_locate_calibration_points)
    monkeypatch.setattr(f'{calibration_report_module}.calculate_succesful_calibration', fake_calculate_success)
    monkeypatch.setattr(f'{calibration_report_module}.evaluate_calibration_thresholds', fake_evaluate_thresholds)

    generator = CalibrationReportGenerator(
        program_name='Calibration',
        pdf_output_path=tmp_path,
        test_metadata=base_metadata.copy(),
        transducer_details=pd.DataFrame(),
        active_channels=['Upstream Pressure', 'Mass Spec Custom'],
        cleaned_data=cleaned_data,
        raw_data=cleaned_data,
        additional_info=pd.DataFrame([
            ['7812500.0', '-10000', '-5000', '0', '5000', '10000'],
            ['Mass Spec Custom', '', '', '', '', ''],
        ]),
        part_windows=pd.DataFrame(),
        channels_to_record=channels_to_record,
        channel_map=channel_map,
    )

    generator.generate()

    assert captured, 'plot_channel_data should be invoked for calibration'
    assert captured[0] == ['Upstream Pressure', 'Mass Spec Custom']