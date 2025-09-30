"""Tests for calibration specific helpers and report generation."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from additional_info_functions import (  # noqa: E402  (added to path above)
    calculate_calibration_regression,
)
from program_handlers import CalibrationReportGenerator  # noqa: E402


def test_calculate_calibration_regression_returns_expected_coefficients():
    counts = pd.Series([0, 1, 2, 3, 4], dtype=float)
    expected = 2 * counts**3 + 3 * counts**2 + 4 * counts + 5

    coefficients = calculate_calibration_regression(counts, expected)

    assert list(coefficients.index) == ["S3", "S2", "S1", "S0"]
    np.testing.assert_allclose(coefficients.values, [2, 3, 4, 5], atol=1e-8)


def test_calibration_report_generator_draws_regression_table(monkeypatch, tmp_path):
    test_metadata = pd.DataFrame(
        {
            1: {
                'Test Procedure Reference': 'Ref',
                'Unique Number': 'Unique',
                'R&D Reference': 'R&D',
                'Valve Description': 'Valve',
                'Job Number': 'Job',
                'Valve Drawing Number': 'Drawing',
                'Test Section Number': 'Section',
                'Test Name': 'Calibration',
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

    transducer_details = pd.DataFrame({1: {'Torque': 'Device'}})
    channels_to_record = pd.DataFrame({0: ['Upstream'], 1: [True]}).set_index(0)
    cleaned_data = pd.DataFrame(
        {
            'Datetime': pd.date_range('2025-01-01', periods=5, freq='s'),
            'Upstream': [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    additional_info = pd.DataFrame(
        [
            ["7812500.0", "-10000", "-5000", "0", "5000", "10000"],
            ["Upstream", "", "", "", "", ""],
        ]
    )

    calibration_indices = pd.DataFrame(
        {
            0: [np.nan, np.nan],
            1: [0, 0],
            2: [1, 1],
            3: [2, 2],
            4: [3, 3],
            5: [4, 4],
        },
        index=[0, 1],
    )

    captured = {}

    class DummyAxis:
        def scatter(self, *args, **kwargs):
            return None

    def fake_plot_channel_data(**_):
        return MagicMock(), {'left': DummyAxis()}, {'Pressure': 'left'}

    def fake_draw_test_details(*_, **__):
        return MagicMock()

    def fake_draw_table(pdf_canvas, dataframe, **__):
        captured['table'] = dataframe

    def fake_draw_regression_table(pdf_canvas, coefficients, **__):
        captured['coefficients'] = coefficients

    def fake_insert_plot_and_logo(*_, **__):
        return None

    def fake_locate_calibration_points(*_, **__):
        return calibration_indices, None

    monkeypatch.setattr('program_handlers.plot_channel_data', fake_plot_channel_data)
    monkeypatch.setattr('program_handlers.draw_test_details', fake_draw_test_details)
    monkeypatch.setattr('program_handlers.draw_table', fake_draw_table)
    monkeypatch.setattr('program_handlers.draw_regression_table', fake_draw_regression_table)
    monkeypatch.setattr('program_handlers.insert_plot_and_logo', fake_insert_plot_and_logo)
    monkeypatch.setattr('program_handlers.locate_calibration_points', fake_locate_calibration_points)

    generator = CalibrationReportGenerator(
        program_name='Calibration',
        pdf_output_path=tmp_path,
        test_metadata=test_metadata.copy(),
        transducer_details=transducer_details,
        active_channels=['Upstream'],
        cleaned_data=cleaned_data,
        raw_data=cleaned_data,
        additional_info=additional_info,
        part_windows=pd.DataFrame(),
        channels_to_record=channels_to_record,
        channel_map={'Upstream': 'Upstream'},
    )

    output_path = generator.generate()
    assert output_path.suffix == '.pdf'

    assert 'coefficients' in captured
    coefficients = captured['coefficients']
    assert not coefficients.dropna().empty
    assert coefficients.index.tolist() == ["S3", "S2", "S1", "S0"]
