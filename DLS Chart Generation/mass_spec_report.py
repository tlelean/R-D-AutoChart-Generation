from __future__ import annotations

"""Helpers for generating per-part mass spectrometer reports."""

from pathlib import Path
from typing import List

import pandas as pd

from graph_plotter import plot_channel_data
from pdf_helpers import draw_test_details, insert_plot_and_logo


def generate_mass_spec_reports(
    *,
    cleaned_data: pd.DataFrame,
    part_windows: pd.DataFrame,
    mass_spec_channel: str,
    test_metadata: pd.DataFrame,
    transducer_details: pd.DataFrame,
    pdf_output_path: Path,
    channels_to_record: pd.DataFrame,
    channel_map: dict[str, str],
    raw_data: pd.DataFrame,
) -> List[Path]:
    """Generate a mass spectrometer report for each part window.

    Parameters
    ----------
    cleaned_data:
        Full cleaned dataset containing the mass spectrometer channel.
    part_windows:
        DataFrame describing start and stop times for each part. The
        DataFrame is expected to have columns named ``Part``, ``Start``
        and ``Stop`` (case insensitive). Any rows with missing or invalid
        times are ignored.
    mass_spec_channel:
        The name of the mass spectrometer channel in ``cleaned_data``.
    test_metadata, transducer_details:
        Metadata required for :func:`draw_test_details`.
    pdf_output_path:
        Directory in which to save the generated PDFs.
    channels_to_record, channel_map, raw_data:
        Passed through to :func:`plot_channel_data` and
        :func:`draw_test_details` for consistency with the main report
        generation routines.

    Returns
    -------
    list[pathlib.Path]
        A list of paths to the generated PDF files.
    """

    if part_windows is None or part_windows.empty:
        return []

    # Normalise column names for easier lookups
    normalised = {
        c.lower(): c for c in part_windows.columns
    }
    part_col = normalised.get("part")
    start_col = normalised.get("start")
    stop_col = normalised.get("stop")

    if not (part_col and start_col and stop_col):
        return []

    generated_paths: List[Path] = []

    for _, row in part_windows.iterrows():
        part = str(row.get(part_col, "")).strip()
        start = pd.to_datetime(row.get(start_col), errors="coerce", dayfirst=True)
        stop = pd.to_datetime(row.get(stop_col), errors="coerce", dayfirst=True)
        if not part or pd.isna(start) or pd.isna(stop) or start >= stop:
            continue

        data_slice = cleaned_data[
            (cleaned_data["Datetime"] >= start)
            & (cleaned_data["Datetime"] <= stop)
        ]
        if data_slice.empty or mass_spec_channel not in data_slice.columns:
            continue

        figure, _, _ = plot_channel_data(
            active_channels=[mass_spec_channel],
            cleaned_data=data_slice,
            test_metadata=test_metadata,
            is_table=False,
            channel_map=channel_map,
        )

        meta = test_metadata.copy()
        meta.at['Test Name', 1] = part

        filename = f"{meta.at['Test Section Number', 1]}_{part}_{meta.at['Date Time', 1]}.pdf"
        output_path = pdf_output_path / filename

        pdf = draw_test_details(
            meta,
            transducer_details,
            [mass_spec_channel],
            data_slice,
            output_path,
            False,
            raw_data,
        )
        insert_plot_and_logo(figure, pdf, False)
        generated_paths.append(output_path)

    return generated_paths