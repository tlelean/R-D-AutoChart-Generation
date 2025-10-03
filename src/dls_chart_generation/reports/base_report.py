# -*- coding: utf-8 -*-
"""
Defines the base strategy for report generation using the Template Method pattern.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .. import config
from ..pdf_helpers import draw_test_details, insert_plot_and_logo
from ..graph_plotter import plot_channel_data

class ReportStrategy(ABC):
    """
    Abstract base class for report generation strategies (Template Method Pattern).

    This class defines the overall structure of the report generation algorithm,
    allowing subclasses to override specific steps.
    """

    def __init__(self, **kwargs):
        """Initializes the report strategy with common data."""
        self.program_name: str = kwargs.get("program_name")
        self.pdf_output_path: Path = kwargs.get("pdf_output_path")
        self.test_metadata: pd.DataFrame = kwargs.get("test_metadata")
        self.transducer_details: pd.DataFrame = kwargs.get("transducer_details")
        self.active_channels: List[str] = kwargs.get("active_channels")
        self.cleaned_data: pd.DataFrame = kwargs.get("cleaned_data")
        self.raw_data: pd.DataFrame = kwargs.get("raw_data")
        self.additional_info: pd.DataFrame = kwargs.get("additional_info")
        self.part_windows: pd.DataFrame = kwargs.get("part_windows")
        self.channels_to_record: pd.DataFrame = kwargs.get("channels_to_record")
        self.channel_map: Dict[str, str] = kwargs.get("channel_map")

    def generate(self) -> Path:
        """
        Template method for generating a standard report.
        This method orchestrates the report creation process.
        """
        is_table = self.is_table_report()
        unique_path = self._build_output_path(self.test_metadata)

        # Step 1: Create the plot
        figure, _, _ = self.create_plot(is_table)

        # Step 2: Create the PDF canvas and draw standard details
        pdf = self.create_pdf(unique_path, is_table)

        # Step 3: Add any additional tables or content to the PDF
        self.add_extra_content(pdf)

        # Step 4: Insert the plot and logo into the PDF
        insert_plot_and_logo(figure, pdf, is_table)

        return unique_path

    def _build_output_path(self, test_metadata: pd.DataFrame) -> Path:
        """Constructs the output PDF path from metadata."""
        date_time_str = test_metadata.at['Date Time', 1].replace(':', '-').replace('/', '-')
        return self.pdf_output_path / (
            f"{test_metadata.at['Test Section Number', 1]} "
            f"{test_metadata.at['Test Name', 1]}_"
            f"{date_time_str}.pdf"
        )

    def _channels_for_main_plot(self, include_mass_spec: bool = False) -> List[str]:
        """
        Returns the active channels for the primary plot, excluding the
        mass spectrometer channel by default.
        """
        channels = list(self.active_channels or [])
        if include_mass_spec:
            return channels

        mass_spec_channel = self.channel_map.get(config.MASS_SPECTROMETER_CHANNEL)
        if mass_spec_channel and mass_spec_channel in channels:
            return [ch for ch in channels if ch != mass_spec_channel]
        return channels

    def create_pdf(self, output_path: Path, is_table: bool):
        """Creates the PDF canvas and draws the standard test details."""
        return draw_test_details(
            test_metadata=self.test_metadata,
            transducer_details=self.transducer_details,
            active_channels=self.active_channels,
            cleaned_data=self.cleaned_data,
            pdf_output_path=output_path,
            is_table=is_table,
            raw_data=self.raw_data,
        )

    @abstractmethod
    def create_plot(self, is_table: bool):
        """
        Abstract method for creating the main plot for the report.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def add_extra_content(self, pdf_canvas: Any):
        """
        Hook for adding extra content (like tables) to the PDF.
        Subclasses can override this method if needed.
        """
        pass  # Default implementation does nothing

    def is_table_report(self) -> bool:
        """
        Determines if the report should include a table.
        Subclasses can override this.
        """
        return False