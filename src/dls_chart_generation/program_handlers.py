# -*- coding: utf-8 -*-
"""
Implements the Factory pattern for creating report generation strategies.

This module contains the `ReportGeneratorFactory`, which is responsible for
instantiating the correct report generation strategy based on a given
program name. This decouples the main application logic from the concrete
report generator implementations.
"""

from typing import Type, Dict

# Import the base strategy and all concrete strategy implementations
from dls_chart_generation.reports.base_report import ReportStrategy
from dls_chart_generation.reports.generic_report import GenericReportGenerator
from dls_chart_generation.reports.holds_report import HoldsReportGenerator
from dls_chart_generation.reports.breakouts_report import BreakoutsReportGenerator
from dls_chart_generation.reports.signatures_report import SignaturesReportGenerator
from dls_chart_generation.reports.calibration_report import CalibrationReportGenerator
from dls_chart_generation.reports.do_nothing_report import DoNothingReportGenerator


class ReportGeneratorFactory:
    """
    Factory for creating report generator instances based on the program name.

    This class uses the Factory design pattern to provide a central point for
    creating report generator objects.
    """
    _handlers: Dict[str, Type[ReportStrategy]] = {
        # Maps program names from the test details file to the appropriate
        # report generator strategy class.
        "Initial Cycle": GenericReportGenerator,
        "Atmospheric Breakouts": BreakoutsReportGenerator,
        "Atmospheric Cyclic": BreakoutsReportGenerator,
        "Dynamic Cycles PR2": BreakoutsReportGenerator,
        "Dynamic Cycles Petrobras": BreakoutsReportGenerator,
        "Pulse Cycles": GenericReportGenerator,
        "Signatures": SignaturesReportGenerator,
        "Holds-Seat": HoldsReportGenerator,
        "Holds-Body": HoldsReportGenerator,
        "Holds-Body onto Seat": HoldsReportGenerator,
        "Open-Close": BreakoutsReportGenerator,
        "Number Of Turns": DoNothingReportGenerator,
        "Calibration": CalibrationReportGenerator,
        "Data Logger": GenericReportGenerator,
    }

    @staticmethod
    def get_strategy(program_name: str, **kwargs) -> ReportStrategy:
        """
        Retrieves the appropriate report generation strategy and initializes it.

        This method looks up the program name in the `_handlers` dictionary
        and returns an initialized instance of the corresponding strategy class.

        Args:
            program_name: The name of the program to find a handler for.
            **kwargs: Keyword arguments to pass to the strategy's constructor,
                      such as dataframes and file paths.

        Returns:
            An initialized report generation strategy instance.

        Raises:
            ValueError: If no handler is found for the given program name.
        """
        handler_class = ReportGeneratorFactory._handlers.get(program_name)
        if handler_class is None:
            raise ValueError(f"Unsupported program: {program_name}")
        return handler_class(**kwargs)