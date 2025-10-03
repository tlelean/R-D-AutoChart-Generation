# -*- coding: utf-8 -*-
"""
Defines a strategy that performs no action.
"""

from dls_chart_generation.reports.base_report import ReportStrategy

class DoNothingReportGenerator(ReportStrategy):
    """
    A strategy that performs no action and generates no report. This is used
    for program types that are recognized but do not require a PDF output.
    """
    def generate(self) -> None:
        """This method does nothing and returns None."""
        return None