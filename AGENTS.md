# Agent Instructions for DLS Chart Generation Project

This document provides guidance for AI agents working on this codebase.

## Project Overview

This project is a command-line tool for generating PDF reports from scientific data. The goal of the recent refactoring was to improve code quality, maintainability, and extensibility.

## Key Architectural Concepts

- **`src` directory:** All source code is located in the `src` directory.
- **`dls_chart_generation` package:** The main application code is within this package.
- **`config.py`:** All configuration and "magic numbers" should be stored here.
- **`data_loading.py`:** Handles all data ingestion and validation.
- **`reports` package:** Contains the different report generation strategies.
  - **`base_report.py`:** Defines the `ReportStrategy` abstract base class.
  - Each report type has its own module (e.g., `holds_report.py`).
- **`program_handlers.py`:** Contains the `ReportGeneratorFactory` for creating report generators.
- **`utils` package:** Contains helper modules for tasks like PDF generation and data analysis.

## Development Guidelines

1.  **Follow PEP 8:** Adhere to Python's official style guide.
2.  **Single Responsibility Principle:** Each function, class, and module should have a single, well-defined responsibility.
3.  **Use the Factory Pattern:** When adding new report types, register them in the `ReportGeneratorFactory` in `program_handlers.py`.
4.  **Use the Strategy Pattern:** New report generators must inherit from the `ReportStrategy` ABC and implement the required methods.
5.  **Configuration over Hard-coding:** Do not hard-code values. Add them to `config.py` instead.
6.  **Write Unit Tests:** All new features and bug fixes must be accompanied by unit tests. Place them in the `tests` directory, mirroring the `src` structure.
7.  **Keep Documentation Updated:** Update the `README.md` and docstrings as you make changes.

## Running Tests

To run the test suite, use the following command:

```bash
python -m pytest
```

Ensure all tests are passing before submitting any changes.