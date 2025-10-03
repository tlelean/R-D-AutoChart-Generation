# DLS Chart Generation

This project is a command-line tool for generating PDF reports from scientific data, specifically for R&D test reports. It processes CSV files containing test data, analyzes it according to different program specifications, and generates detailed PDF reports with plots and tables.

## Features

- **Data Processing:** Ingests and processes data from CSV files.
- **Multiple Report Types:** Supports various report generation strategies for different test programs (e.g., "Holds", "Breakouts", "Signatures", "Calibration").
- **PDF Generation:** Creates professional-looking PDF reports with graphs, tables, and test metadata.
- **Command-Line Interface:** Provides a simple CLI for running the report generation process.
- **Test Mode:** Includes a test mode for batch processing of multiple test cases.

## Project Structure

The project is organized as follows:

```
.
├── src/
│   └── dls_chart_generation/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── data_loading.py
│       ├── reports/
│       │   ├── __init__.py
│       │   ├── base_report.py
│       │   ├── generic_report.py
│       │   └── ... (other report types)
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── pdf_helpers.py
│       │   └── ... (other helper modules)
│       └── tests/
│           ├── __init__.py
│           └── ... (test files)
├── data/
│   ├── sample_data.csv
│   └── ... (other data files)
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dls-chart-generation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application can be run from the command line in two modes:

### Standard Mode

In standard mode, you provide the paths to the primary data file, the test details file, and the output PDF path.

```bash
python src/dls_chart_generation/main.py <primary_data_file> <test_details_file> <pdf_output_path>
```

### Test Mode

In test mode, the application processes a series of predefined test cases from `src/dls_chart_generation/test_config.py`.

```bash
python src/dls_chart_generation/main.py --run-tests
```

## Extending the Application

To add a new report type:

1.  Create a new report generator class in the `src/dls_chart_generation/reports/` directory that inherits from `BaseReportGenerator`.
2.  Implement the `generate` method to define the specific logic for the new report type.
3.  Register the new report generator in the `HANDLERS` dictionary in `src/dls_chart_generation/program_handlers.py`.