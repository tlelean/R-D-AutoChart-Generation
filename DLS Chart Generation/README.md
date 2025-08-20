# R&D Test Report Generator

This script generates PDF reports from CSV data collected during R&D testing.

## Prerequisites

Before running the script, ensure you have Python installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

## How to Run

The script can be run in two modes: **Test Mode** and **Implementation Mode**.

### Test Mode

Test mode is designed for running the script on a development machine to verify that code changes have not negatively impacted the output. It processes a predefined list of test cases and generates a PDF for each one.

To run in test mode, use the `--run-tests` flag:

```bash
python3 ./DLS Chart Generation/main.py --run-tests
```

This will:
1. Read the list of test cases from `test_config.py`.
2. For each test case, generate a corresponding PDF report in the `Example Data` directory.
3. You can then visually inspect the generated PDFs to check for any unintended changes.

#### Managing Test Cases

To add, remove, or change test cases, edit the `TEST_CASES` list in the `test_config.py` file. Each test case is a dictionary specifying the paths to the data and details files.

Example `test_config.py`:
```python
TEST_CASES = [
    {
        "primary_data_file": "./Example Data/_Data_13-8-2025_15-9-15.csv",
        "test_details_file": "./Example Data/_Test_Details_13-8-2025_15-9-15.csv",
        "pdf_output_path": "./Example Data/",
    },
    # Add another test case below
    # {
    #     "primary_data_file": "./path/to/your/data.csv",
    #     "test_details_file": "./path/to/your/details.csv",
    # },
]
```

### Implementation Mode

Implementation mode is used when running the script on the target hardware (e.g., a Raspberry Pi) for a single, specific test run. It takes the file paths as command-line arguments.

To run in implementation mode, provide the primary data file and the test details file as arguments:

```bash
python3 main.py path/to/your/_Data_file.csv path/to/your/_Test_Details_file.csv
```

**Note:** The output PDF path is currently hardcoded to `"./Example Data/"`. This can be changed in `main.py` if needed.

You can also specify if the script is being run with a GUI using the `--gui` flag. This affects which logo file is used.

```bash
# Run in implementation mode with the GUI flag
python3 main.py path/to/your/_Data_file.csv path/to/your/_Test_Details_file.csv --gui
```
