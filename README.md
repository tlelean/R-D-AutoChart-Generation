# R&D AutoChart Generation

This repository contains utilities for generating PDF charts from CSV test data.

## Running with custom data

On the Raspberry Pi or any system with Python and the required libraries
installed, you can run the program by providing the paths to the required
files:

```bash
python Rev02/main.py <primary_data.csv> <test_details.csv> <output_dir> <True|False>
```

`True` enables GUI related behaviour, while `False` runs the program headless.

## Quick testing with example data

For development purposes a collection of sample CSV files is included under the
`Example Data` directory.  The `main.py` script now supports a convenient
`--example` flag to run directly against these datasets.

To test using the `Holds` example, run:

```bash
python Rev02/main.py --example Holds
```

Available example names correspond to the folders under
`Example Data/Hydraulic` (e.g. `Atmospheric Breakouts`, `Dynamic Cycles PR2`,
`Dynamic Cycles Petrobras`, `Holds`, `Open-Close`).  When `--example` is
provided the script automatically loads the matching `primary_data.csv` and
`test_details.csv` and writes the output PDF to the same folder.

The original command line behaviour using explicit file paths continues to
work as before.
