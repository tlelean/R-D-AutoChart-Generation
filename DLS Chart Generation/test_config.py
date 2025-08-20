"""
Configuration file for test cases.

Each entry in the TEST_CASES list is a dictionary containing the paths
to the data and test details files for a single test run.
"""

TEST_CASES = [
    {
        "primary_data_file": "./Example Data/_Data_13-8-2025_15-9-15.csv",
        "test_details_file": "./Example Data/_Test_Details_13-8-2025_15-9-15.csv",
        "pdf_output_path": "./Example Data/",
    },
    # Add more test cases here as needed, for example:
    # {
    #     "primary_data_file": "./Example Data/another_data_file.csv",
    #     "test_details_file": "./Example Data/another_details_file.csv",
    # },
]
