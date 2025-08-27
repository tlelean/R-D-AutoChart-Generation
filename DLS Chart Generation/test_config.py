"""
Configuration file for test cases.

Each entry in the TEST_CASES list is a dictionary containing the paths
to the data and test details files for a single test run.
"""

TEST_CASES = [
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Dynamic Cycles PR2/10.0_Data_26-8-2025_12-7-51.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Dynamic Cycles PR2/10.0_Test_Details_26-8-2025_12-7-51.csv",
        "pdf_output_path": "./Example Certs/",
    },
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Holds Seat/4.0_Data_22-8-2025_11-47-35.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Holds Seat/4.0_Test_Details_22-8-2025_11-47-35.csv",
        "pdf_output_path": "./Example Certs/",
    },
        # Add more test cases here as needed, for example:
    # {
    #     "primary_data_file": "./Example Data/another_data_file.csv",
    #     "test_details_file": "./Example Data/another_details_file.csv",
    # },
    # Add more test cases here as needed, for example:
    # {
    #     "primary_data_file": "./Example Data/another_data_file.csv",
    #     "test_details_file": "./Example Data/another_details_file.csv",
    #     "pdf_output_path": "./Example Data/",
    # },
]
