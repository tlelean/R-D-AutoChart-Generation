"""
Configuration file for test cases.

Each entry in the TEST_CASES list is a dictionary containing the paths
to the data and test details files for a single test run.
"""

TEST_CASES = [
    # PR2 Dynamic Cycles
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/PR2 Dynamic Cycles/10.0_Data_26-8-2025_12-7-51.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/PR2 Dynamic Cycles/10.0_Test_Details_26-8-2025_12-7-51.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Holds Seat
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Holds Seat/13.0_Data_1-9-2025_13-26-30.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Holds Seat/13.0_Test_Details_1-9-2025_13-26-30.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Mass Spec
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Mass Spec/_Data_3-9-2025_15-24-3.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Mass Spec/_Test_Details_3-9-2025_15-24-3.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Calibration
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Calibration/Calibration_Data_1-10-2025_11-45-17.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Calibration/Calibration_Test_Details_1-10-2025_11-45-17.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Initial Cycle
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Initial Cycle/_Data_8-9-2025_9-9-42.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Initial Cycle/_Test_Details_8-9-2025_9-9-42.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Atmospheric Breakouts
    {
        "primary_data_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Atmospheric Breakouts/Calibration_Data_1-10-2025_16-23-24.csv",
        "test_details_file": "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/Example Data/Atmospheric Breakouts/Calibration_Test_Details_1-10-2025_16-23-24.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Calibration Mass Spec
    {
        "primary_data_file": "V:/Userdoc/R & D/DAQ_Station/Calibration/DLS005 Right Hand Large Temperature Cabinet/CSV/Calibration_Data_30-9-2025_15-32-33.csv",
        "test_details_file": "V:/Userdoc/R & D/DAQ_Station/Calibration/DLS005 Right Hand Large Temperature Cabinet/CSV/Calibration_Test_Details_30-9-2025_15-32-33.csv",
        "pdf_output_path": "./Example Certs/",
    },
    # Calibration Mass Spec Mantissa
    {
        "primary_data_file": "V:/Userdoc/R & D/DAQ_Station/Calibration/DLS005 Right Hand Large Temperature Cabinet/CSV/Calibration_Data_30-9-2025_15-29-39.csv",
        "test_details_file": "V:/Userdoc/R & D/DAQ_Station/Calibration/DLS005 Right Hand Large Temperature Cabinet/CSV/Calibration_Test_Details_30-9-2025_15-29-39.csv",
        "pdf_output_path": "./Example Certs/",
    },
]