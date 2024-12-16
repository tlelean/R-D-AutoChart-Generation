import pandas as pd

# Define file paths as variables
data_filepath = r'V:\Userdoc\R & D\DAQ_Station\tlelean\Job Number\Valve Drawing Number\CSV\0.2\Test Description_Data_16-12-2024_16-14-26.csv'
test_details_filepath = r'V:\Userdoc\R & D\DAQ_Station\tlelean\Job Number\Valve Drawing Number\CSV\0.2\Test Description_Test_Details_16-12-2024_16-14-26.csv'
output_pdf_filepath = ''

# Read CSV files using the variables
data = pd.read_csv(data_filepath, header=None)
test_details = pd.read_csv(test_details_filepath, header=None)

# Generate headers
date_time_headers = ['Date', 'Time', 'Milliseconds']
channel_names = test_details.iloc[13:34, 0].reset_index(drop=True).tolist()  # Convert to list

data_headers = date_time_headers + channel_names

# Assign headers to the DataFrame
data.columns = data_headers

# Print the updated DataFrame with headers
print(data)
