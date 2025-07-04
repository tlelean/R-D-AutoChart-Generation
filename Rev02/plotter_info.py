# Colour mapping for known channels
CHANNEL_COLOUR_MAP = {
    'Upstream': '#e6194b',            # Vivid Red
    'Downstream': '#3cb44b',          # Vivid Green
    'Body': '#f58231',                # Vivid Orange
    'Actuator': '#000000',            # Black
    'Hyperbaric': '#0082c8',          # Strong Blue
    'Backseat': '#ffe119',            # Bright Yellow
    'Spring Chamber': '#911eb4',      # Vivid Purple
    'Primary Stem Seal': '#00ced1',   # Dark Turquoise
    'Secondary Stem Seal': '#f032e6', # Bright Magenta
    'Relief Port': '#6bb300',         # Strong Olive Green
    'BX Port': '#e377c2',             # Dusty Pink
    'Spare': '#005f5f',               # Dark Teal
    'Leak Rate': '#9f7fff',           # Medium Lavender
    'Mass Spectrometer': '#8b4513',   # Saddle Brown
    'LVDT': '#800000',                # Maroon
    'Torque': '#b2a400',              # Dark Gold
    'Number Of Turns': '#000075',     # Navy Blue
    'Motor Speed': '#00a572',         # Emerald Green
    'Ambient Temperature': '#4363d8', # Vivid Blue
    'Body Temperature': '#ff8c00',    # Dark Orange
    'Monitor Temperature': '#5c5c00', # Dark Olive
    'Chamber Temperature': '#cd5c5c', # Indian Red
    'Hyperbaric Temperature': '#696969', # Dim Grey
    'Close' : "#ff1493",               # Deep Pink
    'Open' : '#1e90ff'                 # Dodger Blue
}

CHANNEL_UNITS_MAP = {
    'Upstream': 'psi',
    'Downstream': 'psi',
    'Body': 'psi',
    'Actuator': 'psi',
    'Hyperbaric': 'psi',
    'Backseat': 'psi',
    'Spring Chamber': 'psi',
    'Primary Stem Seal': 'psi',
    'Secondary Stem Seal': 'psi',
    'Relief Port': 'psi',
    'BX Port': 'psi',
    'Spare': 'N/A',
    'Leak Rate': 'ml/min',
    'Mass Spectrometer': 'mbarl/sec',
    'LVDT': 'mm',
    'Torque': 'lbs-ft',
    'Number Of Turns': '',
    'Motor Speed': 'rpm',
    'Ambient Temperature': '°C',
    'Body Temperature': '°C',
    'Monitor Temperature': '°C',
    'Chamber Temperature': '°C',
    'Hyperbaric Temperature': '°C',
    'Close': '',
    'Open': ''
}

CHANNEL_AXIS_NAMES_MAP = {
    'Upstream': 'Pressure',
    'Downstream': 'Pressure',
    'Body': 'Pressure',
    'Actuator': 'Pressure',
    'Hyperbaric': 'Pressure',
    'Backseat': 'Pressure',
    'Spring Chamber': 'Pressure',
    'Primary Stem Seal': 'Pressure',
    'Secondary Stem Seal': 'Pressure',
    'Relief Port': 'Pressure',
    'BX Port': 'Pressure',
    'Spare': '',
    'Leak Rate': 'Leak Rate',
    'Mass Spectrometer': 'Mass Spectrometer',
    'LVDT': 'LVDT',
    'Torque': 'Torque',
    'Number Of Turns': 'Number Of Turns',
    'Motor Speed': 'Motor Speed',
    'Ambient Temperature': 'Temperature',
    'Body Temperature': 'Temperature',
    'Monitor Temperature': 'Temperature',
    'Chamber Temperature': 'Temperature',
    'Hyperbaric Temperature': 'Temperature',
    'Close': 'Valve State',
    'Open': 'Valve State'
}

AXIS_COLOUR_MAP = {
    'Pressure': '#e6194b',              # Vivid Red
    'Leak Rate': '#9f7fff',             # Medium Lavender
    'Mass Spectrometer': '#8b4513',     # Saddle Brown
    'LVDT': '#800000',                  # Maroon
    'Torque': '#b2a400',                # Dark Gold
    'Number Of Turns': '#000075',       # Navy Blue
    'Motor Speed': '#00a572',           # Emerald Green
    'Temperature': '#4363d8',           # Vivid Blue
    'Valve State' : "#ff1493"
}


AXIS_PRIORITY = [
    'Pressure', 
    'Torque', 
    'Pneumatic', 
    'Leak Rate', 
    'Mass Spectrometer',
    'Number Of Turns', 
    'Motor Speed', 
    'LVDT',
    'Valve State',
    'Temperature'
]

AXIS_LOCATIONS = [
    'left',
    'right_1', 
    'right_2', 
    'right_3', 
    'right_4', 
    'right_5', 
    'right_6', 
    'right_7', 
    'right_8', 
    'right_9'
]