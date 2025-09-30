"""
Handles the mapping between default and custom channel names.
"""

# The default channel names in their fixed, expected order.
DEFAULT_CHANNEL_NAMES = [
    "Upstream",
    "Downstream",
    "Body",
    "Actuator",
    "Hyperbaric",
    "Backseat",
    "Spring Chamber",
    "Primary Stem Seal",
    "Secondary Stem Seal",
    "Relief Port",
    "BX Port",
    "Flow Meter",
    "Spare",
    "Mass Spectrometer",
    "LVDT",
    "Torque",
    "Number Of Turns",
    "Motor Speed",
    "Ambient Temperature",
    "Body Temperature",
    "Monitor Temperature",
    "Chamber Temperature",
    "Hyperbaric Water Temperature",
    "Close",
    "Open",
    "Cycle Count",
]

def create_channel_name_mapping(custom_channel_names: list[str]) -> dict[str, str]:
    """
    Creates a mapping from default channel names to custom channel names.

    Args:
        custom_channel_names: A list of the channel names as they appear in the
                              test_details.csv file. This list must be in the
                              same order as the default channel names.

    Returns:
        A dictionary mapping each default name to its corresponding custom name.
        e.g., {'Upstream': 'Environmental Port', 'Downstream': 'Downstream'}
    """
    if len(custom_channel_names) != len(DEFAULT_CHANNEL_NAMES):
        raise ValueError(
            "The number of custom channel names does not match the number of "
            f"default channel names. Expected {len(DEFAULT_CHANNEL_NAMES)}, "
            f"got {len(custom_channel_names)}."
        )

    return dict(zip(DEFAULT_CHANNEL_NAMES, custom_channel_names))
