import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from plotter_info import CHANNEL_COLOUR_MAP, CHANNEL_UNITS_MAP, CHANNEL_AXIS_NAMES_MAP, AXIS_COLOUR_MAP, AXIS_LOCATIONS, AXIS_PRIORITY

def axis_location(active_channels):
    # Priority order for axis assignment

    axis_types = [CHANNEL_AXIS_NAMES_MAP.get(ch) for ch in active_channels]
    axis_types_set = set(axis_types)

    # Only keep axis types that are present, in priority order
    axis_types_present = [axis for axis in AXIS_PRIORITY if axis in axis_types_set]

    # Assign locations dynamically
    CHANNEL_AXIS_LOCATION_MAP = {}
    for axis_type, axis_location in zip(axis_types_present, AXIS_LOCATIONS):
        CHANNEL_AXIS_LOCATION_MAP[axis_type] = axis_location

    return CHANNEL_AXIS_LOCATION_MAP

def plot_channel_data(active_channels, program_name, cleaned_data, key_time_points, bto_indicies=None, btc_indicies=None, test_metadata=None):
    key_time_indicies = None  # Initialize to None
    data_for_plot = cleaned_data.copy()
    data_for_plot['Datetime'] = pd.to_datetime(data_for_plot['Datetime'], format='%d/%m/%Y %H:%M:%S.%f')

    # Get axis mapping for each channel
    axis_map = axis_location(active_channels)

    # Prepare axes
    fig, ax_main = plt.subplots(figsize=(11.96, 8.49))
    axes = {'left': ax_main}
    color_map = {}
    axis_label_map = {}

    # Create additional axes as needed
    for axis in set(axis_map.values()):
        if axis == 'left':
            continue
        if axis == 'right_1':
            axes[axis] = ax_main.twinx()
        elif axis == 'right_2':
            axes[axis] = ax_main.twinx()
            axes[axis].spines['right'].set_position(('axes', 1.1))
        elif axis == 'right_3':
            axes[axis] = ax_main.twinx()
            axes[axis].spines['right'].set_position(('axes', 1.2))
        elif axis == 'right_4':
            axes[axis] = ax_main.twinx()
            axes[axis].spines['right'].set_position(('axes', 1.3))
        elif axis == 'right_5':
            axes[axis] = ax_main.twinx()
            axes[axis].spines['right'].set_position(('axes', 1.4))
        # Add more elifs for more right axes if needed

    def get_axis_for_channel(channel):
        axis_type = CHANNEL_AXIS_NAMES_MAP.get(channel)
        axis_loc = axis_map.get(axis_type, 'left')
        return axes[axis_loc]

    # Track used axes for legend
    plotted_lines = []
    plotted_labels = []

    for ch in active_channels:
        axis_type = axis_map.get(CHANNEL_AXIS_NAMES_MAP.get(ch))
        if not axis_type:
            continue
        ax = axes[axis_type]
        color = CHANNEL_COLOUR_MAP.get(ch, 'black')
        unit = CHANNEL_UNITS_MAP.get(ch, '')
        if unit:
            label = f"{ch} ({unit})"
        else:
            label = ch
        line, = ax.plot(
            data_for_plot['Datetime'],
            data_for_plot[ch],
            label=label,
            color=color,
            linewidth=1
        )
        color_map[axis_type] = color
        plotted_lines.append(line)
        plotted_labels.append(label)
        # Set axis label for this axis if not already set
        if axis_type not in axis_label_map:
            axis_name = CHANNEL_AXIS_NAMES_MAP.get(ch, '')
            axis_unit = CHANNEL_UNITS_MAP.get(ch, '')
            if axis_unit:
                axis_label_map[axis_type] = f"{axis_name} ({axis_unit})".strip()
            else:
                axis_label_map[axis_type] = axis_name

    # Set axis labels and colors
    for axis_type, ax in axes.items():
        axis_label = axis_label_map.get(axis_type, '')
        axis_name = axis_label.split('(')[0].strip() if axis_label else ''
        axis_color = AXIS_COLOUR_MAP.get(axis_name, color_map.get(axis_type, 'black'))
        if axis_label:
            ax.set_ylabel(axis_label, color=axis_color)
            ax.tick_params(axis='y', colors=axis_color)
        ax.margins(x=0)
        ax.spines['top'].set_visible(False)
        # Set only the visible y spine for each axis
        if axis_type == 'left':
            ax.spines['left'].set_edgecolor(axis_color)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
        else:
            ax.spines['right'].set_edgecolor(axis_color)
            ax.spines['right'].set_linewidth(0.5)
            ax.spines['left'].set_visible(False)
        # Set temperature axis limits and ticks if this is a temperature axis
        if 'Temperature' in axis_name:
            ax.set_ylim(-60, 260)
            ax.yaxis.set_major_locator(MultipleLocator(10))
        # Set pressure axis lower bound to 0 if this is a pressure axis
        if 'Pressure' in axis_name:
            if test_metadata.get('Test Pressure', 0) == 0:
                # keeps the plot looking tidy and the line perfectly flat
                ax.set_ylim(-1, 100)        # or (0, 1) if you prefer starting at zero
            else:
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(0, y_max)
        if 'Valve State' in axis_name:
            ax.set_ylim(-0.05, 1.05)           # Use full axis height for plotting
            ax.set_yticks([0, 1])              # Only show 0 and 1 as ticks
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S'))
        # Set 10 x-ticks, starting and ending at the data range
        x_min = data_for_plot['Datetime'].min()
        x_max = data_for_plot['Datetime'].max()
        x_ticks = pd.date_range(start=x_min, end=x_max, periods=10)
        ax.set_xticks(x_ticks)

        #------------------------------------------------------------------------------
        # Program = Initial Cycle
        #------------------------------------------------------------------------------

        if program_name == "Initial Cycle":
            pass

        #------------------------------------------------------------------------------
        # Program = Holds-Seat or Holds-Body
        #------------------------------------------------------------------------------

        if program_name == "Holds-Seat" or program_name == "Holds-Body":

            # Overlay key time points
            time_columns = ["Start of Stabilisation", "Start of Hold", "End of Hold"]
            key_labels = ['SOS', 'SOH', 'EOH']
            y_min, y_max = axes['left'].get_ylim()
            for col in time_columns:
                if key_time_indicies.iloc[0][col] == '':
                    continue
                x = cleaned_data['Datetime'].loc[key_time_indicies.iloc[0][col]]
                y = cleaned_data[key_time_points.iloc[0]['Main Channel']].loc[key_time_indicies.iloc[0][col]]
                axes['left'].plot(x, y, marker='x', color='black', markersize=10)
                axes['left'].text(
                    x,
                    y + (y_max - y_min) * 0.03,
                    f" {key_labels[time_columns.index(col)]}",
                    color='black',
                    fontsize=10,
                    ha='center',
                    va='bottom'
                )
        #------------------------------------------------------------------------------
        # Program = Atmospheric Breakouts
        #------------------------------------------------------------------------------

        elif program_name == "Atmospheric Breakouts":
            pass

        #------------------------------------------------------------------------------
        # Program = Atmospheric Cyclic
        #------------------------------------------------------------------------------

        elif program_name == "Atmospheric Cyclic":
            pass

        #------------------------------------------------------------------------------
        # Program = Dynamic Cycles PR2
        #------------------------------------------------------------------------------

        elif program_name == "Dynamic Cycles PR2":
            pass

        #------------------------------------------------------------------------------
        # Program = Dynamic Cycles Petrobras
        #------------------------------------------------------------------------------

        elif program_name == "Dynamic Cycles Petrobras":
            pass

        #------------------------------------------------------------------------------
        # Program = Pulse Cycles
        #------------------------------------------------------------------------------

        elif program_name == "Pulse Cycles":
            pass

        #------------------------------------------------------------------------------
        # Program = Signatures
        #------------------------------------------------------------------------------

        elif program_name == "Signatures":
            pass
        #------------------------------------------------------------------------------
        # Program = Open-Close
        #------------------------------------------------------------------------------

        elif program_name == "Open-Close":
            y_min, y_max = axes['left'].get_ylim()
            for idx in bto_indicies:
                x = cleaned_data['Datetime'].iloc[idx]
                y = cleaned_data['Torque'].iloc[idx]
                ax = get_axis_for_channel('Torque')
                ax.plot(x, y, marker='x', color='black', markersize=10)
                ax.text(
                    x,
                    y + (y_max - y_min) * 0.03,
                    f"BTO",
                    color='black',
                    fontsize=10,
                    ha='center',
                    va='bottom'
                )
            for idx in btc_indicies:
                x = cleaned_data['Datetime'].iloc[idx]
                y = cleaned_data['Torque'].iloc[idx]
                ax = get_axis_for_channel('Torque')
                ax.plot(x, y, marker='x', color='black', markersize=10)
                ax.text(
                    x,
                    y + (y_max - y_min) * 0.03,
                    f"BTC",
                    color='black',
                    fontsize=10,
                    ha='center',
                    va='bottom'
                )
        #------------------------------------------------------------------------------
        # Program = Number of Turns
        #------------------------------------------------------------------------------

        elif program_name == "Number Of Turns":
            pass

    # # Create a combined legend
    # fig.legend(plotted_lines, plotted_labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.02))
    # plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Dynamically set legend columns and bottom margin
    max_cols = 5
    n_channels = len(plotted_labels)
    ncol = min(n_channels, max_cols)
    nrows = (n_channels + max_cols - 1) // max_cols
    # Adjust vertical position and bottom margin based on number of rows
    legend_y = 0.02 + 0.03 * (nrows - 1)
    bottom_margin = 0.05 * nrows

    fig.legend(
        plotted_lines,
        plotted_labels,
        loc='lower center',
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y)
    )
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])

    # plt.show()
    
    return fig, key_time_indicies