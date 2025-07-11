"""Plotting utilities for R&D test reports."""

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from plotter_info import (
    CHANNEL_COLOUR_MAP,
    CHANNEL_UNITS_MAP,
    CHANNEL_AXIS_NAMES_MAP,
    AXIS_COLOUR_MAP,
    AXIS_LOCATIONS,
    AXIS_PRIORITY,
)

def plot_crosses(df, channel, data, ax):
    if df is not None:
        idx_cols = [c for c in df.columns if c.endswith("_Index")]
        for col in idx_cols:
            idxs = df[col].astype(int)
            for idx in idxs:
                t = data["Datetime"].loc[idx]
                y = data[channel].loc[idx]
                # compute rough slope
                if 0 < idx < len(data[channel])-1:
                    slope = data[channel].loc[idx+1] - data[channel].loc[idx-1]
                else:
                    slope = 0
                # choose offset above or below
                offset = 5 if slope >= 0 else -5
                va = 'bottom' if offset > 0 else 'top'

                ax.plot(t, y,
                        marker='x',
                        linestyle='none',
                        markersize=8,
                        color='black')
                ax.annotate(
                    col.removesuffix("_Index"),
                    xy=(t, y),
                    xytext=(0, offset),
                    textcoords='offset points',
                    ha='center',
                    va=va,
                    fontsize=10,
                )
                
def axis_location(active_channels):
    """Map each active channel to an axis position."""

    axis_types = [CHANNEL_AXIS_NAMES_MAP.get(ch) for ch in active_channels]
    axis_types_set = set(axis_types)

    axis_types_present = [a for a in AXIS_PRIORITY if a in axis_types_set]

    channel_axis_location_map = {}
    for axis_type, loc in zip(axis_types_present, AXIS_LOCATIONS):
        channel_axis_location_map[axis_type] = loc

    return channel_axis_location_map

def plot_channel_data(active_channels, cleaned_data, test_metadata, is_table):
    """Return matplotlib figure and axes for the given channel data."""

    data_for_plot = cleaned_data.copy()
    data_for_plot["Datetime"] = pd.to_datetime(
        data_for_plot["Datetime"],
        format="%d/%m/%Y %H:%M:%S.%f",
    )

    # Get axis mapping for each channel
    axis_map = axis_location(active_channels)

    # Prepare axes
    if is_table:
        fig, ax_main = plt.subplots(figsize=(11.96, 8.49))
    else:
        fig, ax_main = plt.subplots(figsize=(11.96, 9.37))
    axes = {'left': ax_main}
    color_map = {}
    axis_label_map = {}

    # Create additional axes as needed
    for axis_name in set(axis_map.values()):
        if axis_name == "left":
            continue
        ax = ax_main.twinx()
        if axis_name.startswith("right_"):
            idx = int(axis_name.split("_")[1])
            ax.spines["right"].set_position(("axes", 1 + 0.1 * (idx - 1)))
        axes[axis_name] = ax

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

    return fig, axes, axis_map