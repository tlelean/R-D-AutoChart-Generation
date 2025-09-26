"""Utilities for creating PDF reports from test data."""

import io
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
import os

def format_torque(value):
    """Return a torque string with units or special cases."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"See Table", "N/A"} or stripped.endswith("ft.lbs"):
            return stripped
    try:
        if float(value) == 0:
            return "N/A"
    except (TypeError, ValueError):
        return f"{value} ft.lbs"
    return f"{value} ft.lbs"

class Layout:
    """Class to hold all layout constants for the PDF report."""
    PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)

    # Margins
    MARGIN_LEFT = 15
    MARGIN_RIGHT = 15
    MARGIN_TOP = 15
    MARGIN_BOTTOM = 15

    # Main content area
    CONTENT_X_START = MARGIN_LEFT
    CONTENT_Y_START = MARGIN_BOTTOM
    CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    CONTENT_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    # Header section
    HEADER_X = CONTENT_X_START
    HEADER_Y = 515
    HEADER_W = 600
    HEADER_H = 65

    # Graph index/table section
    TABLE_X = CONTENT_X_START
    TABLE_Y = CONTENT_Y_START
    TABLE_W = HEADER_W
    TABLE_H = 51.5

    # Graph section
    GRAPH_X = CONTENT_X_START
    GRAPH_Y_NO_TABLE = CONTENT_Y_START
    GRAPH_H_NO_TABLE = 470
    GRAPH_Y_TABLE = CONTENT_Y_START + TABLE_H
    GRAPH_H_TABLE = GRAPH_H_NO_TABLE - TABLE_H
    GRAPH_W = HEADER_W

    # Right-hand side boxes
    RIGHT_COL_X = 630
    RIGHT_COL_W = 197

    LOGO_X = RIGHT_COL_X
    LOGO_Y = 515
    LOGO_W = 197
    LOGO_H = 65

    INFO_RIGHT_X = RIGHT_COL_X
    INFO_RIGHT_Y = 300
    INFO_RIGHT_W = RIGHT_COL_W
    INFO_RIGHT_H = 185

    CYCLE_COUNT_X = RIGHT_COL_X
    CYCLE_COUNT_Y = 278.75
    CYCLE_COUNT_W = RIGHT_COL_W
    CYCLE_COUNT_H = 17.5

    TEST_PRESSURE_X = RIGHT_COL_X
    TEST_PRESSURE_Y = 257.5
    TEST_PRESSURE_W = RIGHT_COL_W
    TEST_PRESSURE_H = 17.5

    BREAKOUT_TORQUE_X = RIGHT_COL_X
    BREAKOUT_TORQUE_Y = 218.75
    BREAKOUT_TORQUE_W = RIGHT_COL_W
    BREAKOUT_TORQUE_H = 35

    STAMP_X = RIGHT_COL_X
    STAMP_Y = 35
    STAMP_W = RIGHT_COL_W
    STAMP_H = 180

    # Text positions
    MAIN_TITLE_X = 315
    MAIN_TITLE_Y = 500

    HEADER_COL1_LABEL_X = 20
    HEADER_COL1_VALUE_X = 140
    HEADER_COL2_LABEL_X = 402.5
    HEADER_COL2_VALUE_X = 487.5

    HEADER_ROW1_Y = 571.875
    HEADER_ROW2_Y = 555.625
    HEADER_ROW3_Y = 539.375
    HEADER_ROW4_Y = 523.125

    RIGHT_COL_LABEL_X = 635
    RIGHT_COL_VALUE_X = 725

    EQUIPMENT_TITLE_Y = 475
    DATA_LOGGER_Y = 457.5
    SERIAL_NO_Y = 442.5
    TRANSDUCERS_Y = 427.5
    GAUGES_Y = 367.5

    TORQUE_TRANSDUCER_Y = 307.5
    CYCLE_COUNT_TEXT_Y = 287.5
    TEST_PRESSURE_TEXT_Y = 266.25
    BREAKOUT_TORQUE_TEXT_Y = 245
    RUNNING_TORQUE_TEXT_Y = 227.5

    STAMP_TITLE_Y = 45
    OPERATIVE_Y = 22.5
    OPERATIVE_VALUE_X = 685

    TRANSDUCER_TABLE_START_X = 635
    TRANSDUCER_TABLE_START_Y = 412.5
    TRANSDUCER_COL_WIDTH = 39.375
    TRANSDUCER_ROW_HEIGHT = 15

    GAUGE_TABLE_START_X = 635
    GAUGE_TABLE_START_Y = 352.5
    GAUGE_COL_WIDTH = 50
    GAUGE_ROW_HEIGHT = 15

def insert_plot_and_logo(figure, pdf, is_table):
    png_figure = io.BytesIO()
    figure.savefig(png_figure, format='PNG', bbox_inches='tight', dpi=500)
    png_figure.seek(0)
    plt.close(figure)
    fig_img = ImageReader(png_figure)

    if is_table:
        pdf.drawImage(
            fig_img,
            Layout.GRAPH_X+1,
            Layout.GRAPH_Y_TABLE+1,
            Layout.GRAPH_W-2,
            Layout.GRAPH_H_TABLE-2,
            preserveAspectRatio=False,
            mask="auto",
        )
    else:
        pdf.drawImage(
            fig_img,
            Layout.GRAPH_X+1,
            Layout.GRAPH_Y_NO_TABLE+1,
            Layout.GRAPH_W-2,
            Layout.GRAPH_H_NO_TABLE-2,
            preserveAspectRatio=False,
            mask="auto",
        )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "R&D.png")

    try:
        pdf.drawImage(
            image_path,
            Layout.LOGO_X,
            Layout.LOGO_Y,
            Layout.LOGO_W,
            Layout.LOGO_H,
            preserveAspectRatio=True,
            mask="auto",
        )
    except Exception as e:
        print(f"Warning: Could not load logo image at {image_path}. Error: {e}")

    pdf.save()

def draw_bounding_box(pdf_canvas, x, y, width, height):
    pdf_canvas.setLineWidth(0.5)
    pdf_canvas.rect(x, y, width, height)

def draw_text_on_pdf(
    pdf_canvas,
    text,
    x,
    y,
    font="Helvetica",
    colour="black",
    size=10,
    left_aligned=False,
    replace_empty=False,
):
    text = "" if text is None else str(text)
    if replace_empty:
        text = "N/A" if not text.strip() else text

    pdf_canvas.setFont(font, size)
    text_width = pdf_canvas.stringWidth(text, font, size)
    text_height = size * 0.7

    draw_x = x if left_aligned else x - (text_width / 2)
    draw_y = y - (text_height / 2)

    pdf_canvas.setFillColor(colour if colour else colors.black)
    pdf_canvas.drawString(draw_x, draw_y, text)
    pdf_canvas.setFillColor(colors.black)

def draw_layout_boxes(pdf, is_table):
    PDF_LAYOUT_BOXES = [
        (Layout.HEADER_X, Layout.HEADER_Y, Layout.HEADER_W, Layout.HEADER_H),
        (Layout.GRAPH_X, Layout.GRAPH_Y_TABLE, Layout.GRAPH_W, Layout.GRAPH_H_TABLE) if is_table else (Layout.GRAPH_X, Layout.GRAPH_Y_NO_TABLE, Layout.GRAPH_W, Layout.GRAPH_H_NO_TABLE),
        (Layout.TABLE_X, Layout.TABLE_Y, Layout.TABLE_W, Layout.TABLE_H) if is_table else (0, 0, 0, 0),
        (Layout.CYCLE_COUNT_X, Layout.CYCLE_COUNT_Y, Layout.CYCLE_COUNT_W, Layout.CYCLE_COUNT_H),
        (Layout.TEST_PRESSURE_X, Layout.TEST_PRESSURE_Y, Layout.TEST_PRESSURE_W, Layout.TEST_PRESSURE_H),
        (Layout.BREAKOUT_TORQUE_X, Layout.BREAKOUT_TORQUE_Y, Layout.BREAKOUT_TORQUE_W, Layout.BREAKOUT_TORQUE_H),
        (Layout.STAMP_X, Layout.STAMP_Y, Layout.STAMP_W, Layout.STAMP_H),
        (Layout.INFO_RIGHT_X, Layout.INFO_RIGHT_Y, Layout.INFO_RIGHT_W, Layout.INFO_RIGHT_H),
    ]
    for box in PDF_LAYOUT_BOXES:
        draw_bounding_box(pdf, *box)

def draw_headers(pdf, test_metadata, cleaned_data, light_blue):
    draw_text_on_pdf(
        pdf,
        f"{test_metadata.at['Test Section Number', 1]} {test_metadata.at['Test Name', 1]}",
        Layout.MAIN_TITLE_X,
        Layout.MAIN_TITLE_Y,
        font="Helvetica-Bold",
        size=16,
    )
    draw_text_on_pdf(
        pdf,
        cleaned_data.iloc[0]["Datetime"].strftime("%d/%m/%Y"),
        Layout.HEADER_COL2_VALUE_X,
        Layout.HEADER_ROW3_Y,
        colour=light_blue,
        left_aligned=True,
    )
    draw_text_on_pdf(
        pdf, "Data Recording Equipment Used", Layout.RIGHT_COL_X + (Layout.RIGHT_COL_W / 2), Layout.EQUIPMENT_TITLE_Y, "Helvetica-Bold", size=12
    )
    draw_text_on_pdf(
        pdf, "3rd Party Stamp and Date", Layout.RIGHT_COL_X + (Layout.RIGHT_COL_W / 2), Layout.STAMP_TITLE_Y, "Helvetica-Bold", size=12
    )

def prepare_transducer_dataframe(transducer_details, active_channels):
    empty_rows = pd.DataFrame([["", ""]] * 14)
    used_transducers = transducer_details.loc[active_channels].reset_index(drop=True)
    used_transducers.columns = [0, 1]
    used_transducers = pd.concat([used_transducers, empty_rows], ignore_index=True)
    return used_transducers

def build_static_text_positions(test_metadata, light_blue, black, max_cycle=None):
    return [
        (Layout.HEADER_COL1_LABEL_X, Layout.HEADER_ROW1_Y, "Test Procedure Reference", black, False),
        (Layout.HEADER_COL1_VALUE_X, Layout.HEADER_ROW1_Y, test_metadata.at['Test Procedure Reference', 1], light_blue, True),
        (Layout.HEADER_COL1_LABEL_X, Layout.HEADER_ROW2_Y, "Unique No.", black, False),
        (Layout.HEADER_COL1_VALUE_X, Layout.HEADER_ROW2_Y, test_metadata.at['Unique Number', 1], light_blue, True),
        (Layout.HEADER_COL1_LABEL_X, Layout.HEADER_ROW3_Y, "R&D Reference", black, False),
        (Layout.HEADER_COL1_VALUE_X, Layout.HEADER_ROW3_Y, test_metadata.at['R&D Reference', 1], light_blue, True),
        (Layout.HEADER_COL1_LABEL_X, Layout.HEADER_ROW4_Y, "Valve Description", black, False),
        (Layout.HEADER_COL1_VALUE_X, Layout.HEADER_ROW4_Y, test_metadata.at['Valve Description', 1], light_blue, True),

        (Layout.HEADER_COL2_LABEL_X, Layout.HEADER_ROW1_Y, "Job No.", black, False),
        (Layout.HEADER_COL2_VALUE_X, Layout.HEADER_ROW1_Y, test_metadata.at['Job Number', 1], light_blue, True),
        (Layout.HEADER_COL2_LABEL_X, Layout.HEADER_ROW2_Y, "Test Description", black, False),
        (Layout.HEADER_COL2_VALUE_X, Layout.HEADER_ROW2_Y, test_metadata.at['Test Section Number', 1], light_blue, True),
        (Layout.HEADER_COL2_LABEL_X, Layout.HEADER_ROW3_Y, "Test Date", black, False),
        (Layout.HEADER_COL2_LABEL_X, Layout.HEADER_ROW4_Y, "Valve Drawing No.", black, False),
        (Layout.HEADER_COL2_VALUE_X, Layout.HEADER_ROW4_Y, test_metadata.at['Valve Drawing Number', 1], light_blue, True),

        (Layout.RIGHT_COL_LABEL_X, Layout.TEST_PRESSURE_TEXT_Y, "Test Pressure", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.TEST_PRESSURE_TEXT_Y, f"{test_metadata.at['Test Pressure', 1]} psi", light_blue, True),
        (Layout.RIGHT_COL_LABEL_X, Layout.CYCLE_COUNT_TEXT_Y, "Cycle Count", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.CYCLE_COUNT_TEXT_Y - 1.25, f"{max_cycle}" if max_cycle is not None else "", light_blue, True),
        (Layout.RIGHT_COL_LABEL_X, Layout.BREAKOUT_TORQUE_TEXT_Y, "Breakout Torque", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.BREAKOUT_TORQUE_TEXT_Y, format_torque(test_metadata.at['Breakout Torque', 1]), light_blue, True),
        (Layout.RIGHT_COL_LABEL_X, Layout.RUNNING_TORQUE_TEXT_Y, "Running Torque", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.RUNNING_TORQUE_TEXT_Y, format_torque(test_metadata.at['Running Torque', 1]), light_blue, True),

        (Layout.RIGHT_COL_LABEL_X, Layout.DATA_LOGGER_Y, "Data Logger", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.DATA_LOGGER_Y, test_metadata.at['Data Logger', 1], light_blue, True),
        (Layout.RIGHT_COL_LABEL_X, Layout.SERIAL_NO_Y, "Serial No.", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.SERIAL_NO_Y, test_metadata.at['Serial Number', 1], light_blue, True),
        (Layout.RIGHT_COL_LABEL_X, Layout.TRANSDUCERS_Y, "Transducers", black, False),
        (Layout.RIGHT_COL_LABEL_X, Layout.GAUGES_Y, "Gauges", black, False),
    ]

def build_transducer_and_gauge_positions(used_transducers, light_blue):
    positions = []
    for i in range(15):
        x = Layout.TRANSDUCER_TABLE_START_X + (i % 5) * Layout.TRANSDUCER_COL_WIDTH
        y = Layout.TRANSDUCER_TABLE_START_Y - (i // 5) * Layout.TRANSDUCER_ROW_HEIGHT
        positions.append((x, y, used_transducers.iat[i, 0], light_blue, False))
    for i in range(12):
        x = Layout.GAUGE_TABLE_START_X + (i % 4) * Layout.GAUGE_COL_WIDTH
        y = Layout.GAUGE_TABLE_START_Y - (i // 4) * Layout.GAUGE_ROW_HEIGHT
        positions.append((x, y, used_transducers.iat[i, 1], light_blue, False))
    return positions

def build_torque_and_stamp_positions(transducer_details, test_metadata, light_blue, black):
    return [
        (Layout.RIGHT_COL_LABEL_X, Layout.TORQUE_TRANSDUCER_Y, "Torque Transducer", black, False),
        (Layout.RIGHT_COL_VALUE_X, Layout.TORQUE_TRANSDUCER_Y, transducer_details.at['Torque', 1], light_blue, True),
        (Layout.RIGHT_COL_LABEL_X, Layout.OPERATIVE_Y, "Operative:", black, False),
        (Layout.OPERATIVE_VALUE_X, Layout.OPERATIVE_Y, test_metadata.at['Operative', 1], light_blue, False),
    ]

def draw_table(pdf_canvas, dataframe, x=15, y=15, width=600, height=51.5):
    """Render a pandas DataFrame as a table on the PDF canvas."""
    if dataframe is None:
        return

    # Remove all-null columns (as you had)
    df = dataframe.dropna(axis=1, how="all")

    idx_labels  = list(df.index.astype(str))

    # Body: each row starts with the index label, then row values
    body_rows = [[idx_labels[i]] + [str(v) for v in row] for i, row in enumerate(df.values.tolist())]

    data = body_rows

    # ---- Dimensions ----
    rows = len(data)
    cols = len(data[0])

    # Give the first (index) column a bit more width
    col_widths = width / cols

    row_height = height / rows

    table = Table(
        data,
        colWidths=col_widths,
        rowHeights=[row_height] * rows,
    )

    style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.black),

        # Body
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR',  (0, 1), (-1, -1), colors.black),

        # Alignment: centre first column (row labels) AND header row cell
        ('ALIGN',      (0, 0), (0, -1), 'CENTER'),
        ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),

        ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.black),
    ])

    error_labels = [
        "Abs Error (µA) - ±3.6 mV",
        "Abs Error (mV) - ±0.12 mV"
    ]

    error_row_idx = None
    selected_label = None

    for i, row_label in enumerate(df.index):
        if row_label in error_labels:
            error_row_idx = i
            selected_label = row_label
            break

    if error_row_idx is not None:
        # numeric values from the DataFrame (skip None)
        numeric_row = pd.to_numeric(df.loc[selected_label], errors="coerce")

        # Set threshold based on which label matched
        threshold = 3.6 if "µA" in selected_label else 0.12

        # In the table, data columns start at col 1 (col 0 is the index labels)
        for col_offset, val in enumerate(numeric_row, start=1):
            if pd.isna(val):
                continue
            if abs(val) < threshold:
                style.add('BACKGROUND', (col_offset, error_row_idx), (col_offset, error_row_idx), colors.limegreen)
            else:
                style.add('BACKGROUND', (col_offset, error_row_idx), (col_offset, error_row_idx), colors.red)

    table.setStyle(style)

    # ---- Draw ----
    table.wrapOn(pdf_canvas, width, height)
    table.drawOn(pdf_canvas, x, y)

def draw_all_text(pdf, pdf_text_positions):
    for x, y, text, colour, replace_empty in pdf_text_positions:
        draw_text_on_pdf(pdf, text, x, y, colour=colour, size=10, left_aligned=True, replace_empty=replace_empty)

def draw_test_details(
    test_metadata,
    transducer_details,
    active_channels,
    cleaned_data,
    pdf_output_path,
    is_table,
    raw_data,
    has_breakout_table: bool = False,
):
    if is_table and has_breakout_table:
        for field in ("Breakout Torque", "Running Torque"):
            test_metadata.at[field, 1] = "See Table"
    else:
        for field in ("Breakout Torque", "Running Torque"):
            test_metadata.at[field, 1] = format_torque(
                test_metadata.at[field, 1]
            )
    pdf = canvas.Canvas(str(pdf_output_path), pagesize=landscape(A4))
    pdf.setStrokeColor(colors.black)
    draw_layout_boxes(pdf, is_table)
    light_blue = Color(0.325, 0.529, 0.761)
    black = Color(0, 0, 0)
    draw_headers(pdf, test_metadata, cleaned_data, light_blue)
    used_transducers = prepare_transducer_dataframe(transducer_details, active_channels)
    max_cycle = int(raw_data["Cycle Count"].max())
    pdf_text_positions = build_static_text_positions(test_metadata, light_blue, black, max_cycle)
    pdf_text_positions += build_transducer_and_gauge_positions(used_transducers, light_blue)
    pdf_text_positions += build_torque_and_stamp_positions(transducer_details, test_metadata, light_blue, black)
    draw_all_text(pdf, pdf_text_positions)
    return pdf
