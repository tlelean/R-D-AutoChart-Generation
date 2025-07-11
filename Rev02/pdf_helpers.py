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

def insert_plot_and_logo(figure, pdf, is_gui, is_table):
    png_figure = io.BytesIO()
    figure.savefig(png_figure, format='PNG', bbox_inches='tight', dpi=500)
    png_figure.seek(0)
    plt.close(figure)
    fig_img = ImageReader(png_figure)
    if is_table:
        pdf.drawImage(
            fig_img,
            16,
            67.5,
            598,
            416.5,
            preserveAspectRatio=False,
            mask="auto",
        )
    else:
        pdf.drawImage(
            fig_img,
            16,
            16,
            598,
            468,
            preserveAspectRatio=False,
            mask="auto",
        )
    image_path = (
        "V:/Userdoc/R & D/Logos/R&D_Page_2.png"
        if is_gui
        else "/var/opt/codesys/PlcLogic/R&D_Page_2.png"
    )
    pdf.drawImage(
        image_path,
        629,
        515,
        197,
        65,
        preserveAspectRatio=True,
        mask="auto",
    )
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

    # Convert text to string or set to "" if None
    text = "" if text is None else str(text)

    if replace_empty:
        # If replace_empty is True, replace if empty/whitespace
        text = "N/A" if not text.strip() else text

    pdf_canvas.setFont(font, size)
    text_width = pdf_canvas.stringWidth(text, font, size)
    text_height = size * 0.7  # Approx. 70% of the font size as a typical measure

    if left_aligned:
        draw_x = x
        draw_y = y - (text_height / 2)
    else:
        draw_x = x - (text_width / 2)
        draw_y = y - (text_height / 2)

    pdf_canvas.setFillColor(colour if colour else colors.black)
    pdf_canvas.drawString(draw_x, draw_y, text)
    pdf_canvas.setFillColor(colors.black)  # Reset to black afterwards

def draw_layout_boxes(pdf):
    PDF_LAYOUT_BOXES = [
        (15, 515, 600, 65),         # Info Top Left
        (15, 66.5, 600, 418.5),     # Graph
        (15, 15, 600, 51.5),        # Graph Index
        (630, 278.75, 197, 17.5),   # Cycle Count
        (630, 257.5, 197, 17.5),    # Test Pressure
        (630, 218.75, 197, 35),     # Breakout Torque
        (630, 35, 197, 180),        # 3rd Party Stamp
        (630, 300, 197, 185)        # Info Right
    ]
    for box in PDF_LAYOUT_BOXES:
        draw_bounding_box(pdf, *box)

def draw_headers(pdf, test_metadata, cleaned_data, light_blue):
    draw_text_on_pdf(
        pdf,
        f"{test_metadata.at['Test Section Number', 1]} {test_metadata.at['Test Name', 1]}",
        315,
        500,
        font="Helvetica-Bold",
        size=16,
    )
    draw_text_on_pdf(
        pdf,
        cleaned_data.iloc[0]["Datetime"].strftime("%d/%m/%Y"),
        487.5,
        539.375,
        colour=light_blue,
        left_aligned=True,
    )
    draw_text_on_pdf(pdf, "Data Recording Equipment Used", 728.5, 475, "Helvetica-Bold", size=12)
    draw_text_on_pdf(pdf, "3rd Party Stamp and Date", 728.5, 45, "Helvetica-Bold", size=12)

def prepare_transducer_dataframe(transducer_details, active_channels):
    empty_rows = pd.DataFrame([["", ""]] * 14)
    used_transducers = transducer_details.loc[active_channels].reset_index(drop=True)
    used_transducers.columns = [0, 1]
    used_transducers = pd.concat([used_transducers, empty_rows], ignore_index=True)
    return used_transducers

def build_static_text_positions(test_metadata, light_blue, black):
    return [
        # Left column
        (20, 571.875, "Test Procedure Reference", black, False),
        (140, 571.875, test_metadata.at['Test Procedure Reference', 1], light_blue, True),
        (20, 555.625, "Unique No.", black, False),
        (140, 555.625, test_metadata.at['Unique Number', 1], light_blue, True),
        (20, 539.375, "R&D Reference", black, False),
        (140, 539.375, test_metadata.at['R&D Reference', 1], light_blue, True),
        (20, 523.125, "Valve Description", black, False),
        (140, 523.125, test_metadata.at['Valve Description', 1], light_blue, True),
        # Right column
        (402.5, 571.875, "Job No.", black, False),
        (487.5, 571.875, test_metadata.at['Job Number', 1], light_blue, True),
        (402.5, 555.625, "Test Description", black, False),
        (487.5, 555.625, test_metadata.at['Test Section Number', 1], light_blue, True),
        (402.5, 539.375, "Test Date", black, False),
        (402.5, 523.125, "Valve Drawing No.", black, False),
        (487.5, 523.125, test_metadata.at['Valve Drawing Number', 1], light_blue, True),
        # Pressures & torques
        (635, 266.25, "Test Pressure", black, False),
        (725, 266.25, f"{test_metadata.at['Test Pressure', 1]} psi", light_blue, True),
        (635, 287.5, "Cycle Count", black, False),
        #(725, 286.25, ),
        (635, 245, "Breakout Torque", black, False),
        (725, 245, f"{test_metadata.at['Breakout Torque', 1]} ft.lbs" if test_metadata.at['Breakout Torque', 1] != "See Table" else "See Table", light_blue, True),
        (635, 227.5, "Running Torque", black, False),
        (725, 227.5, f"{test_metadata.at['Running Torque', 1]} ft.lbs" if test_metadata.at['Running Torque', 1] != "See Table" else "See Table", light_blue, True),
        (635, 457.5, "Data Logger", black, False),
        (725, 457.5, test_metadata.at['Data Logger', 1], light_blue, True),
        (635, 442.5, "Serial No.", black, False),
        (725, 442.5, test_metadata.at['Serial Number', 1], light_blue, True),
        (635, 427.5, "Transducers", black, False),
        (635, 367.5, "Gauges", black, False),
    ]

def build_transducer_and_gauge_positions(used_transducers, light_blue):
    positions = []
    for i in range(15):
        x = 635 + (i % 5) * 39.375
        y = 412.5 - (i // 5) * 15
        positions.append((x, y, used_transducers.iat[i, 0], light_blue, False))
    for i in range(12):
        x = 635 + (i % 4) * 50
        y = 352.5 - (i // 4) * 15
        positions.append((x, y, used_transducers.iat[i, 1], light_blue, False))
    return positions

def build_torque_and_stamp_positions(transducer_details, test_metadata, light_blue, black):
    return [
        (635, 307.5, "Torque Transducer", black, False),
        (725, 307.5, transducer_details.at['Torque', 1], light_blue, True),
        (635, 22.5, "Operative:", black, False),
        (685, 22.5, test_metadata.at['Operative', 1], light_blue, False),
    ]

def draw_table(pdf_canvas, dataframe, x=15, y=15, width=600, height=51.5):
    if dataframe is not None:
        # 1) Build a list-of-lists with the header as the first row
        header = list(dataframe.columns.astype(str))
        body   = dataframe.astype(str).values.tolist()
        data   = [header] + body

        # 2) Recompute rows/cols
        rows = len(data)
        cols = len(data[0])  # guaranteed >= 1

        # 3) Compute uniform cell sizes
        col_width  = width  / cols
        row_height = height / rows

        # 4) Create the Table
        table = Table(
            data,
            colWidths  =[col_width ] * cols,
            rowHeights =[row_height] * rows,
        )

        # 5) Style: give the header a grey background, rest white
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),  # header row
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),     # body
            ('TEXTCOLOR',  (0, 1), (-1, -1), colors.black),
            ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)

        # 6) Draw it
        table.wrapOn(pdf_canvas, width, height)
        table.drawOn(pdf_canvas, x, y)


def draw_all_text(pdf, pdf_text_positions):
    for x, y, text, colour, replace_empty in pdf_text_positions:
        draw_text_on_pdf(pdf, text, x, y, colour=colour, size=10, left_aligned=True, replace_empty=replace_empty)

def draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, pdf_output_path, is_table):
    if is_table:
        test_metadata.at['Breakout Torque', 1] = 'See Table'
        test_metadata.at['Running Torque', 1] = 'See Table'    
    pdf = canvas.Canvas(str(pdf_output_path), pagesize=landscape(A4))
    pdf.setStrokeColor(colors.black)
    draw_layout_boxes(pdf)
    light_blue = Color(0.325, 0.529, 0.761)
    black = Color(0, 0, 0)
    draw_headers(pdf, test_metadata, cleaned_data, light_blue)
    used_transducers = prepare_transducer_dataframe(transducer_details, active_channels)
    pdf_text_positions = build_static_text_positions(test_metadata, light_blue, black)
    pdf_text_positions += build_transducer_and_gauge_positions(used_transducers, light_blue)
    pdf_text_positions += build_torque_and_stamp_positions(transducer_details, test_metadata, light_blue, black)
    draw_all_text(pdf, pdf_text_positions)
    return pdf
