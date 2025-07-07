import matplotlib.pyplot as plt
import io
from reportlab.lib import colors
from reportlab.lib.colors import Color
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle

def locate_key_time_rows(cleaned_data, key_time_points):
    """Return indices of key time points closest to provided timestamps."""

    time_columns = ["Start of Stabilisation", "Start of Hold", "End of Hold"]
    key_time_indices = key_time_points.copy()
    date_time_index = cleaned_data.set_index('Datetime')

    for col in time_columns:
        key_time_points.at[0, col] = pd.to_datetime(key_time_points.at[0, col], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce', dayfirst=True)
        key_time_indices.at[0, col] = date_time_index.index.get_indexer([key_time_points.at[0, col]], method='nearest')[0]

    return key_time_indices

def locate_bto_btc_rows(raw_data, additional_info):
    bto_indices: list[int] = []
    btc_indices: list[int] = []

    # Early‑exit if data seem to be missing --------------------------------
    if additional_info.iloc[1, 0] == "NaN":
        return additional_info, bto_indices, btc_indices

    torque_data = raw_data["Torque"]
    cycle_count_data = raw_data["Cycle Count"]

    # ── Process one cycle at a time ───────────────────────────────────────
    for i, cycle_num in enumerate(sorted(cycle_count_data.unique())):
        mask = cycle_count_data == cycle_num
        torque_cycle = torque_data[mask]

        n_points = len(torque_cycle)
        if n_points == 0:
            # Skip empty cycles (shouldn't happen, but better to be safe)
            continue

        # Compute slice boundaries — each third is as equal as integer division allows
        third_len = max(1, n_points // 3)
        first_slice  = slice(0, third_len)
        middle_slice = slice(third_len, 2 * third_len)

        torque_first_third  = torque_cycle.iloc[first_slice]
        torque_middle_third = torque_cycle.iloc[middle_slice]

        # ── Determine BTO and BTC values ──────────────────────────────────
        bto = torque_first_third.min().round(1)
        btc = torque_middle_third.max().round(1)

        # Record the row indices at which these extremes occur -------------
        bto_indices.append(int(torque_first_third.idxmin()))
        btc_indices.append(int(torque_middle_third.idxmax()))

        # Write results back into *additional_info* (row offset by +1) -----
        additional_info.iloc[i + 1, 1] = bto
        additional_info.iloc[i + 1, 2] = btc

    return additional_info, bto_indices, btc_indices


def insert_plot_and_logo(figure, pdf, is_gui):
    png_figure = io.BytesIO()
    figure.savefig(png_figure, format='PNG', bbox_inches='tight', dpi=500)
    png_figure.seek(0)
    plt.close(figure)
    fig_img = ImageReader(png_figure)
    pdf.drawImage(fig_img, 16, 67.5, 598, 416.5, preserveAspectRatio=False, mask='auto')
    image_path = 'V:/Userdoc/R & D/Logos/R&D_Page_2.png' if is_gui else '/var/opt/codesys/PlcLogic/R&D_Page_2.png'
    pdf.drawImage(image_path, 629, 515, 197, 65, preserveAspectRatio=True, mask='auto')
    pdf.save()

def draw_bounding_box(pdf_canvas, x, y, width, height):
    pdf_canvas.setLineWidth(0.5)
    pdf_canvas.rect(x, y, width, height)


def draw_text_on_pdf(pdf_canvas, text, x, y, font="Helvetica", colour='black', size=10, left_aligned=False, replace_empty=False):

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
        (630, 271.66, 197, 17.5),   # Test Pressures
        (630, 225.83, 197, 35),     # Breakout Torque
        (630, 35, 197, 180),        # 3rd Party Stamp
        (630, 300, 197, 185)        # Info Right
    ]
    for box in PDF_LAYOUT_BOXES:
        draw_bounding_box(pdf, *box)

def draw_headers(pdf, test_metadata, cleaned_data, light_blue):
    draw_text_on_pdf(pdf,
        f"{test_metadata.at['Test Section Number', 1]} {test_metadata.at['Test Name', 1]}",
        315, 500, font="Helvetica-Bold", size=16)
    draw_text_on_pdf(
        pdf,
        cleaned_data.at[0, 'Datetime'].strftime('%d/%m/%Y'),
        487.5, 539.375,
        colour=light_blue,
        left_aligned=True
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
        (635, 280.41, "Test Pressure", black, False),
        (725, 280.41, f"{test_metadata.at['Test Pressure', 1]} psi", light_blue, True),
        (635, 252.08, "Breakout Torque", black, False),
        (725, 252.08, f"{test_metadata.at['Breakout Torque', 1]} ft.lbs", light_blue, True),
        (635, 234.58, "Running Torque", black, False),
        (725, 234.58, f"{test_metadata.at['Running Torque', 1]} ft.lbs", light_blue, True),
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

def draw_breakouts_table(pdf_canvas, additional_info):
    """Render the simplified Open-Close table."""
    data = additional_info.astype(str).values.tolist()
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 1
    col_width = 600 / cols
    row_height = 51.5 / rows if rows > 0 else 51.5
    table = Table(
        data,
        colWidths=[col_width] * cols,
        rowHeights=[row_height] * rows,
    )
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])
    table.setStyle(style)
    table.wrapOn(pdf_canvas, 600, 51.5)
    table.drawOn(pdf_canvas, 15, 15)

def draw_holds_table(key_time_indices, cleaned_data, positions, black, light_blue):
    indices = key_time_indices.iloc[0]
    main_ch = key_time_indices.iloc[0]['Main Channel']
    for label, ypos, col in [
        ("Start of Stabilisation", 56.5, 'Start of Stabilisation'),
        ("Start of Hold", 41.25, 'Start of Hold'),
        ("End of Hold", 25, 'End of Hold')
    ]:
        idx = indices[col]
        if idx != '':
            time = cleaned_data.at[idx, 'Datetime'].strftime('%d/%m/%Y %H:%M:%S')
            psi  = int(cleaned_data.at[idx, main_ch])
            temp = cleaned_data.at[idx, 'Ambient Temperature']
            positions.extend([
                (20, ypos, label, black, False),
                (120, ypos,
                    f"{time}   {psi} psi   {temp}\u00B0C",
                    light_blue, False
                )
            ])

def draw_all_text(pdf, pdf_text_positions):
    for x, y, text, colour, replace_empty in pdf_text_positions:
        draw_text_on_pdf(pdf, text, x, y, colour=colour, size=10, left_aligned=True, replace_empty=replace_empty)

def draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, pdf_output_path, additional_info, program_name):
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
