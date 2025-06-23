import matplotlib.pyplot as plt
import io
from reportlab.lib import colors
from reportlab.lib.colors import Color
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader

def insert_plot_and_logo(figure, pdf, is_gui):
    """
    Convert a matplotlib Figure to a BytesIO stream (PNG format),
    for later embedding into a PDF.

    Parameters:
        figure (matplotlib.figure.Figure): The matplotlib figure to convert.

    Returns:
        io.BytesIO: A binary stream containing the rendered PNG.
    """
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
    """
    Draw a rectangular bounding box on the PDF canvas.

    Parameters:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas.
        x (float): X-coordinate of the lower-left corner.
        y (float): Y-coordinate of the lower-left corner.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
    """
    pdf_canvas.setLineWidth(0.5)
    pdf_canvas.rect(x, y, width, height)


def draw_text_on_pdf(pdf_canvas, text, x, y, font="Helvetica", colour='black', size=10, left_aligned=False, replace_empty=False):
    """
    Draw text onto the PDF canvas, with options for alignment (left or centre).
    
    If the text is a Timestamp, it can optionally preserve only the date (dd/mm/yyyy) 
    or include milliseconds precision to 3 decimal places.

    Parameters:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas to draw upon.
        text (str or pd.Timestamp): The text (or datetime) to be printed.
        x (float): Horizontal reference (centre or left).
        y (float): Vertical centre position.
        font (str): Font name (default: Helvetica).
        colour (str): Text colour (default: black).
        size (int): Font size (default: 10).
        left_aligned (bool): If True, text is left-aligned; otherwise it is centred.
        date_only (bool): If True, only the date (dd/mm/yyyy) is shown for datetime inputs.
    """

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
        (630, 225.83, 197, 35),        # Breakout Torque
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

def build_program_specific_info(program_name, additional_info_indicies, additional_info_value, cleaned_data, black, light_blue):
    positions = []

    #------------------------------------------------------------------------------
    # Program = Initial Cycle
    #------------------------------------------------------------------------------

    if program_name == "Initial Cycle":
        pass

    #------------------------------------------------------------------------------
    # Program = Holds-Seat or Holds-Body
    #------------------------------------------------------------------------------  

    elif program_name == "Holds-Seat" or program_name == "Holds-Body":

        indices = additional_info_indicies.iloc[0]
        main_ch = additional_info_value.iloc[0]['Main Channel']
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

    #------------------------------------------------------------------------------
    # Program = Atmospheric Breakouts
    #------------------------------------------------------------------------------

    elif program_name == "Atmospheric Breakouts":
        
        for i in range(3):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    #------------------------------------------------------------------------------
    # Program = Atmospheric Cyclic
    #------------------------------------------------------------------------------

    elif program_name == "Atmospheric Cyclic":
        
        for i in range(len(additional_info_value.index)):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    #------------------------------------------------------------------------------
    # Program = Dynamic Cycles PR2
    #------------------------------------------------------------------------------

    elif program_name == "Dynamic Cycles PR2":

        for i in range(3):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    #------------------------------------------------------------------------------
    # Program = Dynamic Cycles Petrobras
    #------------------------------------------------------------------------------

    elif program_name == "Dynamic Cycles Petrobras":

        for i in range(3):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    #------------------------------------------------------------------------------
    # Program = Pulse Cycles
    #------------------------------------------------------------------------------

    elif program_name == "Pulse Cycles":

        for i in range(len(additional_info_value.index)):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )
    #------------------------------------------------------------------------------
    # Program = Signatures
    #------------------------------------------------------------------------------

    elif program_name == "Signatures":

        for i in range(len(additional_info_value.index)):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    #------------------------------------------------------------------------------
    # Program = Open-Close
    #------------------------------------------------------------------------------

    elif program_name == "Open-Close":
    
        for i in range(len(additional_info_value.index)):
            val = additional_info_value.iat[i, 1]
            if val not in ['', 0, 0.0]:
                positions.append(
                    (20, 56.5 - i*15, f"Breakout {i+1}", black, False)
                )
                positions.append(
                    (75, 56.5 - i*15, f"{val} ft.lbs", light_blue, False)
                )

    #------------------------------------------------------------------------------
    # Program = Number of Turns
    #------------------------------------------------------------------------------

    elif program_name == "Number Of Turns":
        pass
    
    return positions

def draw_all_text(pdf, pdf_text_positions):
    for x, y, text, colour, replace_empty in pdf_text_positions:
        draw_text_on_pdf(pdf, text, x, y, colour=colour, size=10, left_aligned=True, replace_empty=replace_empty)

def draw_test_details(test_metadata, transducer_details, active_channels, cleaned_data, pdf_output_path, additional_info_indicies, additional_info_value, program_name):
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
    pdf_text_positions += build_program_specific_info(program_name, additional_info_indicies, additional_info_value, cleaned_data, black, light_blue)
    draw_all_text(pdf, pdf_text_positions)
    return pdf