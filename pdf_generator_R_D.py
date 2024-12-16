import io
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

def convert_fig_to_image_stream(fig):
    """Converts a matplotlib figure to a BytesIO stream."""
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', dpi = 1000)  # Save the figure with tight bounding box
    buf.seek(0)  # Reset the stream's position to the beginning
    return buf

def draw_rectangle(c, x, y, width, height):
    """Draws a rectangle with the given dimensions."""
    c.rect(x, y, width, height)

def add_text(c, text, x, y, font="Helvetica", size=12, color=None):
    """Adds text at the specified location, with specified colour. Defaults to black if no colour is specified."""
    c.setFont(font, size)
    c.setFillColor(color if color else colors.black)  # Set color to black if none is specified
    c.drawString(x, y, text)
    c.setFillColor(colors.black)  # Reset color to black for future text
    
def draw_centered_text(c, text, x, y, font="Helvetica", size=12):
    """Draws centered text with the specified font and size."""
    c.setFont(font, size)  # Set the font and size
    width = c.stringWidth(text, font, size)  # Get the width of the text
    c.drawCentredString(x, y, text)  # Draw centered text

def draw_grid_with_labels(c, x, y, width, height, rows, cols, headers, row_labels):
    """Draws a grid with headers and row labels at the specified location."""
    cell_width = width / cols
    cell_height = height / rows

    # Draw horizontal lines
    for i in range(rows + 1):
        c.line(x, y + i * cell_height, x + width, y + i * cell_height)
    
    # Draw vertical lines
    for j in range(cols + 1):
        c.line(x + j * cell_width, y, x + j * cell_width, y + height)
    
     # Draw headers (centered)
    for j, header in enumerate(headers):
        text_width = c.stringWidth(header, "Helvetica", 12)
        text_height = 12  # Approximate height of the text
        c.drawString(x + j * cell_width + (cell_width - text_width) / 2, y + (rows - 1) * cell_height + (cell_height - text_height) / 2, header)
    
    # Draw row labels (centered inside the first column cells)
    for i, row_label in enumerate(row_labels):
        text_width = c.stringWidth(row_label, "Helvetica", 12)
        c.drawString(x + (cell_width - text_width) / 2, y + (rows - i - 1) * cell_height + (cell_height - 12) / 2, row_label)

def create_pdf_with_figures(output_pdf_path, placeholders, figures, entries, results):
    c = canvas.Canvas(output_pdf_path, pagesize=landscape(A4))
    c.setStrokeColor(colors.black)

    # Define layout rectangles (x, y, width, height)
    layout_rectangles = [
        (15, 515, 600, 65),  # Info Top Left
        (15, 110, 600, 375),  # Graph
        (15, 15, 600, 95),  # Graph Index
        (630, 240, 197, 35), # Test Pressures
        (630, 15, 197, 200),  # 3rd Party Stamp
        (630, 300, 197, 185)  # Info Right
    ]

    for rect in layout_rectangles:
        draw_rectangle(c, *rect)

    light_blue = Color(0.325, 0.529, 0.761)
    black = Color(0, 0, 0)

    # Add title
    draw_centered_text(c, f"{entries['test_description'].get()} {placeholders['test_title']}", 315, 492.5, font="Helvetica-Bold", size=20)    
    add_text(c, "Data Recording Equipment Used", 635, 470, "Helvetica-Bold", 12)
    add_text(c, "3rd Party Stamp and Date", 645, 20, "Helvetica-Bold", 14)

    # Add placeholders with dynamic content
    text_positions = [
        (20, 565, f"Test Procedure Reference", black),
        (165, 565, f"{entries['test_procedure_reference'].get()}", light_blue),
        (20, 550, f"Unique No.", black),
        (165, 550, f"{entries['unique_no.'].get()}", light_blue),
        (20, 535, f"R&D Reference", black),
        (165, 535, f"{entries['r&d_reference'].get()}", light_blue),
        (20, 520, f"Valve Description", black),
        (165, 520, f"{entries['valve_description'].get()}", light_blue),
        (325, 565, f"Job No.", black),
        (455, 565, f"{entries['job_no.'].get()}", light_blue),
        (325, 550, f"Test Description", black),
        (455, 550, f"{entries['test_description'].get()}", light_blue),
        (325, 535, f"Test Date", black),
        (455, 535, f"{entries['test_date'].get()}", light_blue),
        (325, 520, f"Valve Drawing Number", black),
        (455, 520, f"{entries['valve_drawing_no.'].get()}", light_blue),
        (635, 260, f"Test Pressure", black),
        (745, 260, f"{entries['test_pressure'].get()} psi", light_blue),
        (635, 245, f"Actuator Pressure", black),
        (745, 245, f"{entries['actuator_pressure'].get()} psi", light_blue),
        (635, 455, f"Data Logger", black),
        (745, 455, f"{entries['data_logger'].get()}", light_blue),
        (635, 440, f"Serial Number", black),
        (745, 440, f"{entries['serial_number'].get()}", light_blue),
        (635, 425, f"Transducers", black),
        (745, 425, f"{entries['transducer_1'].get()}", light_blue),
        (745, 410, f"{entries['transducer_2'].get()}", light_blue),
        (745, 395, f"{entries['transducer_3'].get()}", light_blue),
        (745, 380, f"{entries['transducer_4'].get()}", light_blue),
        (635, 365, f"Gauges", black),
        (745, 365, f"{entries['gauge_1'].get()}", light_blue),
        (745, 350, f"{entries['gauge_2'].get()}", light_blue),
        (745, 335, f"{entries['gauge_3'].get()}", light_blue),
        (745, 320, f"{entries['gauge_4'].get()}", light_blue),
        (635, 305, f"Torque Transducer", black),
        (745, 305, f"{entries['torque_transducer'].get()}", light_blue)
    ]

    for x, y, text, color in text_positions:
        add_text(c, text, x, y, color=color)

    img_stream = convert_fig_to_image_stream(figures)  # Convert figure to BytesIO stream
    img = ImageReader(img_stream)  # Create an ImageReader object
    c.drawImage(img, 16, 111, 598, 373, preserveAspectRatio=False, mask='auto')  # Use ImageReader object (Graph)
    
    c.drawImage('V:/Userdoc/R & D/Logos/Address logo with address.jpg', 629, 514, 180, 67, preserveAspectRatio=True, mask='auto')

    num_of_cycles = len(results.iloc[:, 1].tolist())
    if num_of_cycles <= 8:
        text_size = 8
        bottom_padding = 2
    else:
        text_size = 10
        bottom_padding = 4

    data_list = [['Cycle'] + results.columns.values.tolist()] + results.reset_index().values.tolist()

    rows, cols = results.shape
    table = Table(data_list, 
                  colWidths=600/(cols+1) if num_of_cycles <= 8 else 812/(cols+1), 
                  rowHeights=(95/(rows+1)) if num_of_cycles <= 8 else 15)
    style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), text_size),
    ('BOTTOMPADDING', (0, 0), (-1, -1), bottom_padding),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    table.setStyle(style)

    c.save()
