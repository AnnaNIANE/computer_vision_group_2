from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import time
import os

def generate_pdf_report(stats):
    output_path = f"output/report_{int(time.time())}.pdf"
    os.makedirs("output", exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Vehicle Tracking Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Summary", styles['Heading2']))
    table_data = [
        ['Metric', 'Value'],
        ['Frames Processed', stats['frames']],
        ['Total Detections', stats['detections']],
        ['Unique Tracks', stats['tracks']],
        ['Emergency Alerts', len(stats['alerts'])]
    ]
    for class_name, count in stats['class_counts'].items():
        if count > 0:
            table_data.append([f"{class_name.capitalize()} Detections", count])
    det_metrics = stats['detection_metrics']
    table_data.extend([
        ['Precision', f"{det_metrics['precision']:.3f}"],
        ['Recall', f"{det_metrics['recall']:.3f}"],
        ['mAP', f"{det_metrics['mAP']:.3f}"]
    ])
    track_metrics = stats['tracking_metrics']
    table_data.extend([
        ['MOTA', f"{track_metrics['MOTA']:.3f}"],
        ['ID Switches', track_metrics['ID_Switches']]
    ])
    table = Table(table_data)
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#d0d0d0'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ])
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Notes", styles['Heading2']))
    story.append(Paragraph("Emergency vehicles detected using color heuristics and audio siren detection. "
                           "Heatmap generated to show object position density. "
                           "Trajectory prediction added for future positions.", styles['Normal']))
    doc.build(story)
    return output_path