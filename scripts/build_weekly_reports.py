"""Build S5-S7: uv run --locked --group reports python scripts/build_weekly_reports.py."""

import json

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs/reports"
WIDTH = A4[0] - 108
STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, leading=25, alignment=TA_CENTER, spaceAfter=22),
    "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=8, spaceAfter=10),
    "subheading": ParagraphStyle("subheading", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=13, spaceAfter=8),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=12.5, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10.5, spaceAfter=5),
    "compact": ParagraphStyle("compact", fontName="Helvetica", fontSize=8.5, leading=11),
}


def paragraph(text, style="body"):
    return Paragraph(escape(text).replace("\n", "<br/>"), STYLES[style])


def table(headers, rows, widths, compact=False):
    style = "compact" if compact else "cell"
    cells = [[Paragraph(f"<b>{escape(value)}</b>", STYLES[style]) for value in headers]]
    cells += [[paragraph(value, style) for value in row] for row in rows]
    result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5 if compact else 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if compact else 6),
    ]))
    return result


def page_frame(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawRightString(A4[0] - 54, A4[1] - 34, "SEMINARIO III - 2026-II")
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawCentredString(A4[0] / 2, 30, "Línea base - Proyecto Alzheimer con Vision Transformers")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 54, 30, str(document.page))
    canvas.restoreState()


def indicators(report):
    week = report["week"]
    return [
        paragraph(f"Reporte Semanal Semana {week}", "title"),
        paragraph("1. Indicadores de seguimiento", "heading"),
        table(("ID", "Indicador", f"Resultado S{week}", "Evidencia y seguimiento"),
              report["indicators"], [25, 91, 166, WIDTH - 282], compact=True),
        paragraph("1.1 Estado general", "subheading"),
        table(("Estado", "Situación de la semana", "Acción"),
              [report["status"]], [59, 215, WIDTH - 274], compact=True),
    ]


def build_report(report):
    week = report["week"]
    output = OUTPUT_DIR / f"Reporte_semanal_S{week}_final.pdf"
    document = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=54, rightMargin=54,
        topMargin=61, bottomMargin=49,
        title=f"Reporte Semanal Semana {week} - Proyecto Alzheimer ViT",
        author="Malak Sanchez y Juan Simancas",
        subject=f"Seguimiento académico S{week}",
    )
    story = indicators(report) + [PageBreak()]
    story += [paragraph("2. Reporte semanal", "heading"),
              table(("Campo", "Registro"), report["report"], [117, WIDTH - 117])]
    story += [PageBreak(), paragraph("3. Control de cambios", "heading"),
              table(("Campo", "Registro"), report["changes"], [117, WIDTH - 117])]
    document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = json.loads((ROOT / "configs/reports/weekly_reports.json").read_text())
    for report in reports:
        print(build_report(report))


if __name__ == "__main__":
    main()
