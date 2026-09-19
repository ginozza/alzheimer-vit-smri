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


def indicators(week):
    rows = [
        ("I1", "Actividades cumplidas", "Terminadas / planificadas x 100", "Semanal", ">= 85 %"),
        ("I2", "Entregables aceptados", "Aceptados / 7 x 100", "Por hito", "Según cronograma"),
        ("I3", "Hitos en fecha", "En fecha / vencidos x 100", "Semanal", ">= 80 %"),
        ("I4", "Desviación de esfuerzo", "(Real - plan) / plan x 100", "Semanal", "-15 % a +20 %"),
        ("I5", "Riesgos altos abiertos", "Conteo con exposición >= 15", "Semanal", "Tendencia descendente"),
        ("I6", "Incidencias resueltas", "Cerradas / registradas x 100", "Semanal", ">= 80 % al cierre"),
        ("I7", "Reproducibilidad", "Corridas exitosas / verificaciones", "Por versión", "100 % final"),
        ("I8", "Rendimiento", "AUC, F1, sensibilidad, especificidad y accuracy", "Por modelo", "Superar referencia"),
        ("I9", "Portabilidad", "Tamaño, latencia y degradación", "S13", "Resultado documentado"),
        ("I10", "Cobertura documental", "Artefactos documentados / obligatorios", "Quincenal", "100 % final"),
    ]
    states = [
        ("Verde", "Desviación <= 10 % y sin bloqueo crítico.", "Continuar."),
        ("Amarillo", "Desviación de 11 % a 20 % o riesgo alto.", "Plan de recuperación."),
        ("Rojo", "Desviación > 20 % o hito fallido.", "Analizar impacto y tramitar cambio."),
    ]
    return [
        paragraph(f"Reporte Semanal Semana {week}", "title"),
        paragraph("1. Indicadores de seguimiento", "heading"),
        table(("ID", "Indicador", "Cálculo", "Frecuencia", "Meta"), rows, [27, 112, 158, 68, WIDTH - 365]),
        paragraph("1.1 Estado general", "subheading"),
        table(("Estado", "Condición", "Acción"), states, [82, 252, WIDTH - 334]),
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
    story = indicators(week) + [PageBreak()]
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
