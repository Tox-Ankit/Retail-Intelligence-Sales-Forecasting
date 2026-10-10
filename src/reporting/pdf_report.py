"""Generate downloadable PDF reports for business insights."""

from datetime import datetime
from io import BytesIO
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.insights.engine import InsightsReport


def _safe_paragraph_text(value: object) -> str:
    """Escape text before adding it to a ReportLab paragraph."""

    return escape(str(value)).replace("\n", "<br/>")


def _add_page_number(canvas, document) -> None:
    """Add a page number to the bottom of each PDF page."""

    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(
        A4[0] / 2,
        12 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def build_business_insights_pdf(report: InsightsReport) -> bytes:
    """Build a business insights PDF and return its bytes.

    The PDF is generated in memory. No report file is saved permanently.
    """

    if not isinstance(report, InsightsReport):
        raise TypeError("report must be an InsightsReport instance.")

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="Automated Business Insights Report",
        author="Retail Intelligence & Sales Forecasting",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=21,
        leading=27,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.grey,
        fontSize=10,
        leading=14,
        spaceAfter=18,
    )

    section_style = ParagraphStyle(
        "ReportSection",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    finding_title_style = ParagraphStyle(
        "FindingTitle",
        parent=styles["Heading3"],
        fontSize=11,
        leading=15,
        spaceBefore=5,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6,
        wordWrap="LTR",
    )

    evidence_style = ParagraphStyle(
        "ReportEvidence",
        parent=body_style,
        fontName="Courier",
        fontSize=8,
        leading=11,
        leftIndent=8,
    )

    story = []

    generated_at = datetime.now().astimezone().strftime(
        "%d %B %Y, %I:%M %p %Z"
    )

    story.append(
        Paragraph(
            "Automated Business Insights Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Retail Intelligence &amp; Sales Forecasting",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Dataset:</b> "
            f"{_safe_paragraph_text(report.dataset_name)}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{_safe_paragraph_text(generated_at)}",
            body_style,
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Executive Summary", section_style))

    summary_data = [
        [
            Paragraph("<b>Rows analyzed</b>", body_style),
            Paragraph("<b>Columns analyzed</b>", body_style),
            Paragraph("<b>Total findings</b>", body_style),
        ],
        [
            str(report.rows_analyzed),
            str(report.columns_analyzed),
            str(report.total_insights),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[55 * mm, 55 * mm, 55 * mm],
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#202A35")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Business Findings", section_style))

    if not report.insights:
        story.append(
            Paragraph(
                "No business findings were generated for this dataset.",
                body_style,
            )
        )

    for index, finding in enumerate(report.insights, start=1):
        severity = str(finding.severity).upper()

        finding_heading = Paragraph(
            f"{index}. {_safe_paragraph_text(finding.title)}",
            finding_title_style,
        )

        finding_details = [
            Paragraph(
                f"<b>Category:</b> "
                f"{_safe_paragraph_text(finding.category)}",
                body_style,
            ),
            Paragraph(
                f"<b>Severity:</b> { _safe_paragraph_text(severity) }",
                body_style,
            ),
            Paragraph(
                _safe_paragraph_text(finding.description),
                body_style,
            ),
        ]

        if finding.evidence:
            evidence_json = json.dumps(
                finding.evidence,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

            finding_details.extend(
                [
                    Paragraph("<b>Supporting evidence:</b>", body_style),
                    Paragraph(
                        _safe_paragraph_text(evidence_json),
                        evidence_style,
                    ),
                ]
            )

        if finding.recommendation:
            finding_details.append(
                Paragraph(
                    "<b>Suggested next step:</b> "
                    f"{_safe_paragraph_text(finding.recommendation)}",
                    body_style,
                )
            )

        story.append(
            KeepTogether(
                [
                    finding_heading,
                    *finding_details,
                    HRFlowable(
                        width="100%",
                        thickness=0.5,
                        color=colors.lightgrey,
                        spaceBefore=3,
                        spaceAfter=8,
                    ),
                ]
            )
        )

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "This report contains findings generated from the selected "
            "dataset. Review the evidence and business context before "
            "acting on any recommendation.",
            styles["Italic"],
        )
    )

    document.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes