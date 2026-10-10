from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityFinding:
    """
    Represents one data-quality issue or observation.
    """

    issue_type: str
    severity: str
    message: str
    column_name: str | None = None
    affected_rows: int = 0
    affected_percentage: float = 0.0
    suggested_action: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """
    Complete data-quality assessment for one dataset.
    """

    dataset_name: str
    findings: list[QualityFinding] = field(default_factory=list)

    @property
    def total_findings(self) -> int:
        """Return total number of quality findings."""
        return len(self.findings)

    @property
    def critical_findings(self) -> list[QualityFinding]:
        """Return critical findings."""
        return [
            finding
            for finding in self.findings
            if finding.severity == "critical"
        ]

    @property
    def warning_findings(self) -> list[QualityFinding]:
        """Return warning findings."""
        return [
            finding
            for finding in self.findings
            if finding.severity == "warning"
        ]

    @property
    def info_findings(self) -> list[QualityFinding]:
        """Return informational findings."""
        return [
            finding
            for finding in self.findings
            if finding.severity == "info"
        ]

    def add_finding(self, finding: QualityFinding) -> None:
        """Add a quality finding to the report."""
        self.findings.append(finding)