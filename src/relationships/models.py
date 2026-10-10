from dataclasses import dataclass, field
from typing import Any


@dataclass
class RelationshipCandidate:
    """
    Represents a possible relationship between two columns
    belonging to two different datasets.
    """

    left_dataset: str
    left_column: str

    right_dataset: str
    right_column: str

    relationship_type: str

    confidence: float

    evidence: list[str] = field(default_factory=list)

    left_unique: bool = False
    right_unique: bool = False

    left_non_null_count: int = 0
    right_non_null_count: int = 0

    value_overlap_count: int = 0
    value_overlap_percentage: float = 0.0

    status: str = "pending"

    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Return True when confidence is at least 80%."""
        return self.confidence >= 80.0

    def accept(self) -> None:
        """Mark the relationship as accepted."""
        self.status = "accepted"

    def reject(self) -> None:
        """Mark the relationship as rejected."""
        self.status = "rejected"


@dataclass
class RelationshipResult:
    """
    Stores all relationship candidates detected across
    a collection of datasets.
    """

    candidates: list[RelationshipCandidate] = field(
        default_factory=list
    )

    def add_candidate(
        self,
        candidate: RelationshipCandidate,
    ) -> None:
        """Add a relationship candidate."""
        self.candidates.append(candidate)

    @property
    def total_candidates(self) -> int:
        """Return total number of detected candidates."""
        return len(self.candidates)

    @property
    def high_confidence_candidates(
        self,
    ) -> list[RelationshipCandidate]:
        """Return candidates with confidence >= 80%."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.is_high_confidence
        ]

    @property
    def pending_candidates(
        self,
    ) -> list[RelationshipCandidate]:
        """Return candidates awaiting user decision."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.status == "pending"
        ]

    @property
    def accepted_candidates(
        self,
    ) -> list[RelationshipCandidate]:
        """Return accepted relationships."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.status == "accepted"
        ]

    @property
    def rejected_candidates(
        self,
    ) -> list[RelationshipCandidate]:
        """Return rejected relationships."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.status == "rejected"
        ]

