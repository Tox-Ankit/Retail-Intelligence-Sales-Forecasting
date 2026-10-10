from src.relationships.models import (
    RelationshipCandidate,
    RelationshipResult,
)


class RelationshipManager:
    """
    Manages detected relationship candidates and their
    user-approved status.

    The manager does not automatically accept relationships.
    """

    def __init__(self, result: RelationshipResult):
        if not isinstance(result, RelationshipResult):
            raise TypeError(
                "result must be a RelationshipResult."
            )

        self.result = result

    def get_pending(self) -> list[RelationshipCandidate]:
        """Return relationships waiting for user decision."""
        return self.result.pending_candidates

    def get_accepted(self) -> list[RelationshipCandidate]:
        """Return accepted relationships."""
        return self.result.accepted_candidates

    def get_rejected(self) -> list[RelationshipCandidate]:
        """Return rejected relationships."""
        return self.result.rejected_candidates

    def _get_candidate(
        self,
        index: int,
    ) -> RelationshipCandidate:
        """Return a candidate by index."""

        if not isinstance(index, int):
            raise TypeError("index must be an integer.")

        if index < 0 or index >= len(self.result.candidates):
            raise IndexError(
                "Relationship candidate index is out of range."
            )

        return self.result.candidates[index]

    def accept(self, index: int) -> RelationshipCandidate:
        """Accept a relationship candidate."""

        candidate = self._get_candidate(index)

        if candidate.status == "rejected":
            raise ValueError(
                "A rejected relationship cannot be accepted "
                "without first resetting its status."
            )

        candidate.accept()

        return candidate

    def reject(self, index: int) -> RelationshipCandidate:
        """Reject a relationship candidate."""

        candidate = self._get_candidate(index)

        if candidate.status == "accepted":
            raise ValueError(
                "An accepted relationship cannot be rejected "
                "without first resetting its status."
            )

        candidate.reject()

        return candidate

    def modify(
        self,
        index: int,
        left_dataset: str | None = None,
        left_column: str | None = None,
        right_dataset: str | None = None,
        right_column: str | None = None,
        relationship_type: str | None = None,
    ) -> RelationshipCandidate:
        """
        Modify a relationship candidate.

        Modified relationships are returned to pending status
        so the user must explicitly approve them again.
        """

        candidate = self._get_candidate(index)

        if left_dataset is not None:
            candidate.left_dataset = left_dataset

        if left_column is not None:
            candidate.left_column = left_column

        if right_dataset is not None:
            candidate.right_dataset = right_dataset

        if right_column is not None:
            candidate.right_column = right_column

        if relationship_type is not None:
            candidate.relationship_type = relationship_type

        candidate.status = "pending"

        return candidate

    def reset(self, index: int) -> RelationshipCandidate:
        """Return a candidate to pending status."""

        candidate = self._get_candidate(index)

        candidate.status = "pending"

        return candidate
    