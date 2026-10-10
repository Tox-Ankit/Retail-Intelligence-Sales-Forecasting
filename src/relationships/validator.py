from src.relationships.models import (
    RelationshipCandidate,
    RelationshipResult,
)


def validate_relationship(
    candidate: RelationshipCandidate,
    minimum_overlap: float = 20.0,
    minimum_confidence: float = 60.0,
) -> bool:
    """
    Validate whether a relationship candidate is strong enough
    to be presented to the user.

    The relationship is not automatically accepted.
    This only determines whether the candidate is credible enough
    to display.
    """

    if not isinstance(candidate, RelationshipCandidate):
        raise TypeError(
            "candidate must be a RelationshipCandidate."
        )

    if not 0 <= minimum_overlap <= 100:
        raise ValueError(
            "minimum_overlap must be between 0 and 100."
        )

    if not 0 <= minimum_confidence <= 100:
        raise ValueError(
            "minimum_confidence must be between 0 and 100."
        )

    # Confidence must be sufficiently strong.
    if candidate.confidence < minimum_confidence:
        return False

    # Some actual value overlap should exist.
    if candidate.value_overlap_percentage < minimum_overlap:
        return False

    # A relationship should normally have at least one
    # key-like column.
    if not candidate.left_unique and not candidate.right_unique:
        return False

    return True


def validate_relationships(
    result: RelationshipResult,
    minimum_overlap: float = 20.0,
    minimum_confidence: float = 60.0,
) -> RelationshipResult:
    """
    Filter relationship candidates using validation rules.

    Returns a new RelationshipResult containing only candidates
    that pass validation.
    """

    if not isinstance(result, RelationshipResult):
        raise TypeError(
            "result must be a RelationshipResult."
        )

    validated_result = RelationshipResult()

    for candidate in result.candidates:

        if validate_relationship(
            candidate,
            minimum_overlap=minimum_overlap,
            minimum_confidence=minimum_confidence,
        ):
            validated_result.add_candidate(candidate)

    validated_result.candidates.sort(
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )

    return validated_result