from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CleaningAction:
    """
    Records one transformation performed during cleaning.
    """

    action_type: str
    description: str
    column_name: str | None = None
    affected_rows: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class CleaningResult:
    """
    Stores the cleaned dataset and complete cleaning audit trail.
    """

    dataset_name: str
    original_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    actions: list[CleaningAction] = field(default_factory=list)

    def add_action(self, action: CleaningAction) -> None:
        """Record a cleaning action."""
        self.actions.append(action)

    @property
    def total_actions(self) -> int:
        """Return number of transformations performed."""
        return len(self.actions)

    @property
    def original_shape(self) -> tuple[int, int]:
        """Return original dataset shape."""
        return self.original_data.shape

    @property
    def cleaned_shape(self) -> tuple[int, int]:
        """Return cleaned dataset shape."""
        return self.cleaned_data.shape

    @property
    def rows_removed(self) -> int:
        """Return number of rows removed during cleaning."""
        return len(self.original_data) - len(self.cleaned_data)