
"""Dataset preparation utilities for machine learning."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SUPPORTED_TASKS = {"classification", "regression"}


def _validate_dataframe(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The dataset is empty.")

    if data.columns.has_duplicates:
        raise ValueError("Duplicate column names are not supported.")

    if len(data.columns) < 2:
        raise ValueError(
            "The dataset must contain a target column and at least one feature."
        )


def suggest_task_type(target):
    """Suggest classification or regression from a target Series.

    This is a heuristic, not a definitive decision. The user should
    confirm the task based on the meaning of the target.
    """
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series.")

    valid = target.dropna()

    if valid.empty:
        raise ValueError("The target contains no valid values.")

    if pd.api.types.is_bool_dtype(valid.dtype):
        return "classification"

    if (
        pd.api.types.is_object_dtype(valid.dtype)
        or isinstance(valid.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(valid.dtype)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(valid.dtype):
        unique_count = valid.nunique()

        # Small integer-like targets may represent class labels.
        # This is only a suggestion; numeric values can also be categories.
        if (
            unique_count <= 10
            and (valid % 1 == 0).all()
        ):
            return "classification"

        return "regression"

    raise ValueError(
        "Unable to suggest a task type for this target. "
        "Convert it to a suitable numeric or categorical column first."
    )


def validate_target(data, target_column):
    """Validate a selected target and return its feature and target data."""
    _validate_dataframe(data)

    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a non-empty column name.")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    y = data[target_column].copy()

    if y.isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values. "
            "Resolve them explicitly before training."
        )

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    X = data.drop(columns=[target_column]).copy()

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    return X, y


def split_dataset(
    data,
    target_column,
    task_type,
    test_size=0.2,
    random_state=42,
):
    """Split a dataset into training and test sets.

    Classification uses stratification when there are enough examples
    in each class and enough rows for both partitions.
    """
    _validate_dataframe(data)

    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be between 0 and 1.")

    X, y = validate_target(data, target_column)

    if len(data) < 5:
        raise ValueError(
            "At least 5 rows are required for a train/test split."
        )

    stratify = None

    if task_type == "classification":
        class_counts = y.value_counts()
        number_of_classes = len(class_counts)

        test_rows = max(1, int(__import__("math").ceil(len(data) * test_size)))
        train_rows = len(data) - test_rows

        if class_counts.min() < 2:
            raise ValueError(
                "Every class needs at least two rows for a reliable "
                "stratified split. Collect more examples or review "
                "the target labels."
            )

        if test_rows < number_of_classes or train_rows < number_of_classes:
            raise ValueError(
                "The training and test partitions must each have enough "
                "rows to represent every class. Adjust test_size or "
                "provide more data."
            )

        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(X):
    """Build a preprocessing transformer for feature columns.

    Numeric columns: median imputation and standard scaling.
    Categorical columns: most-frequent imputation and one-hot encoding.

    Fit this transformer on X_train only, preferably inside a model
    Pipeline. Do not fit it on the complete dataset before splitting.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain rows and feature columns.")

    if X.columns.has_duplicates:
        raise ValueError("Duplicate feature column names are not supported.")

    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    if datetime_columns:
        raise ValueError(
            "Datetime columns must be transformed into date features first. "
            f"Unprocessed datetime columns: {datetime_columns}"
        )

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
