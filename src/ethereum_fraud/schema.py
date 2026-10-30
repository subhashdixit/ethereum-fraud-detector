"""Shared input contract for training, API, form and CSV scoring."""

import math
import pandas as pd

# These fields are present in both source tables and the supplied unseen workbook.
FEATURES = {
    "Avg min between sent tnx": (
        "Average minutes between sent transactions",
        0,
        1000000000000000.0,
    ),
    "Avg min between received tnx": (
        "Average minutes between received transactions",
        0,
        1000000000000000.0,
    ),
    "Time Diff between first and last (Mins)": (
        "Observed account lifetime (minutes)",
        0,
        1000000000000000.0,
    ),
    "Sent tnx": ("Sent transaction count", 0, 1000000000000000.0),
    "Received Tnx": ("Received transaction count", 0, 1000000000000000.0),
    "Number of Created Contracts": ("Created contracts", 0, 1000000000000000.0),
    "Unique Received From Addresses": (
        "Unique incoming addresses",
        0,
        1000000000000000.0,
    ),
    "Unique Sent To Addresses": ("Unique outgoing addresses", 0, 1000000000000000.0),
    "min value received": ("Minimum received value", 0, 1000000000000000.0),
    "max value received ": ("Maximum received value", 0, 1000000000000000.0),
    "avg val received": ("Average received value", 0, 1000000000000000.0),
    "min val sent": ("Minimum sent value", 0, 1000000000000000.0),
    "max val sent": ("Maximum sent value", 0, 1000000000000000.0),
    "avg val sent": ("Average sent value", 0, 1000000000000000.0),
    "total Ether sent": ("Total ETH sent", 0, 1000000000000000.0),
    "total ether received": ("Total ETH received", 0, 1000000000000000.0),
    "total ether balance": ("Net ETH balance", -1000000000000000.0, 1000000000000000.0),
}
INTEGER_FIELDS = [
    "Sent tnx",
    "Received Tnx",
    "Number of Created Contracts",
    "Unique Received From Addresses",
    "Unique Sent To Addresses",
]
LABELS = {"0": "Non-fraud source label", "1": "Fraud source label"}


def validate_records(records):
    """Validate ordered numeric features; explicit nulls mean missing values."""
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("Provide a list of 1 to 500 records.")
    cleaned = []
    for row_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Row {row_number}: each record must be an object.")
        missing = set(FEATURES) - set(record)
        if missing:
            raise ValueError(
                f"Row {row_number}: missing columns: {', '.join(sorted(missing))}"
            )
        row = {}
        for field, (_, lower, upper) in FEATURES.items():
            value = record[field]
            if value is None or value == "":
                row[field] = float("nan")
                continue
            try:
                if isinstance(value, bool):
                    raise ValueError
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {row_number}: {field} must be numeric.") from exc
            if not math.isfinite(number) or not lower <= number <= upper:
                raise ValueError(
                    f"Row {row_number}: {field} must be between {lower} and {upper}."
                )
            if field in INTEGER_FIELDS and number != int(number):
                raise ValueError(f"Row {row_number}: {field} must be a whole number.")
            row[field] = number
        if all(math.isnan(value) for value in row.values()):
            raise ValueError(f"Row {row_number}: provide at least one observed value.")
        cleaned.append(row)
    return pd.DataFrame(cleaned, columns=list(FEATURES))
