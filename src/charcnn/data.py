"""Data audit and leakage-safe chronological split creation.

Only `url`, `first_seen`, and `label` are ever read from DeepURLBench.  The
timestamp is renamed to `date` and is used solely for data splitting and the
online test order; it is never a model feature.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import yaml
from datasets import Dataset, load_dataset


class CoverageError(RuntimeError):
    """Raised when the requested instructor-specified temporal range is absent."""


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: date
    end: date

    @property
    def end_exclusive(self) -> date:
        return self.end + timedelta(days=1)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"dataset", "splits", "paths", "project"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"config.yaml missing required section(s): {sorted(missing)}")
    return config


def split_ranges(config: dict[str, Any]) -> list[SplitRange]:
    ranges = [
        SplitRange(name, date.fromisoformat(values["start"]), date.fromisoformat(values["end"]))
        for name, values in config["splits"].items()
    ]
    if [item.name for item in ranges] != ["train", "validation", "test"]:
        raise ValueError("Split names must be exactly train, validation, test in that order.")
    for previous, current in zip(ranges, ranges[1:]):
        if previous.end >= current.start:
            raise ValueError("Split ranges may not overlap.")
    return ranges


def _prepare_directories(config: dict[str, Any]) -> dict[str, Path]:
    paths = {name: Path(value) for name, value in config["paths"].items()}
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["month", "label", "row_count"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _query_dict(connection: duckdb.DuckDBPyConnection, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, parameters or [])
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _sql_string(value: str | Path) -> str:
    """Return a quoted DuckDB string literal for a local, controlled path."""
    return "'" + str(value).replace("'", "''") + "'"


def _load_selected_dataset(config: dict[str, Any], cache_dir: Path) -> Dataset:
    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["repository"],
        dataset_config["config"],
        split=dataset_config.get("split", "train"),
        cache_dir=str(cache_dir),
    )
    wanted = [
        dataset_config["url_column"],
        dataset_config["date_column"],
        dataset_config["label_column"],
    ]
    missing = set(wanted) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset schema does not contain required column(s): {sorted(missing)}")
    return dataset.select_columns(wanted)


def _export_source(dataset: Dataset, source_path: Path) -> None:
    """Persist just the three approved columns, in /kaggle/working.

    Hugging Face writes in Arrow batches, avoiding a full pandas materialization.
    """
    if source_path.exists():
        source_path.unlink()
    dataset.to_parquet(str(source_path), batch_size=100_000)


def _library_versions() -> dict[str, str]:
    import datasets
    import numpy
    import pandas
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "datasets": datasets.__version__,
        "duckdb": duckdb.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pa.__version__,
        "torch": torch.__version__,
    }


def audit_and_split(config_path: str | Path) -> dict[str, Any]:
    """Audit coverage, deduplicate globally, then write three fixed date splits.

    A duplicate URL is assigned to its earliest `first_seen` record before any
    split is made. This prevents the same URL from appearing in more than one
    split, including when the duplicate occurred months later.
    """
    config = load_config(config_path)
    ranges = split_ranges(config)
    paths = _prepare_directories(config)
    data_dir = paths["data_dir"]
    source_path = data_dir / "source_url_date_label.parquet"
    audit_path = data_dir / "audit.json"
    coverage_path = data_dir / "monthly_class_counts.csv"
    manifest_path = data_dir / "manifest.json"

    selected = _load_selected_dataset(config, paths["hf_cache_dir"])
    _export_source(selected, source_path)

    date_column = config["dataset"]["date_column"]
    url_column = config["dataset"]["url_column"]
    label_column = config["dataset"]["label_column"]
    range_start, range_end = ranges[0].start.isoformat(), ranges[-1].end_exclusive.isoformat()
    allowed_labels = list(config["dataset"].get("allowed_labels", ["benign", "phishing"]))
    if set(allowed_labels) != {"benign", "phishing"}:
        raise ValueError("This experiment must retain exactly benign and phishing labels.")

    connection = duckdb.connect(database=":memory:")
    connection.execute("PRAGMA threads=4")
    connection.execute(
        f'''CREATE VIEW raw AS
            SELECT
                CAST("{url_column}" AS VARCHAR) AS url,
                CAST("{date_column}" AS TIMESTAMP) AS date,
                CAST("{label_column}" AS VARCHAR) AS label
            FROM read_parquet({_sql_string(source_path)})''',
    )
    connection.execute(
        """CREATE VIEW clean AS
           SELECT url, date, label
           FROM raw
           WHERE url IS NOT NULL AND trim(url) <> ''
             AND date IS NOT NULL
             AND label IS NOT NULL AND trim(label) <> ''"""
    )

    overall = _query_dict(
        connection,
        "SELECT count(*) AS rows, min(date) AS min_date, max(date) AS max_date FROM clean",
    )[0]
    monthly_all = _query_dict(
        connection,
        """SELECT CAST(date_trunc('month', date) AS DATE) AS month, label, count(*) AS row_count
           FROM clean GROUP BY 1, 2 ORDER BY 1, 2""",
    )
    _write_csv(coverage_path, monthly_all)
    # Deduplicate the full source first. This makes a URL's earliest observed
    # label authoritative before malware is excluded from the binary task.
    connection.execute(
        """CREATE VIEW deduplicated AS
           SELECT url, date, label
           FROM (
               SELECT *, row_number() OVER (PARTITION BY url ORDER BY date ASC, label ASC) AS url_rank
               FROM clean
           ) WHERE url_rank = 1"""
    )
    connection.execute(
        "CREATE VIEW task_data AS SELECT url, date, label FROM deduplicated WHERE label IN ('benign', 'phishing')"
    )
    task_monthly = _query_dict(
        connection,
        """SELECT CAST(date_trunc('month', date) AS DATE) AS month, label, count(*) AS row_count
           FROM task_data GROUP BY 1, 2 ORDER BY 1, 2""",
    )
    _write_csv(data_dir / "binary_monthly_class_counts.csv", task_monthly)
    present_months = {str(row["month"])[:7] for row in task_monthly}
    missing_labels_by_split: dict[str, list[str]] = {}
    for item in ranges:
        rows = _query_dict(
            connection,
            """SELECT label FROM task_data
               WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)
               GROUP BY label""",
            [item.start.isoformat(), item.end_exclusive.isoformat()],
        )
        found = {row["label"] for row in rows}
        missing_labels_by_split[item.name] = sorted(set(allowed_labels) - found)
    expected_months: set[str] = set()
    for item in ranges:
        cursor = item.start.replace(day=1)
        final_month = item.end.replace(day=1)
        while cursor <= final_month:
            expected_months.add(cursor.isoformat()[:7])
            cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)

    coverage_ok = (
        overall["min_date"] is not None
        and str(overall["min_date"])[:10] <= ranges[0].start.isoformat()
        and str(overall["max_date"])[:10] >= ranges[-1].end.isoformat()
        and expected_months <= present_months
        and not any(missing_labels_by_split.values())
    )
    audit = {
        "dataset": config["dataset"],
        "allowed_columns": ["url", "date", "label"],
        "requested_splits": [
            {"name": item.name, "start": item.start.isoformat(), "end": item.end.isoformat()}
            for item in ranges
        ],
        "source": overall,
        "task_labels": allowed_labels,
        "excluded_labels": config["dataset"].get("excluded_labels", ["mal"]),
        "required_months": sorted(expected_months),
        "months_with_binary_rows": sorted(present_months),
        "missing_required_months": sorted(expected_months - present_months),
        "missing_binary_labels_by_split": missing_labels_by_split,
        "coverage_ok": coverage_ok,
        "monthly_class_counts_file": str(coverage_path),
        "note": "Date is used only for split boundaries and chronological ordering, never as a model input.",
    }
    _write_json(audit_path, audit)
    if not coverage_ok:
        required_window = f"{ranges[0].start.isoformat()} through {ranges[-1].end.isoformat()}"
        raise CoverageError(
            f"Dataset does not cover the required {required_window} period with both binary labels in every split. "
            f"Audit written to {audit_path}; no dates were adapted and no splits were created."
        )

    counts_before = _query_dict(
        connection,
        "SELECT count(*) AS rows FROM clean WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)",
        [range_start, range_end],
    )[0]["rows"]
    counts_after = _query_dict(
        connection,
        "SELECT count(*) AS rows FROM task_data WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)",
        [range_start, range_end],
    )[0]["rows"]

    split_summary: dict[str, Any] = {}
    for item in ranges:
        output = data_dir / f"{item.name}.parquet"
        start, end = item.start.isoformat(), item.end_exclusive.isoformat()
        # Deterministic date then URL ordering is preserved in the written Parquet file.
        connection.execute(
            f"""COPY (
                   SELECT url, date, label
                   FROM task_data
                   WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)
                   ORDER BY date ASC, url ASC
               ) TO {_sql_string(output)} (FORMAT PARQUET, COMPRESSION ZSTD)""",
            [start, end],
        )
        summary = _query_dict(
            connection,
            """SELECT count(*) AS rows, min(date) AS min_date, max(date) AS max_date
               FROM task_data WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)""",
            [start, end],
        )[0]
        class_counts = _query_dict(
            connection,
            """SELECT label, count(*) AS row_count FROM task_data
               WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)
               GROUP BY label ORDER BY label""",
            [start, end],
        )
        monthly_counts = _query_dict(
            connection,
            """SELECT CAST(date_trunc('month', date) AS DATE) AS month, label, count(*) AS row_count
               FROM task_data WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)
               GROUP BY 1, 2 ORDER BY 1, 2""",
            [start, end],
        )
        _write_csv(data_dir / f"{item.name}_monthly_class_counts.csv", monthly_counts)
        split_summary[item.name] = {
            **summary,
            "path": str(output),
            "class_counts": class_counts,
            "monthly_class_counts_file": str(data_dir / f"{item.name}_monthly_class_counts.csv"),
        }

    overlap = _query_dict(
        connection,
        """SELECT count(*) AS duplicate_urls_across_splits FROM (
             SELECT url FROM (
                SELECT url FROM read_parquet(%s) UNION ALL
                SELECT url FROM read_parquet(%s) UNION ALL
                SELECT url FROM read_parquet(%s)
             ) GROUP BY url HAVING count(*) > 1
           )""" % (
            _sql_string(data_dir / "train.parquet"),
            _sql_string(data_dir / "validation.parquet"),
            _sql_string(data_dir / "test.parquet"),
        ),
    )[0]["duplicate_urls_across_splits"]
    if overlap != 0:
        raise RuntimeError(f"Leakage check failed: {overlap} URLs occur in more than one split.")

    manifest = {
        "dataset": config["dataset"],
        "seed": config["project"]["seed"],
        "date_usage": "Split boundaries and chronological ordering only; never a model input.",
        "task_labels": allowed_labels,
        "label_map": config["labels"]["label_map"],
        "deduplication": "Global URL deduplication before binary-label filtering and splitting; retained the earliest first_seen record per URL.",
        "rows_in_requested_period_before_deduplication": counts_before,
        "rows_in_requested_period_after_deduplication": counts_after,
        "duplicate_urls_across_splits": overlap,
        "splits": split_summary,
        "library_versions": _library_versions(),
    }
    _write_json(data_dir / "label_map.json", config["labels"]["label_map"])
    _write_json(manifest_path, manifest)
    connection.close()

    # The raw three-column staging parquet and Hugging Face cache are useful
    # only while this one-time preparation run is active. Removing them keeps
    # the Kaggle output small enough to publish as a reusable prepared-data
    # Dataset. The three split parquet files and all audit artifacts remain.
    source_path.unlink(missing_ok=True)
    shutil.rmtree(paths["hf_cache_dir"], ignore_errors=True)
    return manifest
