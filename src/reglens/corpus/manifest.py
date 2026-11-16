"""The corpus manifest: one row per document RegLens indexes.

``data/sources.csv`` is the single source of truth. The downloader, the ingestion step
and the citation cards in the UI all read it, so every answer can be traced back to a
publisher URL, a version or date, and a legal status (in force, draft, proposed, ...).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = (
    "id",
    "title",
    "publisher",
    "jurisdiction",
    "doc_type",
    "status",
    "version_date",
    "priority",
    "fetch",
    "format",
    "url",
    "landing_url",
    "filename",
    "verification",
    "notes",
)
# Columns that must not be blank ("notes" may be).
NON_EMPTY_COLUMNS = tuple(c for c in REQUIRED_COLUMNS if c != "notes")

STATUSES = ("in_force", "final", "draft", "consultation", "proposed")
FETCH_MODES = ("auto", "manual")
FORMATS = ("pdf", "html")
JURISDICTIONS = ("EU", "US", "UK", "International")
DOC_TYPES = (
    "regulation",
    "statute",
    "guidelines",
    "opinion",
    "framework",
    "recommendation",
    "treaty",
    "summary",
    "web_guidance",
    "policy_statement",
)
PRIORITIES = (1, 2)  # 1 = core document, 2 = optional

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ManifestError(ValueError):
    """The manifest is missing, malformed or inconsistent. The message lists every problem."""


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    publisher: str
    jurisdiction: str
    doc_type: str
    status: str
    version_date: str
    priority: int
    fetch: str
    format: str
    url: str
    landing_url: str
    filename: str
    verification: str
    notes: str = ""

    @property
    def is_core(self) -> bool:
        return self.priority == 1

    @property
    def is_manual(self) -> bool:
        return self.fetch == "manual"


def load_manifest(path: str | Path) -> list[Source]:
    """Load and validate the manifest. Raises ``ManifestError`` listing all problems found."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")

    # utf-8-sig: tolerate a byte-order mark if the file was re-saved by Excel or Notepad.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing_columns:
            raise ManifestError(f"{path.name}: missing column(s): {', '.join(missing_columns)}")
        rows = list(reader)

    problems: list[str] = []
    sources: list[Source] = []
    seen_ids: dict[str, int] = {}
    seen_urls: dict[str, str] = {}
    seen_files: dict[str, str] = {}

    for line_no, row in enumerate(rows, start=2):  # line 1 is the header
        sid = (row.get("id") or "").strip()
        where = f"line {line_no} ({sid or 'no id'})"

        if None in row:
            problems.append(f"{where}: too many fields - is there an unquoted comma?")
            continue
        missing_fields = [c for c in REQUIRED_COLUMNS if row.get(c) is None]
        if missing_fields:
            problems.append(f"{where}: too few fields (missing {', '.join(missing_fields)}) - check quoting")
            continue

        values = {c: (row[c] or "").strip() for c in REQUIRED_COLUMNS}
        row_problems: list[str] = []

        for column in NON_EMPTY_COLUMNS:
            if not values[column]:
                row_problems.append(f"{column} is empty")

        if values["id"] and not _ID_RE.match(values["id"]):
            row_problems.append("id must be lowercase letters, digits and single hyphens")
        for column, allowed in (
            ("status", STATUSES),
            ("fetch", FETCH_MODES),
            ("format", FORMATS),
            ("jurisdiction", JURISDICTIONS),
            ("doc_type", DOC_TYPES),
        ):
            if values[column] and values[column] not in allowed:
                row_problems.append(f"{column}={values[column]!r} is not one of {', '.join(allowed)}")

        priority = 0
        if values["priority"]:
            if values["priority"].isdigit() and int(values["priority"]) in PRIORITIES:
                priority = int(values["priority"])
            else:
                row_problems.append(f"priority={values['priority']!r} must be 1 (core) or 2 (optional)")

        for column in ("url", "landing_url"):
            if values[column] and not values[column].startswith("https://"):
                row_problems.append(f"{column} must start with https://")

        expected_name = f"{values['id']}.{values['format']}"
        if values["id"] and values["format"] in FORMATS and values["filename"] != expected_name:
            row_problems.append(f"filename must be {expected_name!r} (id + extension), not {values['filename']!r}")

        if values["id"] in seen_ids:
            row_problems.append(f"duplicate id (first used on line {seen_ids[values['id']]})")
        elif values["id"]:
            seen_ids[values["id"]] = line_no
        if values["url"] in seen_urls:
            row_problems.append(f"url already used by {seen_urls[values['url']]}")
        elif values["url"]:
            seen_urls[values["url"]] = values["id"] or where
        if values["filename"] in seen_files:
            row_problems.append(f"filename already used by {seen_files[values['filename']]}")
        elif values["filename"]:
            seen_files[values["filename"]] = values["id"] or where

        if row_problems:
            problems.extend(f"{where}: {p}" for p in row_problems)
            continue

        sources.append(Source(**{**values, "priority": priority}))

    if problems:
        raise ManifestError(f"{path.name} has {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
    if not sources:
        raise ManifestError(f"{path.name} contains no documents")
    return sources
