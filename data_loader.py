"""Load and validate JSONL data. Prepare text for embedding."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QoSMetrics:
    rt_ms: Optional[float] = None
    tp_rps: Optional[float] = None
    availability: Optional[float] = None
    valid_qos: bool = False


@dataclass
class APIRecord:
    api_id: str
    category: str
    name: str
    description: str
    method: str
    url: str
    file: str
    tool: str
    qos: Optional[QoSMetrics] = None
    embedding_text: str = ""


def load_jsonl(filepath: Path) -> list[dict]:
    """Read a JSONL file into a list of raw dicts."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            "Please place the JSONL files in the data/ directory."
        )
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_num} in {filepath.name}: {e}"
                )
    return records


def _build_embedding_text(record: APIRecord) -> str:
    """Build text for embedding. Uses fallback for empty descriptions."""
    desc = record.description.strip()
    if desc:
        return f"[{record.category}] {record.name}: {desc}"
    return f"[{record.category}] {record.name} - {record.method} API endpoint"


def _parse_qos(raw: dict) -> QoSMetrics:
    """Parse the qos sub-dict, handling nulls gracefully."""
    qos_data = raw.get("qos", {})
    if not qos_data or not isinstance(qos_data, dict):
        return QoSMetrics()
    return QoSMetrics(
        rt_ms=qos_data.get("rt_ms"),
        tp_rps=qos_data.get("tp_rps"),
        availability=qos_data.get("availability"),
        valid_qos=bool(qos_data.get("valid_qos", False)),
    )


def parse_records(raw_list: list[dict], has_qos: bool = False) -> list[APIRecord]:
    """Convert raw dicts into APIRecord dataclasses."""
    records = []
    for raw in raw_list:
        rec = APIRecord(
            api_id=raw.get("api_id", ""),
            category=raw.get("category", "Unknown"),
            name=raw.get("name", "Unnamed API"),
            description=raw.get("description", ""),
            method=raw.get("method", "GET").upper(),
            url=raw.get("url", ""),
            file=raw.get("_file", ""),
            tool=raw.get("_tool", ""),
        )
        if has_qos:
            rec.qos = _parse_qos(raw)
        rec.embedding_text = _build_embedding_text(rec)
        records.append(rec)
    return records


def get_categories(records: list[APIRecord]) -> list[str]:
    """Return sorted list of unique category names."""
    return sorted({r.category for r in records})


def load_all_data() -> tuple[list[APIRecord], list[APIRecord]]:
    """Load both JSONL files and return (no_qos_records, with_qos_records)."""
    from config import NO_QOS_PATH, WITH_QOS_PATH

    raw_no_qos = load_jsonl(NO_QOS_PATH)
    raw_with_qos = load_jsonl(WITH_QOS_PATH)

    no_qos_records = parse_records(raw_no_qos, has_qos=False)
    with_qos_records = parse_records(raw_with_qos, has_qos=True)

    return no_qos_records, with_qos_records


def build_record_maps(
    no_qos_records: list[APIRecord], with_qos_records: list[APIRecord]
) -> tuple[dict[str, APIRecord], dict[str, APIRecord]]:
    """Build api_id -> APIRecord lookup dicts for both datasets."""
    no_qos_map = {r.api_id: r for r in no_qos_records}
    qos_map = {r.api_id: r for r in with_qos_records}
    return no_qos_map, qos_map
