from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class TrainingHistory:
    def __init__(self, directory: Path, stem: str = "history") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.directory / f"{stem}.jsonl"
        self.csv_path = self.directory / f"{stem}.csv"
        self._fieldnames: list[str] | None = None

    def append(self, record: dict[str, Any]) -> None:
        flat = self._flatten(record)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        # CSV schemas are stable for almost all training epochs. Cache the header
        # so Drive/FUSE does not reread the entire history on every append. Only
        # read/rewrite old rows when a genuinely new metric field appears.
        if self._fieldnames is None:
            if self.csv_path.is_file():
                with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    self._fieldnames = next(reader, [])
            else:
                self._fieldnames = []
        fieldnames = list(self._fieldnames)
        updated_fields = list(dict.fromkeys(fieldnames + list(flat.keys())))
        if updated_fields != fieldnames:
            existing_rows: list[dict[str, str]] = []
            if self.csv_path.is_file():
                with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                    existing_rows = list(csv.DictReader(handle))
            temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=updated_fields)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(flat)
            temporary.replace(self.csv_path)
            self._fieldnames = updated_fields
        else:
            mode = "a" if self.csv_path.is_file() else "w"
            with self.csv_path.open(mode, newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=updated_fields)
                if mode == "w":
                    writer.writeheader()
                writer.writerow(flat)
            self._fieldnames = updated_fields


    def truncate_from(self, field: str, minimum: int | float) -> int:
        """Remove records whose numeric ``field`` is >= ``minimum``.

        Checkpoints can lag history when checkpoint cadence is greater than one.
        On resume, metrics from epochs newer than the restored checkpoint are no
        longer part of the resumed trajectory and must be removed before they
        are recomputed.
        """
        if not self.jsonl_path.is_file():
            return 0
        records: list[dict[str, Any]] = []
        removed = 0
        raw_lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        nonempty_positions = [i for i, line in enumerate(raw_lines) if line.strip()]
        last_nonempty = nonempty_positions[-1] if nonempty_positions else -1
        repaired_tail = False
        for position, line in enumerate(raw_lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if position != last_nonempty:
                    raise RuntimeError(
                        f"Training history is corrupted before its final record: {self.jsonl_path} line {position + 1}"
                    )
                # Abrupt Colab termination can cut only the append-in-progress tail.
                # Drop that uncommitted record; the checkpoint determines where
                # training resumes and the epoch will be recomputed if necessary.
                repaired_tail = True
                removed += 1
                break
            value = record.get(field)
            if value is None:
                records.append(record)
                continue
            should_remove = float(value) >= float(minimum)
            if should_remove:
                removed += 1
            else:
                records.append(record)
        if removed or repaired_tail:
            self._rewrite(records)
        return removed

    def _rewrite(self, records: list[dict[str, Any]]) -> None:
        tmp_jsonl = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".tmp")
        with tmp_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        tmp_jsonl.replace(self.jsonl_path)

        flattened = [self._flatten(record) for record in records]
        fieldnames: list[str] = []
        for row in flattened:
            fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))
        tmp_csv = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened)
        tmp_csv.replace(self.csv_path)
        self._fieldnames = fieldnames

    @staticmethod
    def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in record.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                output.update(TrainingHistory._flatten(value, name))
            elif isinstance(value, (list, tuple)):
                output[name] = json.dumps(value, default=float)
            else:
                output[name] = value
        return output

