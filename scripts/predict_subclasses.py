#!/usr/bin/env python3
"""Store models 5-8 subclasses and JSON-defined 3-of-4 consensus in SQLite."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import rna_encoding


MODELS = ("5", "6", "7", "8")
RUN_ID = "5-6-7-8"
MEMBER = re.compile(r"Model (?P<model>\d+)'s class (?P<class>\d+)$")


def q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS sequences (
        id INTEGER PRIMARY KEY, species TEXT NOT NULL, sample_index INTEGER NOT NULL,
        contig TEXT NOT NULL, strand TEXT NOT NULL, start_anchor INTEGER NOT NULL,
        gene_id TEXT, sequence TEXT NOT NULL, label INTEGER NOT NULL,
        UNIQUE(species, sample_index))""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sequences)")}
    for column in (*MODELS, RUN_ID):
        if column not in columns:
            connection.execute(f"ALTER TABLE sequences ADD COLUMN {q(column)} INTEGER")


def import_candidates(connection: sqlite3.Connection, path: Path, species: str) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    connection.executemany("""INSERT INTO sequences
        (species, sample_index, contig, strand, start_anchor, gene_id, sequence, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(species, sample_index) DO UPDATE SET
          contig=excluded.contig, strand=excluded.strand, start_anchor=excluded.start_anchor,
          gene_id=excluded.gene_id, sequence=excluded.sequence, label=excluded.label""", [
        (species, int(row["sample_index"]), row["contig"], row["strand"],
         int(row["start_anchor"]), row["gene_id"] or None, row["sequence"], int(row["label"]))
        for row in rows
    ])
    return len(rows)


def load_model(path: Path, device: torch.device):
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    return model.eval().to(device)


def predict(connection: sqlite3.Connection, species: str, model_id: str, model, device: torch.device, batch_size: int) -> Counter:
    rows = connection.execute("SELECT id, sequence FROM sequences WHERE species=? AND label=1 ORDER BY id", (species,)).fetchall()
    connection.execute(f"UPDATE sequences SET {q(model_id)}=NULL WHERE species=?", (species,))
    counts: Counter = Counter()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            values = np.stack([rna_encoding(sequence) for _, sequence in batch])
            inputs = torch.tensor(values, dtype=torch.float32, device=device)
            _, calls = model.project_subclass(inputs, class_id=1)
            calls = [int(call) for call in calls.cpu().tolist()]
            counts.update(calls)
            connection.executemany(f"UPDATE sequences SET {q(model_id)}=? WHERE id=?", zip(calls, (row_id for row_id, _ in batch)))
    return counts


def consensus_mapping(path: Path) -> tuple[int, dict[tuple[str, int], int]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["models"]) != MODELS:
        raise ValueError(f"Unexpected models in {path}: {config['models']}")
    mapping = {}
    for name, members in config["consensus_clusters"].items():
        cluster = int(name.removeprefix("consensus "))
        for member in members:
            match = MEMBER.fullmatch(member)
            if not match:
                raise ValueError(f"Unparseable consensus member: {member}")
            mapping[(match["model"], int(match["class"]))] = cluster
    return int(config["n_nice"]), mapping


def store_consensus(connection: sqlite3.Connection, species: str, n_nice: int, mapping: dict[tuple[str, int], int]) -> Counter:
    rows = connection.execute(f"SELECT id, {', '.join(q(model) for model in MODELS)} FROM sequences WHERE species=? AND label=1", (species,)).fetchall()
    calls = []
    for row in rows:
        id_, subclasses = row[0], row[1:]
        clusters = [mapping[(model, int(subclass))] for model, subclass in zip(MODELS, subclasses) if subclass is not None]
        frequencies = Counter(clusters)
        top = frequencies.most_common(2)
        value = None if not top or top[0][1] < n_nice or (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
        calls.append((value, id_))
    connection.execute(f"UPDATE sequences SET {q(RUN_ID)}=NULL WHERE species=?", (species,))
    connection.executemany(f"UPDATE sequences SET {q(RUN_ID)}=? WHERE id=?", calls)
    return Counter(value if value is not None else "unassigned" for value, _ in calls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "reference/GCF_000022165.1/candidates.tsv")
    parser.add_argument("--database", type=Path, default=ROOT / "data.db")
    parser.add_argument("--consensus", type=Path, default=ROOT / "models/5-6-7-8.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    species = "GCF_000022165.1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with sqlite3.connect(args.database) as connection:
        ensure_schema(connection)
        total = import_candidates(connection, args.candidates, species)
        for model_id in MODELS:
            counts = predict(connection, species, model_id, load_model(ROOT / f"models/{model_id}/0.pth", device), device, args.batch_size)
            print(f"model={model_id}, subclasses={dict(sorted(counts.items()))}")
        n_nice, mapping = consensus_mapping(args.consensus)
        counts = store_consensus(connection, species, n_nice, mapping)
        connection.commit()
    print(f"device={device}, candidates={total}, consensus_n_nice={n_nice}, consensus={dict(sorted(counts.items(), key=str))}")


if __name__ == "__main__":
    main()
