#!/usr/bin/env python3
"""Create the tax-style ATG TIR dataset and an auditable candidate table.

The dataset contains a 34 nt window (-25..+8 relative to AUG) for every
complete genomic ATG. Annotated CDS starts are label 1; all other candidate
starts are label 0. This mirrors tax/dataset.py and keeps per-candidate
coordinates so experimental locus tags can later be joined unambiguously.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from Bio import SeqIO


UPSTREAM, DOWNSTREAM = 25, 6
WINDOW = UPSTREAM + 3 + DOWNSTREAM
DNA = {"A": (1, 0, 0, 0), "T": (0, 1, 0, 0), "G": (0, 0, 1, 0), "C": (0, 0, 0, 1)}
RC = str.maketrans("ATGC", "TACG")


def attributes(field: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in field.split(";") if "=" in item)


def annotated_starts(gff: Path) -> dict[tuple[str, str, int], str]:
    """Return (contig, strand, zero-based start-anchor) -> locus tag."""
    result: dict[tuple[str, str, int], str] = {}
    with gff.open(encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "CDS" or fields[6] not in {"+", "-"}:
                continue
            attrs = attributes(fields[8])
            locus = attrs.get("locus_tag")
            if not locus:
                continue
            # GFF positions are 1-based inclusive; the model anchor is the
            # A of AUG on + or the exclusive right coordinate on -.
            anchor = int(fields[3]) - 1 if fields[6] == "+" else int(fields[4])
            result.setdefault((fields[0], fields[6], anchor), locus)
    return result


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC)[::-1]


def one_hot(sequence: str) -> np.ndarray:
    return np.asarray([DNA[base] for base in sequence], dtype=np.int8).T


def build_records(fasta: Path, gff: Path):
    positives = annotated_starts(gff)
    rows, encoded, labels = [], [], []
    for record in SeqIO.parse(fasta, "fasta"):
        contig = record.id
        sequence = str(record.seq).upper()
        for strand in ("+", "-"):
            query = "ATG" if strand == "+" else "CAT"
            for left in range(len(sequence) - 2):
                if sequence[left:left + 3] != query:
                    continue
                anchor = left if strand == "+" else left + 3
                low, high = anchor - UPSTREAM, anchor + 3 + DOWNSTREAM
                if low < 0 or high > len(sequence):
                    continue
                tir = sequence[low:high] if strand == "+" else reverse_complement(sequence[low:high])
                if len(tir) != WINDOW or any(base not in DNA for base in tir):
                    continue
                locus = positives.get((contig, strand, anchor), "")
                labels.append(int(bool(locus)))
                encoded.append(one_hot(tir))
                rows.append((len(rows), contig, strand, anchor, locus, tir, int(bool(locus))))
    return rows, encoded, labels


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=root / "reference/GCF_000022165.1/genomic.fna")
    parser.add_argument("--gff", type=Path, default=root / "reference/GCF_000022165.1/genomic.gff")
    parser.add_argument("--output-dir", type=Path, default=root / "reference/GCF_000022165.1")
    args = parser.parse_args()
    rows, encoded, labels = build_records(args.fasta, args.gff)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "data.pkl").open("wb") as handle:
        pickle.dump((encoded, labels), handle)
    with (args.output_dir / "candidates.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample_index", "contig", "strand", "start_anchor", "gene_id", "sequence", "label"))
        writer.writerows(rows)
    print(f"candidates={len(rows)}, labelled_starts={sum(labels)}, data={args.output_dir / 'data.pkl'}")


if __name__ == "__main__":
    main()
