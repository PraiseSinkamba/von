"""Audit a training corpus for shingle overlap against public benchmark test sets.

Von submits to three external leaderboards (jabr/classifier-benchmark, JevBench,
the Decision Index). Training on their test items would invalidate any score
before it is even measured. This module builds an 8-gram shingle index from
each benchmark's public state/option/criteria text and flags any corpus record
that shares a shingle with it.

A shingle hit is a strong signal, not lexical-overlap noise: 8 consecutive
normalised words matching by chance between unrelated texts is vanishingly
rare, but a paraphrase or a reformatted copy of the same source item will
almost always share several.

Usage:
    uv run python -m training.contamination_audit --corpus data/train.jsonl
    uv run python -m training.contamination_audit --corpus data/train.jsonl --refs jevbench,jabr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

SHINGLE_N = 8

JEVBENCH_PUBLIC_DIR = "/tmp/jevbench/datasets/public"
JABR_CASES_DIR = "/tmp/scratch/classifier-benchmark/cases"
DECISION_INDEX_SUITE_DIR = "/tmp/di-suite"


def _normalise(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _shingles(text: str, n: int = SHINGLE_N) -> Set[Tuple[str, ...]]:
    words = _normalise(text)
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _record_text(record: dict) -> str:
    """Every string a training record exposes: state, question, option text."""
    parts = [str(record.get("state", "")), str(record.get("question", ""))]
    for opt in record.get("options", []) or []:
        if isinstance(opt, dict):
            parts.append(str(opt.get("description", "")))
        else:
            parts.append(str(opt))
    return " ".join(parts)


def _jevbench_texts(path: str = JEVBENCH_PUBLIC_DIR) -> Iterable[Tuple[str, str]]:
    """Yields (item_id, text) for every public JevBench tier file present."""
    if not os.path.isdir(path):
        return
    for fname in ("easy.jsonl", "original.jsonl", "hard.jsonl"):
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q = row.get("question") or {}
                crit = q.get("criteria") if isinstance(q, dict) else None
                bits = [str(row.get("state", "")), str(q.get("instructions", "") if isinstance(q, dict) else "")]
                if isinstance(crit, dict):
                    bits.extend(str(v) for v in crit.values())
                elif isinstance(crit, list):
                    bits.extend(str(v) for v in crit)
                yield f"jevbench_{fname}:{row.get('id', '?')}", " ".join(bits)


def _jabr_texts(path: str = JABR_CASES_DIR) -> Iterable[Tuple[str, str]]:
    """Yields (item_id, text) for every jabr v1/v2 TOML case file present."""
    if not os.path.isdir(path):
        return
    for fname in os.listdir(path):
        if not fname.endswith(".toml"):
            continue
        fpath = os.path.join(path, fname)
        with open(fpath, "rb") as f:
            data = tomllib.load(f)
        for task in data.get("task", []):
            tid = task.get("id", "?")
            q = task.get("question") or {}
            crit = q.get("criteria") if isinstance(q, dict) else None
            base_bits = [str(q.get("instructions", "") if isinstance(q, dict) else "")]
            if isinstance(crit, dict):
                base_bits.extend(str(v) for v in crit.values())
            for i, case in enumerate(task.get("cases", []) or []):
                bits = list(base_bits) + [str(case.get("state", ""))]
                yield f"jabr_{fname}:{tid}#{i}", " ".join(bits)


def _decision_index_texts(path: str = DECISION_INDEX_SUITE_DIR) -> Iterable[Tuple[str, str]]:
    """Yields (item_id, text) from a downloaded Decision Index suite, if present."""
    fpath = os.path.join(path, "selected-rows.jsonl")
    fpath_gz = fpath + ".gz"
    if os.path.exists(fpath_gz) and not os.path.exists(fpath):
        import gzip

        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
        fpath = fpath_gz
    elif os.path.exists(fpath):
        opener = lambda p: open(p, "r", encoding="utf-8")
    else:
        return
    with opener(fpath) as f:
        for line in f:
            row = json.loads(line)
            bits = [str(row.get("state", ""))]
            for q in (row.get("questions") or {}).values():
                bits.append(str(q.get("instructions", "")))
                crit = q.get("criteria") or {}
                if isinstance(crit, dict):
                    bits.extend(str(v) for v in crit.values())
            yield f"decision_index:{row.get('id', '?')}", " ".join(bits)


REF_SOURCES = {
    "jevbench": _jevbench_texts,
    "jabr": _jabr_texts,
    "decision_index": _decision_index_texts,
}


def build_reference_index(names: List[str]) -> Dict[Tuple[str, ...], List[str]]:
    """Maps each reference shingle to the list of ref item ids it appeared in."""
    index: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    counts = Counter()
    for name in names:
        fn = REF_SOURCES.get(name)
        if fn is None:
            print(f"  [audit] unknown ref source '{name}' - skipping", file=sys.stderr)
            continue
        n = 0
        for item_id, text in fn():
            n += 1
            for sh in _shingles(text):
                if sh:
                    index[sh].append(item_id)
        counts[name] = n
        print(f"  [audit] {name}: {n} reference items indexed")
    if sum(counts.values()) == 0:
        print("  [audit] WARNING: no reference items found for any requested source "
              "(missing clones/downloads) - audit cannot catch anything this run.",
              file=sys.stderr)
    return index


def audit_corpus(corpus_path: str, ref_index: Dict[Tuple[str, ...], List[str]],
                  max_examples_per_source: int = 3) -> Tuple[int, Dict[str, List[str]]]:
    """Returns (n_hit_records, {source_kind: [example record ids/snippets]})."""
    hits_by_source: Dict[str, List[str]] = defaultdict(list)
    n_hit = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            src = record.get("source")
            src_kind = src.get("kind") if isinstance(src, dict) else str(src)
            text = _record_text(record)
            matched_refs: Set[str] = set()
            for sh in _shingles(text):
                if sh in ref_index:
                    matched_refs.update(ref_index[sh])
                    if len(matched_refs) > 5:
                        break
            if matched_refs:
                n_hit += 1
                if len(hits_by_source[src_kind]) < max_examples_per_source:
                    hits_by_source[src_kind].append(
                        f"line {lineno} matches {sorted(matched_refs)[:3]}"
                    )
    return n_hit, hits_by_source


def run_audit(corpus_path: str, ref_names: Optional[List[str]] = None) -> int:
    """Returns the number of contaminated records (0 = clean). Prints a report."""
    ref_names = ref_names or list(REF_SOURCES)
    print(f"[audit] building reference index from: {ref_names}")
    ref_index = build_reference_index(ref_names)
    print(f"[audit] scanning corpus: {corpus_path}")
    n_hit, hits_by_source = audit_corpus(corpus_path, ref_index)
    if n_hit == 0:
        print("[audit] CLEAN - 0 records share an 8-gram shingle with any reference benchmark.")
        return 0
    print(f"[audit] CONTAMINATED - {n_hit} records share an 8-gram shingle with a reference benchmark:")
    for source_kind, examples in sorted(hits_by_source.items(), key=lambda kv: -len(kv[1])):
        print(f"  source.kind={source_kind}")
        for ex in examples:
            print(f"    {ex}")
    return n_hit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Path to a training corpus .jsonl")
    parser.add_argument("--refs", default=",".join(REF_SOURCES),
                         help="Comma-separated reference sources to check against "
                              f"(available: {','.join(REF_SOURCES)})")
    args = parser.parse_args()
    n_hit = run_audit(args.corpus, args.refs.split(","))
    sys.exit(1 if n_hit else 0)


if __name__ == "__main__":
    main()
