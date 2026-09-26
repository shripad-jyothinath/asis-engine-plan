#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from fastembed import TextEmbedding

DATA_DIR = (Path(__file__).resolve().parent.parent / "asis-data-export").resolve()
MODEL_NAME = "BAAI/bge-large-en-v1.5"

def get_graph_db():
    db_path = DATA_DIR / "asis_graph.db"
    if not db_path.exists():
        print(f"Error: Knowledge graph not found at {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)

def find_definition(symbol: str, subsystem: str = None):
    conn = get_graph_db()
    query = "SELECT symbol_name, symbol_kind, rel_path, start_line, end_line, subsystem FROM symbols WHERE symbol_name = ?"
    params = [symbol]
    if subsystem and subsystem != "all":
        query += " AND subsystem = ?"
        params.append(subsystem)

    cursor = conn.execute(query, params)
    results = [
        {
            "symbol": row[0],
            "kind": row[1],
            "rel_path": row[2],
            "start_line": row[3],
            "end_line": row[4],
            "subsystem": row[5]
        }
        for row in cursor.fetchall()
    ]
    print(json.dumps(results, indent=2))

def graph_module(module_name: str):
    conn = get_graph_db()
    cursor = conn.execute("SELECT module_name, module_type, def_path, subsystem FROM build_modules WHERE module_name = ?", (module_name,))
    row = cursor.fetchone()
    if row:
        print(json.dumps({
            "module": row[0],
            "type": row[1],
            "path": row[2],
            "subsystem": row[3]
        }, indent=2))
    else:
        print(json.dumps({"error": f"Module {module_name} not found"}, indent=2))

def read_source(rel_path: str, start_line: int, end_line: int):
    # In sharded mode, source text can also be read directly from chunks parquet
    parquet_files = sorted(DATA_DIR.glob("chunks-*.parquet"))
    for pf in parquet_files:
        tbl = pq.read_table(pf)
        df = tbl.to_pydict()
        for idx in range(len(df["chunk_id"])):
            if df["rel_path"][idx] == rel_path and df["start_line"][idx] <= start_line and df["end_line"][idx] >= end_line:
                print(df["text"][idx])
                return
    print(f"Code window not found in available shards for {rel_path}:{start_line}-{end_line}")

def search_dense(query: str, top_k: int = 5):
    print(f"[ASIS] Computing query embedding with {MODEL_NAME}...")
    model = TextEmbedding(model_name=MODEL_NAME)
    query_vec = list(model.embed([query]))[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    parquet_files = sorted(DATA_DIR.glob("chunks-*.parquet"))
    if not parquet_files:
        print("No chunk parquet shards available.")
        return

    candidates = []
    for pf in parquet_files:
        tbl = pq.read_table(pf)
        df = tbl.to_pydict()
        shard_ids = df["embedding_shard_id"]
        offsets = df["embedding_offset"]

        # Cache open binary files
        open_shards = {}
        for idx in range(len(df["chunk_id"])):
            s_id = shard_ids[idx]
            offset = offsets[idx]
            if s_id not in open_shards:
                s_path = DATA_DIR / "embeddings" / "dense" / s_id
                if s_path.exists():
                    open_shards[s_id] = open(s_path, "rb")
                else:
                    continue

            f = open_shards[s_id]
            f.seek(offset)
            vec_bytes = f.read(1024 * 4) # 1024 float32
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                score = float(np.dot(query_vec, vec))
                candidates.append((score, {
                    "score": round(score, 4),
                    "chunk_id": df["chunk_id"][idx],
                    "rel_path": df["rel_path"][idx],
                    "lines": f"{df['start_line'][idx]}-{df['end_line'][idx]}",
                    "symbol": df["symbol"][idx],
                    "snippet": df["text"][idx][:200] + "..."
                }))

        for f in open_shards.values():
            f.close()

    candidates.sort(key=lambda x: x[0], reverse=True)
    results = [c[1] for c in candidates[:top_k]]
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIS Query Interface")
    subparsers = parser.add_subparsers(dest="command")

    p_def = subparsers.add_parser("find_definition")
    p_def.add_argument("symbol", type=str)
    p_def.add_argument("--subsystem", type=str, default=None)

    p_graph = subparsers.add_parser("graph")
    p_graph.add_argument("module_name", type=str)

    p_read = subparsers.add_parser("read_source")
    p_read.add_argument("path", type=str)
    p_read.add_argument("start_line", type=int)
    p_read.add_argument("end_line", type=int)

    p_search = subparsers.add_parser("search")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "find_definition":
        find_definition(args.symbol, args.subsystem)
    elif args.command == "graph":
        graph_module(args.module_name)
    elif args.command == "read_source":
        read_source(args.path, args.start_line, args.end_line)
    elif args.command == "search":
        search_dense(args.query, args.top_k)
    else:
        parser.print_help()
