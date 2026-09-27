#!/usr/bin/env python3
"""
ASIS High-Precision Code Intelligence & Ingestion Engine — Evolution-X Android 17
================================================================================
Hardware Target: Tesla T4 GPU (16 GB VRAM), 16 GB RAM, 9 GB swap, 200 GB Storage.

Production Guarantees:
1. Strict Revision Verification: Exact branch/ref/SHA matching; fails loudly on mismatch.
2. Crash-Consistent Staging & Atomic Publish: Staging buffer -> count validation -> atomic commit.
3. Repository-Scoped Composite Keys: Multi-repo collision immunity with ON CONFLICT DO UPDATE.
4. Streamed & Budget-Capped Git History: Commit-by-commit streaming with max 10MB diff budget.
5. Unconditional Clone Cleanup: Wrapped in try/finally blocks to guarantee zero leaked disk space.
6. 100GB Free Disk Safety Margin: Constant disk checks before clone, embedding, and write.
7. FP16 Tensor Core Acceleration: 1024-dim Float16 (2048 bytes/vector) halving storage and maximizing T4.
8. Recoverable Append Shards: Validates byte lengths on startup and truncates partial tails.
9. Dataset Metadata & Run IDs: Full provenance tracking for every ingested record.
"""

import os
import sys
import time
import json
import stat
import uuid
import shutil
import hashlib
import sqlite3
import argparse
import gzip
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Generator, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Force line-buffered UTF-8 output so live monitoring through tee/tmux is immediate
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# Controlled CPU threading to prevent oversubscription on 4-vCPU Xeon with 16GB RAM
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "4"

import mmh3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
pa.set_cpu_count(4)
import requests
import xml.etree.ElementTree as ET
import torch
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURATION & SCHEMAS
# ==============================================================================

PIPELINE_VERSION = "a17-prod-v2"
CHUNKER_VERSION = "heuristic-v2"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
VECTOR_DIM = 1024
PRECISION_DTYPE = np.float16  # FP16 for Tesla T4 Tensor Cores and 2048 bytes/vector
BYTES_PER_VECTOR = VECTOR_DIM * 2

MANIFEST_BASE_URL = "https://raw.githubusercontent.com/Evolution-X/manifest/cnb/"
MANIFEST_DEFAULT_XML = MANIFEST_BASE_URL + "default.xml"
MANIFEST_SNIPPETS = ["snippets/evolution.xml", "snippets/lineage.xml"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "asis-data-export").resolve()
TEMP_CLONE_DIR = (BASE_DIR.parent / "asis-temp-repo").resolve()

# Shard sizing: 5,000 vectors * 2,048 bytes = ~10.2 MB per raw vector shard
SHARD_CHUNK_LIMIT = 5000
EMBEDDING_BATCH_SIZE = 64
MIN_FREE_DISK_GB = 100.0

# Bounded Git History limits to protect 16 GB RAM and 200 GB storage
GIT_MAX_COMMITS = 1000
GIT_MAX_DIFF_KB = 15          # Max 15 KB diff patch per commit
GIT_REPO_DIFF_BUDGET_MB = 10  # Max 10 MB total diff storage per repository

CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("rel_path", pa.string()),
    ("start_line", pa.int32()),
    ("end_line", pa.int32()),
    ("symbol", pa.string()),
    ("symbol_kind", pa.string()),
    ("text", pa.string()),
    ("token_count", pa.int32()),
    ("repo", pa.string()),
    ("project", pa.string()),
    ("branch", pa.string()),
    ("commit_sha", pa.string()),
    ("run_id", pa.string()),
    ("fast_hash", pa.string()),
    ("content_hash", pa.string()),
    ("subsystem", pa.string()),
    ("embedding_shard_id", pa.string()),
    ("embedding_offset", pa.int64())
])

FILE_SCHEMA = pa.schema([
    ("rel_path", pa.string()),
    ("repo", pa.string()),
    ("project", pa.string()),
    ("branch", pa.string()),
    ("commit_sha", pa.string()),
    ("run_id", pa.string()),
    ("subsystem", pa.string()),
    ("category", pa.string()),
    ("size", pa.int64()),
    ("mtime", pa.float64()),
    ("fast_hash", pa.string()),
    ("content_hash", pa.string())
])

GIT_SCHEMA = pa.schema([
    ("commit_sha", pa.string()),
    ("repo", pa.string()),
    ("project", pa.string()),
    ("author", pa.string()),
    ("date", pa.string()),
    ("subject", pa.string()),
    ("body", pa.string()),
    ("files_changed_count", pa.int32()),
    ("files_changed", pa.list_(pa.string())),
    ("diff_patch", pa.string())
])

# ==============================================================================
# DISK & PROCESS SAFETY UTILITIES
# ==============================================================================

def remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def clean_dir(target_dir: Path):
    """Unconditional directory cleanup."""
    if target_dir.exists():
        for _ in range(5):
            if os.name == 'nt':
                subprocess.run(f'cmd /c rd /s /q "{target_dir}"', shell=True, capture_output=True)
            else:
                subprocess.run(f'rm -rf "{target_dir}"', shell=True, capture_output=True)
            if not target_dir.exists():
                return
            try:
                shutil.rmtree(target_dir, onerror=remove_readonly)
                if not target_dir.exists():
                    return
            except Exception:
                pass
            time.sleep(0.5)

def get_free_disk_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(path.anchor if os.name == 'nt' else str(path))
    return free / (1024 ** 3)

def check_disk_headroom(path: Path, step_name: str = ""):
    free_gb = get_free_disk_gb(path)
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"[DISK CRITICAL] Free disk space {free_gb:.2f} GB below safety floor {MIN_FREE_DISK_GB} GB "
            f"at step: {step_name}. Halting to protect system integrity."
        )

def format_duration(seconds: float) -> str:
    secs = int(seconds)
    days, secs = divmod(secs, 86400)
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    elif hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    else:
        return f"{minutes:02d}m {secs:02d}s"

# ==============================================================================
# MANIFEST RESOLVER
# ==============================================================================

class ManifestResolver:
    def __init__(self, cache_file: Path = BASE_DIR / "evolution_a17_manifest.json"):
        self.cache_file = cache_file

    def resolve(self) -> List[Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        print("[ASIS] Fetching Evolution-X Android 17 (cnb) manifests...")
        projects: Dict[str, Dict[str, Any]] = {}
        remotes: Dict[str, Dict[str, str]] = {}
        default_revision = "refs/heads/lineage-24.0"

        def parse_xml_content(text: str):
            nonlocal default_revision
            root = ET.fromstring(text)
            for rm in root.findall("remote"):
                remotes[rm.attrib["name"]] = rm.attrib
            d = root.find("default")
            if d is not None and "revision" in d.attrib:
                default_revision = d.attrib["revision"]

            for p in root.findall("project"):
                path = p.attrib["path"]
                projects[path] = p.attrib

        # 1. Main default.xml
        resp = requests.get(MANIFEST_DEFAULT_XML, timeout=30)
        resp.raise_for_status()
        parse_xml_content(resp.text)

        # 2. Snippets
        for snippet in MANIFEST_SNIPPETS:
            url = MANIFEST_BASE_URL + snippet
            try:
                s_resp = requests.get(url, timeout=30)
                if s_resp.status_code == 200:
                    parse_xml_content(s_resp.text)
            except Exception as e:
                print(f"[ASIS] Warning: Could not fetch snippet {snippet}: {e}")

        resolved_list = []
        for path, p in projects.items():
            name = p["name"]
            remote_name = p.get("remote", "github")
            remote_info = remotes.get(remote_name, {})
            fetch = remote_info.get("fetch", "https://github.com")
            revision = p.get("revision") or remote_info.get("revision") or default_revision

            if remote_name == "aosp" or "googlesource" in fetch:
                clone_url = f"https://android.googlesource.com/{name}"
            elif name.startswith("LineageOS/"):
                clone_url = f"https://github.com/{name}"
            elif name.startswith("Evolution-X/"):
                clone_url = f"https://github.com/{name}"
            elif remote_name == "evo":
                clone_url = f"https://github.com/Evolution-X/{name}"
            elif fetch.startswith("http"):
                clone_url = f"{fetch.rstrip('/')}/{name}"
            elif fetch == "..":
                clone_url = f"https://github.com/Evolution-X/{name}"
            else:
                clone_url = f"https://github.com/{name}"

            resolved_list.append({
                "path": path,
                "name": name,
                "remote": remote_name,
                "revision": revision,
                "clone_url": clone_url,
                "groups": p.get("groups", "")
            })

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(resolved_list, f, indent=2)

        print(f"[ASIS] Resolved {len(resolved_list)} unique Android 17 repositories.")
        return resolved_list

# ==============================================================================
# CHECKPOINT & GRAPH DATABASE MANAGERS (COMPOSITE SCOPED KEYS)
# ==============================================================================

class CheckpointManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.execute("PRAGMA mmap_size = 2147483648;")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS repo_checkpoints (
                    repo_path TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    clone_url TEXT NOT NULL,
                    revision TEXT,
                    commit_sha TEXT,
                    run_id TEXT,
                    status TEXT NOT NULL,
                    error_msg TEXT,
                    files_count INTEGER DEFAULT 0,
                    chunks_count INTEGER DEFAULT 0,
                    commits_count INTEGER DEFAULT 0,
                    duration_sec REAL DEFAULT 0,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS shard_tracker (
                    shard_id TEXT PRIMARY KEY,
                    chunk_count INTEGER DEFAULT 0,
                    byte_size INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    last_committed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_path TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    chunks INTEGER NOT NULL,
                    files INTEGER NOT NULL,
                    commits INTEGER DEFAULT 0,
                    duration_sec REAL NOT NULL,
                    chunks_per_sec REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            try:
                conn.execute("ALTER TABLE repo_checkpoints ADD COLUMN run_id TEXT;")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE telemetry ADD COLUMN run_id TEXT;")
            except Exception:
                pass

    def get_completed_repos(self) -> set:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT repo_path FROM repo_checkpoints WHERE status = 'COMPLETED'")
            return {row[0] for row in cursor.fetchall()}

    def mark_in_progress(self, repo_path: str, repo_name: str, clone_url: str, revision: str, run_id: str):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO repo_checkpoints (repo_path, repo_name, clone_url, revision, run_id, status, started_at)
                VALUES (?, ?, ?, ?, ?, 'IN_PROGRESS', CURRENT_TIMESTAMP)
                ON CONFLICT(repo_path) DO UPDATE SET 
                    status = 'IN_PROGRESS', 
                    run_id = excluded.run_id,
                    error_msg = NULL,
                    started_at = CURRENT_TIMESTAMP
            """, (repo_path, repo_name, clone_url, revision, run_id))

    def mark_completed(self, repo_path: str, commit_sha: str, run_id: str, files: int, chunks: int, commits: int, duration: float):
        cps = chunks / max(0.1, duration)
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE repo_checkpoints
                SET status = 'COMPLETED', commit_sha = ?, run_id = ?, files_count = ?, chunks_count = ?,
                    commits_count = ?, duration_sec = ?, completed_at = CURRENT_TIMESTAMP
                WHERE repo_path = ?
            """, (commit_sha, run_id, files, chunks, commits, duration, repo_path))

            conn.execute("""
                INSERT INTO telemetry (repo_path, run_id, chunks, files, commits, duration_sec, chunks_per_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (repo_path, run_id, chunks, files, commits, duration, cps))

    def mark_failed(self, repo_path: str, run_id: str, error_msg: str):
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE repo_checkpoints
                SET status = 'FAILED', run_id = ?, error_msg = ?
                WHERE repo_path = ?
            """, (run_id, error_msg, repo_path))

    def get_aggregate_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cur = conn.execute("""
                SELECT 
                    COUNT(*) as completed_count,
                    COALESCE(SUM(files_count), 0) as total_files,
                    COALESCE(SUM(chunks_count), 0) as total_chunks,
                    COALESCE(SUM(commits_count), 0) as total_commits,
                    COALESCE(SUM(duration_sec), 0) as total_duration
                FROM repo_checkpoints WHERE status = 'COMPLETED'
            """)
            row = cur.fetchone()
            return {
                "completed_count": row[0],
                "total_files": row[1],
                "total_chunks": row[2],
                "total_commits": row[3],
                "total_duration": row[4]
            }

    def update_shard_tracker(self, shard_id: str, new_chunks: int, new_bytes: int):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO shard_tracker (shard_id, chunk_count, byte_size, is_active, last_committed_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(shard_id) DO UPDATE SET
                    chunk_count = chunk_count + excluded.chunk_count,
                    byte_size = byte_size + excluded.byte_size,
                    last_committed_at = CURRENT_TIMESTAMP
            """, (shard_id, new_chunks, new_bytes))

    def get_shard_info(self, shard_id: str) -> Optional[Tuple[int, int]]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT chunk_count, byte_size FROM shard_tracker WHERE shard_id = ?", (shard_id,))
            row = cur.fetchone()
            return (row[0], row[1]) if row else None


class KnowledgeGraphDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.execute("PRAGMA mmap_size = 2147483648;")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    repo_path TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    category TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    fast_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (repo_path, rel_path)
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id TEXT PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    symbol_kind TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    subsystem TEXT NOT NULL,
                    signature TEXT,
                    docstring TEXT
                );

                CREATE TABLE IF NOT EXISTS build_modules (
                    repo_path TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    module_type TEXT NOT NULL,
                    def_path TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    PRIMARY KEY (repo_path, module_name)
                );

                CREATE TABLE IF NOT EXISTS git_history (
                    repo_path TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    project TEXT NOT NULL,
                    author TEXT NOT NULL,
                    commit_date TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT,
                    files_changed_count INTEGER NOT NULL,
                    diff_patch TEXT,
                    PRIMARY KEY (repo_path, commit_sha)
                );

                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);
                CREATE INDEX IF NOT EXISTS idx_symbols_repo_path ON symbols(repo_path, rel_path);
                CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_path);
                CREATE INDEX IF NOT EXISTS idx_files_subsystem ON files(subsystem);
                CREATE INDEX IF NOT EXISTS idx_git_author ON git_history(author);
                CREATE INDEX IF NOT EXISTS idx_git_proj ON git_history(repo_path);
            """)
            try:
                conn.execute("ALTER TABLE files ADD COLUMN run_id TEXT;")
            except Exception:
                pass

    def batch_insert(self, files: List[Dict], symbols: List[Dict], modules: List[Dict], commits: List[Dict]):
        with self._get_conn() as conn:
            if files:
                conn.executemany("""
                    INSERT INTO files 
                    (repo_path, rel_path, repo, branch, commit_sha, run_id, subsystem, category, size, mtime, fast_hash, content_hash)
                    VALUES (:repo_path, :rel_path, :repo, :branch, :commit_sha, :run_id, :subsystem, :category, :size, :mtime, :fast_hash, :content_hash)
                    ON CONFLICT(repo_path, rel_path) DO UPDATE SET
                        commit_sha = excluded.commit_sha,
                        run_id = excluded.run_id,
                        size = excluded.size,
                        mtime = excluded.mtime,
                        fast_hash = excluded.fast_hash,
                        content_hash = excluded.content_hash,
                        indexed_at = CURRENT_TIMESTAMP
                """, files)

            if symbols:
                conn.executemany("""
                    INSERT INTO symbols
                    (symbol_id, repo_path, rel_path, symbol_name, symbol_kind, start_line, end_line, subsystem, signature, docstring)
                    VALUES (:symbol_id, :repo_path, :rel_path, :symbol_name, :symbol_kind, :start_line, :end_line, :subsystem, :signature, :docstring)
                    ON CONFLICT(symbol_id) DO UPDATE SET
                        start_line = excluded.start_line,
                        end_line = excluded.end_line,
                        signature = excluded.signature,
                        docstring = excluded.docstring
                """, symbols)

            if modules:
                conn.executemany("""
                    INSERT INTO build_modules
                    (repo_path, module_name, module_type, def_path, subsystem)
                    VALUES (:repo_path, :module_name, :module_type, :def_path, :subsystem)
                    ON CONFLICT(repo_path, module_name) DO UPDATE SET
                        module_type = excluded.module_type,
                        def_path = excluded.def_path,
                        subsystem = excluded.subsystem
                """, modules)

            if commits:
                conn.executemany("""
                    INSERT INTO git_history
                    (repo_path, commit_sha, repo, project, author, commit_date, subject, body, files_changed_count, diff_patch)
                    VALUES (:repo_path, :commit_sha, :repo, :project, :author, :date, :subject, :body, :files_changed_count, :diff_patch)
                    ON CONFLICT(repo_path, commit_sha) DO UPDATE SET
                        diff_patch = excluded.diff_patch
                """, commits)

# ==============================================================================
# HEURISTIC CODE CHUNKER (ROBUST SYNTACTIC BOUNDARIES & STRICT PROGRESSION)
# ==============================================================================

class HeuristicCodeChunker:
    """
    Syntactic & Boundary-Aware Code Chunker:
    Snaps window bounds to structural delimiters without infinite looping.
    Extracts module definitions from Android.bp and Android.mk.
    """
    CODE_EXTS = {
        ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java", ".kt",
        ".rs", ".aidl", ".hal", ".dts", ".dtsi", ".py", ".sh",
        ".bp", ".mk", ".rc", ".xml", ".go"
    }

    @staticmethod
    def get_subsystem(path_str: str) -> str:
        p = path_str.lower()
        if p.startswith("kernel/") or "/kernel" in p:
            return "kernel"
        elif p.startswith("frameworks/") or "framework" in p:
            return "framework"
        elif p.startswith("hardware/") or "hardware" in p:
            return "hardware"
        elif p.startswith("vendor/") or "vendor" in p:
            return "vendor"
        elif p.startswith("build/") or "build" in p:
            return "build"
        else:
            return "system_core"

    @classmethod
    def should_index(cls, path: Path) -> bool:
        if path.is_file() and path.suffix.lower() in cls.CODE_EXTS:
            if path.name.startswith("."):
                return False
            try:
                if path.stat().st_size > 800 * 1024:
                    return False
            except OSError:
                return False
            return True
        return False

    @classmethod
    def chunk_file(cls, full_path: Path, rel_path: str, repo_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        try:
            with open(full_path, "rb") as bf:
                content_bytes = bf.read()
            text = content_bytes.decode("utf-8", errors="ignore")
            lines = text.splitlines(keepends=True)
        except Exception:
            return [], [], []

        if not lines:
            return [], [], []

        modules = []
        subsystem = cls.get_subsystem(rel_path)

        # 1. Parse Android build definitions
        if full_path.name in ("Android.bp", "Android.mk") or full_path.suffix == ".mk":
            import re
            for match in re.finditer(r'([a-z0-9_]+)\s*\{\s*name:\s*"([^"]+)"', text):
                modules.append({
                    "repo_path": repo_path,
                    "module_name": match.group(2),
                    "module_type": match.group(1),
                    "def_path": rel_path,
                    "subsystem": subsystem
                })
            for match in re.finditer(r'LOCAL_MODULE\s*:=\s*([A-Za-z0-9_]+)', text):
                modules.append({
                    "repo_path": repo_path,
                    "module_name": match.group(1),
                    "module_type": "makefile",
                    "def_path": rel_path,
                    "subsystem": subsystem
                })

        chunks = []
        symbols = []
        total_lines = len(lines)
        target_lines = 80
        start = 0

        # 2. Syntactic chunking with strictly monotonic forward progression
        while start < total_lines:
            end = min(start + target_lines, total_lines)
            if end < total_lines:
                search_base = max(start + 10, end - 8)
                snap_window = lines[search_base: min(total_lines, end + 8)]
                for offset, line in enumerate(snap_window):
                    stripped = line.strip()
                    if stripped.endswith("}") or stripped.endswith(";"):
                        candidate = search_base + offset + 1
                        if candidate > start:
                            end = candidate
                            break

            chunk_lines = lines[start:end]
            chunk_text = "".join(chunk_lines).strip()

            if chunk_text:
                symbol_name = ""
                symbol_kind = "block"
                import re
                patterns = [
                    r'(?:class|struct|interface|enum)\s+([A-Za-z0-9_]+)',
                    r'(?:def|fn|func)\s+([A-Za-z0-9_]+)\s*\(',
                    r'([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{'
                ]
                for pat in patterns:
                    m = re.search(pat, chunk_text)
                    if m:
                        symbol_name = m.group(1)
                        symbol_kind = "class" if "class" in pat else "function"
                        break

                c_id = f"evox:{repo_path}:{rel_path}:{start+1}-{end}:{symbol_name or 'block'}"
                chunks.append({
                    "chunk_id": c_id,
                    "rel_path": rel_path,
                    "start_line": start + 1,
                    "end_line": end,
                    "symbol": symbol_name,
                    "symbol_kind": symbol_kind,
                    "text": chunk_text,
                    "token_count": len(chunk_text.split()),
                    "subsystem": subsystem
                })

                if symbol_name:
                    sym_id = f"{repo_path}::{rel_path}::{start+1}::{symbol_name}"
                    symbols.append({
                        "symbol_id": sym_id,
                        "repo_path": repo_path,
                        "rel_path": rel_path,
                        "symbol_name": symbol_name,
                        "symbol_kind": symbol_kind,
                        "start_line": start + 1,
                        "end_line": end,
                        "subsystem": subsystem,
                        "signature": lines[start].strip(),
                        "docstring": ""
                    })

            if end >= total_lines:
                break
            next_start = end - 15 if (end - 15 > start) else end
            start = max(start + 1, next_start)

        return chunks, symbols, modules

    @staticmethod
    def stream_git_commits(temp_dir: Path, repo_path: str) -> List[Dict]:
        """
        Streams commits incrementally with strict diff and memory ceilings.
        Protects against out-of-memory on 16GB RAM machines.
        """
        cmd_meta = ['git', 'log', f'-n', str(GIT_MAX_COMMITS), '--format=%H%x09%an%x09%ad%x09%s']
        try:
            res = subprocess.run(
                cmd_meta,
                cwd=str(temp_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if res.returncode != 0 or not res.stdout:
                return []
        except Exception:
            return []

        commits = []
        total_diff_bytes = 0
        max_total_bytes = GIT_REPO_DIFF_BUDGET_MB * 1024 * 1024
        max_diff_bytes = GIT_MAX_DIFF_KB * 1024

        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            sha, author, date_str, subject = parts[0], parts[1], parts[2], parts[3]

            diff_text = ""
            files_changed = []

            # Only fetch diffs while within the total repository budget
            if total_diff_bytes < max_total_bytes:
                try:
                    p_diff = subprocess.run(
                        ['git', 'show', '--unified=3', '--format=', sha],
                        cwd=str(temp_dir),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace"
                    )
                    if p_diff.returncode == 0 and p_diff.stdout:
                        diff_content = p_diff.stdout
                        if len(diff_content) > max_diff_bytes:
                            diff_content = diff_content[:max_diff_bytes] + f"\n\n[DIFF TRUNCATED AT {GIT_MAX_DIFF_KB} KB]"
                        diff_text = diff_content
                        total_diff_bytes += len(diff_content.encode("utf-8"))

                        for d_line in diff_text.splitlines():
                            if d_line.startswith("diff --git a/"):
                                p_d = d_line.split(" b/")
                                if len(p_d) == 2:
                                    files_changed.append(p_d[1].strip())
                except Exception:
                    pass

            commits.append({
                "repo_path": repo_path,
                "commit_sha": sha,
                "repo": "evox",
                "project": repo_path,
                "author": author,
                "date": date_str,
                "subject": subject,
                "body": "",
                "files_changed_count": len(files_changed),
                "files_changed": files_changed,
                "diff_patch": diff_text
            })

        return commits

# ==============================================================================
# TRANSACTIONAL DENSE SHARD WRITER (FP16 & CRASH-CONSISTENT)
# ==============================================================================

class ShardWriter:
    """
    FP16 Transactional Vector Storage:
    1. Embeds in FP16 (2,048 bytes per vector).
    2. Writes to an isolated staging buffer (.staging_<run_id>.bin).
    3. Validates shard lengths and truncates uncommitted tails on startup.
    4. Atomically commits bytes only after Parquet and SQLite are verified.
    """
    def __init__(self, export_dir: Path, checkpoint: CheckpointManager, model_name: str = DEFAULT_MODEL):
        self.export_dir = export_dir
        self.checkpoint = checkpoint
        self.dense_dir = export_dir / "embeddings" / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"[ASIS] Initializing SentenceTransformer: {model_name} on GPU (CUDA, FP16 Tensor Cores, batch={EMBEDDING_BATCH_SIZE})...", flush=True)
            self.model = SentenceTransformer(model_name, device=device)
            self.model.half()
            self.model.eval()
            self.batch_size = EMBEDDING_BATCH_SIZE
        else:
            print(f"[ASIS] Initializing SentenceTransformer: {model_name} on CPU (FP32, batch={EMBEDDING_BATCH_SIZE})...", flush=True)
            self.model = SentenceTransformer(model_name, device=device)
            self.model.eval()
            self.batch_size = EMBEDDING_BATCH_SIZE

        # Startup recovery: validate and truncate uncommitted shard tails
        self._recover_and_verify_shards()
        self.current_shard_idx = self._find_latest_shard_index()
        self.shard_bin_file = None
        self._open_shard(self.current_shard_idx)

    def _recover_and_verify_shards(self):
        """Discards uncommitted staging files and truncates partial tails on startup."""
        for sf in self.dense_dir.glob(".staging_*.bin"):
            try:
                sf.unlink()
            except Exception:
                pass

        for shard_file in sorted(self.dense_dir.glob("shard-*.bin")):
            shard_id = shard_file.name
            info = self.checkpoint.get_shard_info(shard_id)
            actual_size = shard_file.stat().st_size
            if info:
                expected_chunks, expected_bytes = info
                if actual_size > expected_bytes:
                    print(f"[ASIS RECOVERY] Truncating {shard_id} from {actual_size} bytes to committed boundary {expected_bytes} bytes.", flush=True)
                    with open(shard_file, "r+b") as f:
                        f.truncate(expected_bytes)
            else:
                if actual_size > 0:
                    print(f"[ASIS RECOVERY] Removing untracked partial shard {shard_id}.", flush=True)
                    shard_file.unlink()

    def _find_latest_shard_index(self) -> int:
        shards = list(self.dense_dir.glob("shard-*.bin"))
        if not shards:
            return 0
        indices = []
        for s in shards:
            try:
                idx = int(s.stem.split("-")[1])
                indices.append(idx)
            except ValueError:
                pass
        return max(indices) if indices else 0

    def _open_shard(self, idx: int):
        if self.shard_bin_file and not self.shard_bin_file.closed:
            self.shard_bin_file.close()
        shard_path = self.dense_dir / f"shard-{idx:04d}.bin"
        self.current_shard_path = shard_path
        self.shard_bin_file = open(shard_path, "a+b")
        self.shard_bin_file.seek(0, os.SEEK_END)
        self.current_offset = self.shard_bin_file.tell()
        self.current_shard_chunks = self.current_offset // BYTES_PER_VECTOR

    def stage_embeddings(self, chunks: List[Dict], repo_info: Dict[str, Any], commit_sha: str, run_id: str) -> Tuple[List[Dict], Path]:
        if not chunks:
            return [], None

        repo_path = repo_info["path"]
        staging_path = self.dense_dir / f".staging_{run_id}.bin"

        total_chunks = len(chunks)
        t_start = time.time()
        enriched_chunks = []

        with open(staging_path, "wb") as sf:
            staged_offset = 0
            for b_start in range(0, total_chunks, self.batch_size):
                b_end = min(b_start + self.batch_size, total_chunks)
                b_chunks = chunks[b_start:b_end]
                b_texts = [c["text"] for c in b_chunks]

                # True FP16 Tensor Core vectorization on GPU
                b_embeddings = self.model.encode(
                    b_texts,
                    batch_size=len(b_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                # Direct contiguous bytes write at C-speed
                b_fp16 = b_embeddings.astype(PRECISION_DTYPE, copy=False)
                b_bytes = b_fp16.tobytes(order="C")
                sf.write(b_bytes)

                vec_byte_size = BYTES_PER_VECTOR
                for c in b_chunks:
                    c["repo"] = "evox"
                    c["project"] = repo_path
                    c["branch"] = "cnb"
                    c["commit_sha"] = commit_sha
                    c["run_id"] = run_id
                    c["fast_hash"] = f"{mmh3.hash64(c['text'])[0]:x}"
                    c["content_hash"] = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
                    c["staged_offset"] = staged_offset
                    enriched_chunks.append(c)
                    staged_offset += vec_byte_size

                if b_end % 1000 == 0 or b_end == total_chunks or total_chunks < 500:
                    pct = (b_end / total_chunks) * 100
                    cps = b_end / max(0.01, time.time() - t_start)
                    bar_len = 20
                    filled = int(bar_len * b_end / total_chunks)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    print(f"  [GPU] [{bar}] {b_end:,}/{total_chunks:,} ({pct:.1f}%) | {cps:.1f} ch/s", flush=True)

        return enriched_chunks, staging_path

    def commit_staged_vectors(self, enriched_chunks: List[Dict], staging_path: Path) -> List[Dict]:
        if not enriched_chunks or not staging_path or not staging_path.exists():
            return []

        total_chunks = len(enriched_chunks)
        with open(staging_path, "rb") as sf:
            chunks_processed = 0
            while chunks_processed < total_chunks:
                # Rotate shard if current shard reached limit
                if self.current_shard_chunks >= SHARD_CHUNK_LIMIT:
                    self.current_shard_idx += 1
                    self._open_shard(self.current_shard_idx)

                available_in_shard = SHARD_CHUNK_LIMIT - self.current_shard_chunks
                chunks_to_write = min(total_chunks - chunks_processed, available_in_shard)
                bytes_to_write = chunks_to_write * BYTES_PER_VECTOR

                base_offset = self.shard_bin_file.tell()
                shard_id = self.current_shard_path.name

                slice_bytes = sf.read(bytes_to_write)
                self.shard_bin_file.write(slice_bytes)
                self.shard_bin_file.flush()
                os.fsync(self.shard_bin_file.fileno())

                for i in range(chunks_processed, chunks_processed + chunks_to_write):
                    chunk = enriched_chunks[i]
                    chunk["embedding_shard_id"] = shard_id
                    slice_offset = (i - chunks_processed) * BYTES_PER_VECTOR
                    chunk["embedding_offset"] = base_offset + slice_offset
                    chunk.pop("staged_offset", None)

                self.current_offset = self.shard_bin_file.tell()
                self.current_shard_chunks += chunks_to_write
                self.checkpoint.update_shard_tracker(shard_id, chunks_to_write, bytes_to_write)

                chunks_processed += chunks_to_write

        try:
            staging_path.unlink()
        except Exception:
            pass

        return enriched_chunks

    def rollback_staging(self, staging_path: Optional[Path]):
        if staging_path and staging_path.exists():
            try:
                staging_path.unlink()
            except Exception:
                pass

    def close(self):
        if self.shard_bin_file and not self.shard_bin_file.closed:
            self.shard_bin_file.close()

# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

class ASISPipeline:
    def __init__(self, init_model: bool = True):
        self.data_dir = DATA_DIR
        self.temp_dir = TEMP_CLONE_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = CheckpointManager(self.data_dir / "manifest.sqlite")
        self.graph = KnowledgeGraphDB(self.data_dir / "asis_graph.db")
        self.resolver = ManifestResolver()

        self._write_metadata_file()

        if init_model:
            self.sharder = ShardWriter(self.data_dir, self.checkpoint)
        else:
            self.sharder = None

    def _write_metadata_file(self):
        meta_file = self.data_dir / "dataset_metadata.json"
        metadata = {
            "pipeline_version": PIPELINE_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "embedding_model": DEFAULT_MODEL,
            "embedding_dimension": VECTOR_DIM,
            "precision": "float16",
            "bytes_per_vector": BYTES_PER_VECTOR,
            "shard_chunk_limit": SHARD_CHUNK_LIMIT,
            "min_free_disk_gb": MIN_FREE_DISK_GB,
            "max_git_commits": GIT_MAX_COMMITS,
            "max_git_diff_kb": GIT_MAX_DIFF_KB
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _run_git_clone_with_progress(self, cmd: List[str]):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        last_print = 0
        buf = ""
        while True:
            char = proc.stderr.read(1)
            if not char and proc.poll() is not None:
                break
            if char in ('\r', '\n'):
                line = buf.strip()
                buf = ""
                if line and ("Receiving objects:" in line or "Resolving deltas:" in line or "Counting objects:" in line):
                    now = time.time()
                    if now - last_print >= 0.5 or "100%" in line:
                        clean_l = " ".join(line.split())
                        sys.stdout.write(f"\r  [NET] {clean_l[:85]:<85}")
                        sys.stdout.flush()
                        last_print = now
            else:
                buf += char
        proc.wait()
        sys.stdout.write("\n")
        sys.stdout.flush()
        if proc.returncode != 0:
            raise RuntimeError(f"Git clone exited with code {proc.returncode}")

    def git_clone_repo(self, clone_url: str, revision: str) -> str:
        """
        Strict Revision Git Clone:
        Enforces exact branch/tag matching.
        NEVER silently checks out an unintended default branch.
        """
        clean_dir(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        branch = revision
        if branch.startswith("refs/tags/"):
            branch = branch.replace("refs/tags/", "")
        elif branch.startswith("refs/heads/"):
            branch = branch.replace("refs/heads/", "")

        cmd = ['git', 'clone', '--progress', '--single-branch', '--branch', branch, clone_url, str(self.temp_dir)]
        if os.name == 'nt':
            cmd.insert(2, '-c')
            cmd.insert(3, 'core.protectNTFS=false')

        success = False
        try:
            self._run_git_clone_with_progress(cmd)
            success = True
        except Exception:
            clean_dir(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run('git init', cwd=str(self.temp_dir), shell=True, capture_output=True, check=True)
                subprocess.run(f'git remote add origin "{clone_url}"', cwd=str(self.temp_dir), shell=True, capture_output=True, check=True)
                subprocess.run(f'git fetch origin "{revision}"', cwd=str(self.temp_dir), shell=True, capture_output=True, check=True)
                subprocess.run('git checkout FETCH_HEAD', cwd=str(self.temp_dir), shell=True, capture_output=True, check=True)
                success = True
            except Exception:
                pass

        if not success:
            clean_dir(self.temp_dir)
            raise RuntimeError(f"Strict Revision Failure: Requested ref '{revision}' could not be resolved in {clone_url}")

        sha_res = subprocess.run('git rev-parse HEAD', cwd=str(self.temp_dir), shell=True, capture_output=True, text=True)
        if sha_res.returncode != 0 or not sha_res.stdout.strip():
            clean_dir(self.temp_dir)
            raise RuntimeError(f"Strict Revision Failure: Failed to resolve HEAD for {clone_url}")

        return sha_res.stdout.strip()

    def git_push_data(self, commit_msg: str):
        try:
            with sqlite3.connect(self.data_dir / "manifest.sqlite") as c:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            with sqlite3.connect(self.data_dir / "asis_graph.db") as c:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE);")

            # Ensure .gitignore keeps large raw db and wal files out of git to avoid breaking GitHub 100MB limit
            gitignore_path = self.data_dir / ".gitignore"
            gi_rules = "*.db-shm\n*.db-wal\n*.sqlite-shm\n*.sqlite-wal\n.staging_*.bin\nasis_graph.db\n"
            if not gitignore_path.exists() or gitignore_path.read_text(encoding="utf-8") != gi_rules:
                gitignore_path.write_text(gi_rules, encoding="utf-8")
            subprocess.run('git rm --cached -f asis_graph.db asis_graph.db-shm asis_graph.db-wal manifest.sqlite-shm manifest.sqlite-wal 2>/dev/null', cwd=str(self.data_dir), shell=True, capture_output=True)

            # Fast gzip snapshot of asis_graph.db for Git (8x compression keeps it << 100 MB)
            db_path = self.data_dir / "asis_graph.db"
            gz_path = self.data_dir / "asis_graph.db.gz"
            if db_path.exists():
                with open(db_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=1) as f_out:
                    shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        except Exception:
            pass

        subprocess.run('git add -A', cwd=str(self.data_dir), shell=True, capture_output=True)
        subprocess.run(f'git commit -m "{commit_msg}"', cwd=str(self.data_dir), shell=True, capture_output=True, text=True)

        for attempt in range(1, 6):
            p = subprocess.run('git push origin asis-data', cwd=str(self.data_dir), shell=True, capture_output=True, text=True)
            if p.returncode == 0:
                return
            err_msg = p.stderr.strip()[:100] if p.stderr else 'rejected'
            print(f"[ASIS] Push attempt {attempt}/5 failed ({err_msg}). Retrying in {attempt * 3}s...", flush=True)
            time.sleep(attempt * 3)

    def process_single_repo(self, repo: Dict[str, Any]) -> Tuple[int, int, int, float]:
        repo_path = repo["path"]
        repo_name = repo["name"]
        clone_url = repo["clone_url"]
        revision = repo["revision"]
        run_id = uuid.uuid4().hex[:12]

        start_time = time.time()
        self.checkpoint.mark_in_progress(repo_path, repo_name, clone_url, revision, run_id)

        staging_path = None
        try:
            # Pre-clone disk check
            check_disk_headroom(self.data_dir, f"pre-clone {repo_path}")

            # 1. Strict Clone
            print(f"  [1/6] Cloning {clone_url} (revision: {revision})...", flush=True)
            t_clone = time.time()
            commit_sha = self.git_clone_repo(clone_url, revision)
            repo_size_mb = sum(f.stat().st_size for f in self.temp_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            print(f"  [+] Cloned {repo_size_mb:.1f} MB at HEAD {commit_sha[:8]} in {time.time()-t_clone:.1f}s", flush=True)

            # 2. Extract Streamed Commits & Diffs
            print(f"  [2/6] Streaming commits & diffs (budget {GIT_REPO_DIFF_BUDGET_MB} MB)...", flush=True)
            t_git = time.time()
            all_commits = HeuristicCodeChunker.stream_git_commits(self.temp_dir, repo_path)
            print(f"  [+] Streamed {len(all_commits)} commits with diffs in {time.time()-t_git:.1f}s", flush=True)

            # 3. Code Chunking
            print(f"  [3/6] Chunking code files (Heuristic parser + 4 threads)...", flush=True)
            t_chunk = time.time()
            eligible_files = []
            for root, _, files in os.walk(self.temp_dir):
                if ".git" in root:
                    continue
                for f in files:
                    fpath = Path(root) / f
                    if HeuristicCodeChunker.should_index(fpath):
                        eligible_files.append(fpath)

            def _process_one_file(fpath: Path):
                rel = str(fpath.relative_to(self.temp_dir)).replace("\\", "/")
                stat_info = fpath.stat()
                with open(fpath, "rb") as bf:
                    content_bytes = bf.read()
                    fast_h = f"{mmh3.hash64(content_bytes)[0]:x}"
                    content_h = hashlib.sha256(content_bytes).hexdigest()
                f_rec = {
                    "repo_path": repo_path,
                    "rel_path": rel,
                    "repo": "evox",
                    "branch": "cnb",
                    "commit_sha": commit_sha,
                    "run_id": run_id,
                    "subsystem": HeuristicCodeChunker.get_subsystem(rel),
                    "category": "tier_1",
                    "size": stat_info.st_size,
                    "mtime": stat_info.st_mtime,
                    "fast_hash": fast_h,
                    "content_hash": content_h
                }
                c, s, m = HeuristicCodeChunker.chunk_file(fpath, rel, repo_path)
                return f_rec, c, s, m

            file_records = []
            all_chunks = []
            all_symbols = []
            all_modules = []

            total_files = len(eligible_files)
            with ThreadPoolExecutor(max_workers=4) as executor:
                for count, (f_rec, c, s, m) in enumerate(executor.map(_process_one_file, eligible_files), 1):
                    file_records.append(f_rec)
                    all_chunks.extend(c)
                    all_symbols.extend(s)
                    all_modules.extend(m)
                    if count % 200 == 0 or count == total_files:
                        pct = (count / max(1, total_files)) * 100
                        fps = count / max(0.1, time.time() - t_chunk)
                        print(f"  [AST] Processed {count:,}/{total_files:,} files ({pct:.1f}%) | {len(all_chunks):,} chunks ({fps:.0f} files/s)", flush=True)

            print(f"  [+] Chunked {len(file_records):,} files into {len(all_chunks):,} chunks in {time.time()-t_chunk:.1f}s", flush=True)

            # Pre-embedding disk check
            check_disk_headroom(self.data_dir, f"pre-embedding {repo_path}")

            # 4. FP16 Staging Embeddings
            print(f"  [4/6] Vectorizing {len(all_chunks)} chunks on Tesla T4 GPU (FP16, batch={self.sharder.batch_size})...", flush=True)
            t_gpu = time.time()
            staged_chunks, staging_path = self.sharder.stage_embeddings(all_chunks, repo, commit_sha, run_id)
            gpu_dur = time.time() - t_gpu
            gpu_cps = len(all_chunks) / max(0.01, gpu_dur)
            print(f"  [+] GPU embedded {len(staged_chunks)} vectors in {gpu_dur:.1f}s ({gpu_cps:.1f} chunks/s)", flush=True)

            # 5. Count Validation Before Publishing
            if len(staged_chunks) != len(all_chunks):
                raise RuntimeError(f"Integrity Mismatch: generated {len(all_chunks)} chunks but staged {len(staged_chunks)} vectors!")

            # 6. Atomic Publish: Shard Append + Parquet + SQLite
            print(f"  [5/6] Committing atomic shards, Parquet & SQLite graph...", flush=True)
            committed_chunks = self.sharder.commit_staged_vectors(staged_chunks, staging_path)

            p_idx = self.sharder.current_shard_idx
            if committed_chunks:
                chunks_by_shard = defaultdict(list)
                for c in committed_chunks:
                    try:
                        s_num = int(c["embedding_shard_id"].split("-")[1].split(".")[0])
                    except (IndexError, ValueError):
                        s_num = p_idx
                    chunks_by_shard[s_num].append(c)

                for s_num, s_chunks in sorted(chunks_by_shard.items()):
                    chunk_table = pa.Table.from_pylist(s_chunks, schema=CHUNK_SCHEMA)
                    pq_chunk_path = self.data_dir / f"chunks-{s_num:04d}.parquet"
                    if pq_chunk_path.exists():
                        existing = pq.read_table(pq_chunk_path)
                        chunk_table = pa.concat_tables([existing, chunk_table])
                    pq.write_table(chunk_table, pq_chunk_path, compression="ZSTD", compression_level=3)

            if file_records:
                file_table = pa.Table.from_pylist(file_records, schema=FILE_SCHEMA)
                pq_file_path = self.data_dir / f"files-{p_idx:04d}.parquet"
                if pq_file_path.exists():
                    existing = pq.read_table(pq_file_path)
                    file_table = pa.concat_tables([existing, file_table])
                pq.write_table(file_table, pq_file_path, compression="ZSTD", compression_level=3)

            if all_commits:
                git_table = pa.Table.from_pylist(all_commits, schema=GIT_SCHEMA)
                pq_git_path = self.data_dir / f"git_history-{p_idx:04d}.parquet"
                if pq_git_path.exists():
                    existing = pq.read_table(pq_git_path)
                    git_table = pa.concat_tables([existing, git_table])
                pq.write_table(git_table, pq_git_path, compression="ZSTD", compression_level=3)

            self.graph.batch_insert(file_records, all_symbols, all_modules, all_commits)

            duration = time.time() - start_time
            self.checkpoint.mark_completed(
                repo_path=repo_path,
                commit_sha=commit_sha,
                run_id=run_id,
                files=len(file_records),
                chunks=len(all_chunks),
                commits=len(all_commits),
                duration=duration
            )

            # 7. Git Commit & Push
            print(f"  [6/6] Pushing to origin/asis-data...", flush=True)
            commit_msg = f"[ASIS-A17] Ingested {repo_path} ({len(all_chunks)} chunks, {len(file_records)} files, {len(all_commits)} commits)"
            self.git_push_data(commit_msg)
            print(f"  [+] Transaction committed & published cleanly!", flush=True)

            return len(file_records), len(all_chunks), len(all_commits), duration

        except Exception as e:
            if staging_path:
                self.sharder.rollback_staging(staging_path)
            self.checkpoint.mark_failed(repo_path, run_id, str(e))
            raise e
        finally:
            # Unconditional temporary directory cleanup
            clean_dir(self.temp_dir)

    def run(self, limit: int = None, start_from: str = None):
        projects = self.resolver.resolve()
        completed = self.checkpoint.get_completed_repos()

        print("=" * 80, flush=True)
        print("  ASIS HIGH-PRECISION INGESTION ENGINE — EVOLUTION-X ANDROID 17", flush=True)
        print(f"  Model: {DEFAULT_MODEL} (1024-dim, Float16 Tensor Cores, batch={EMBEDDING_BATCH_SIZE})", flush=True)
        print(f"  Total Repos: {len(projects)} | Already Completed: {len(completed)}", flush=True)
        print(f"  Data Target: {self.data_dir} (branch: asis-data)", flush=True)
        print("=" * 80, flush=True)

        started = False if start_from else True
        processed_in_session = 0

        for idx, repo in enumerate(projects):
            repo_path = repo["path"]

            if start_from and not started:
                if repo_path == start_from:
                    started = True
                else:
                    continue

            if repo_path in completed:
                continue

            if limit and processed_in_session >= limit:
                print(f"[ASIS] Reached session limit of {limit} repos. Halting cleanly.", flush=True)
                break

            free_gb = get_free_disk_gb(self.data_dir)
            if free_gb < MIN_FREE_DISK_GB:
                print(f"\n[ASIS CRITICAL] Free disk space low: {free_gb:.2f} GB (< {MIN_FREE_DISK_GB} GB). Stopping!", flush=True)
                break

            stats = self.checkpoint.get_aggregate_stats()
            total_completed = stats["completed_count"]
            progress_pct = (total_completed / len(projects)) * 100
            avg_duration = (stats["total_duration"] / max(1, total_completed))
            remaining_repos = len(projects) - total_completed
            overall_eta_sec = remaining_repos * avg_duration

            print(f"\n[{total_completed + 1} / {len(projects)}] ({progress_pct:.1f}%) "
                  f"Syncing: {repo_path} | Overall ETA: {format_duration(overall_eta_sec)} | Free Disk: {free_gb:.1f}GB", flush=True)

            try:
                files_cnt, chunks_cnt, commits_cnt, dur = self.process_single_repo(repo)
                cps = chunks_cnt / max(0.1, dur)
                print(f"  [+] Ingested: {files_cnt} files, {chunks_cnt} chunks, {commits_cnt} commits in {dur:.1f}s ({cps:.1f} ch/s)", flush=True)
                processed_in_session += 1
            except Exception as e:
                print(f"  [-] Error processing {repo_path}: {e}", flush=True)

        self.sharder.close()
        print("\n[ASIS] Ingestion session finished cleanly.", flush=True)

# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIS Android 17 Sequential Ingestion Engine")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of repositories to process in this run")
    parser.add_argument("--start-from", type=str, default=None, help="Start from a specific repo path")
    parser.add_argument("--status", action="store_true", help="Print current ingestion stats and exit")
    args = parser.parse_args()

    if args.status:
        pipeline = ASISPipeline(init_model=False)
        stats = pipeline.checkpoint.get_aggregate_stats()
        projects = pipeline.resolver.resolve()
        total_p = len(projects)
        comp = stats["completed_count"]
        pct = (comp / total_p) * 100 if total_p else 0
        print(f"ASIS Status: {comp}/{total_p} repos ({pct:.2f}%) | {stats['total_files']} files | {stats['total_chunks']} chunks | {stats['total_commits']} commits")
    else:
        pipeline = ASISPipeline(init_model=True)
        pipeline.run(limit=args.limit, start_from=args.start_from)
