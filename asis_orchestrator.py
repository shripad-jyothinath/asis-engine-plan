#!/usr/bin/env python3
import os
import sys
import time
import json
import stat
import shutil
import hashlib
import sqlite3
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple

# Force UTF-8 terminal output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import mmh3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import xml.etree.ElementTree as ET
from fastembed import TextEmbedding

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
VECTOR_DIM = 1024
PRECISION_DTYPE = np.float32

MANIFEST_BASE_URL = "https://raw.githubusercontent.com/Evolution-X/manifest/cnb/"
MANIFEST_DEFAULT_XML = MANIFEST_BASE_URL + "default.xml"
MANIFEST_SNIPPETS = ["snippets/evolution.xml", "snippets/lineage.xml"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "asis-data-export").resolve()
TEMP_CLONE_DIR = (BASE_DIR.parent / "asis-temp-repo").resolve()

SHARD_CHUNK_LIMIT = 5000
EMBEDDING_BATCH_SIZE = 16
MIN_FREE_DISK_GB = 3.5

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
    ("files_changed", pa.list_(pa.string()))
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

            # Build Clone URL
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

        # Save cache
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(resolved_list, f, indent=2)

        print(f"[ASIS] Resolved {len(resolved_list)} unique Android 17 repositories.")
        return resolved_list

# ==============================================================================
# CHECKPOINT & GRAPH DATABASE MANAGERS
# ==============================================================================

class CheckpointManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
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
                    status TEXT NOT NULL,
                    error_msg TEXT,
                    files_count INTEGER DEFAULT 0,
                    chunks_count INTEGER DEFAULT 0,
                    commits_count INTEGER DEFAULT 0,
                    duration_sec REAL DEFAULT 0,
                    completed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS shard_tracker (
                    shard_id TEXT PRIMARY KEY,
                    chunk_count INTEGER DEFAULT 0,
                    byte_size INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_path TEXT NOT NULL,
                    chunks INTEGER NOT NULL,
                    files INTEGER NOT NULL,
                    commits INTEGER DEFAULT 0,
                    duration_sec REAL NOT NULL,
                    chunks_per_sec REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def get_completed_repos(self) -> set:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT repo_path FROM repo_checkpoints WHERE status = 'COMPLETED'")
            return {row[0] for row in cursor.fetchall()}

    def mark_in_progress(self, repo_path: str, repo_name: str, clone_url: str, revision: str):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO repo_checkpoints (repo_path, repo_name, clone_url, revision, status)
                VALUES (?, ?, ?, ?, 'IN_PROGRESS')
                ON CONFLICT(repo_path) DO UPDATE SET status = 'IN_PROGRESS', error_msg = NULL
            """, (repo_path, repo_name, clone_url, revision))

    def mark_completed(self, repo_path: str, commit_sha: str, files: int, chunks: int, commits: int, duration: float):
        cps = chunks / max(0.1, duration)
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE repo_checkpoints
                SET status = 'COMPLETED', commit_sha = ?, files_count = ?, chunks_count = ?,
                    commits_count = ?, duration_sec = ?, completed_at = CURRENT_TIMESTAMP
                WHERE repo_path = ?
            """, (commit_sha, files, chunks, commits, duration, repo_path))

            conn.execute("""
                INSERT INTO telemetry (repo_path, chunks, files, commits, duration_sec, chunks_per_sec)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (repo_path, chunks, files, commits, duration, cps))

    def mark_failed(self, repo_path: str, error_msg: str):
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE repo_checkpoints
                SET status = 'FAILED', error_msg = ?
                WHERE repo_path = ?
            """, (error_msg, repo_path))

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

class KnowledgeGraphDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    rel_path TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    project TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    category TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    fast_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id TEXT PRIMARY KEY,
                    symbol_name TEXT NOT NULL,
                    symbol_kind TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    subsystem TEXT NOT NULL,
                    FOREIGN KEY(rel_path) REFERENCES files(rel_path)
                );

                CREATE TABLE IF NOT EXISTS build_modules (
                    module_name TEXT PRIMARY KEY,
                    module_type TEXT NOT NULL,
                    def_path TEXT NOT NULL,
                    subsystem TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS git_history (
                    commit_sha TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    project TEXT NOT NULL,
                    author TEXT NOT NULL,
                    commit_date TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT,
                    files_changed_count INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);
                CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(rel_path);
                CREATE INDEX IF NOT EXISTS idx_files_subsystem ON files(subsystem);
                CREATE INDEX IF NOT EXISTS idx_git_author ON git_history(author);
                CREATE INDEX IF NOT EXISTS idx_git_proj ON git_history(project);
            """)

    def batch_insert(self, files: List[Dict], symbols: List[Dict], modules: List[Dict], commits: List[Dict]):
        with self._get_conn() as conn:
            if files:
                conn.executemany("""
                    INSERT OR REPLACE INTO files 
                    (rel_path, repo, project, branch, commit_sha, subsystem, category, size, mtime, fast_hash, content_hash)
                    VALUES (:rel_path, :repo, :project, :branch, :commit_sha, :subsystem, :category, :size, :mtime, :fast_hash, :content_hash)
                """, files)
            if symbols:
                conn.executemany("""
                    INSERT OR REPLACE INTO symbols
                    (symbol_id, symbol_name, symbol_kind, rel_path, start_line, end_line, subsystem)
                    VALUES (:symbol_id, :symbol_name, :symbol_kind, :rel_path, :start_line, :end_line, :subsystem)
                """, symbols)
            if modules:
                conn.executemany("""
                    INSERT OR REPLACE INTO build_modules
                    (module_name, module_type, def_path, subsystem)
                    VALUES (:module_name, :module_type, :def_path, :subsystem)
                """, modules)
            if commits:
                conn.executemany("""
                    INSERT OR REPLACE INTO git_history
                    (commit_sha, repo, project, author, commit_date, subject, body, files_changed_count)
                    VALUES (:commit_sha, :repo, :project, :author, :date, :subject, :body, :files_changed_count)
                """, commits)

# ==============================================================================
# SOFT AST CHUNKER & GIT PARSERS
# ==============================================================================

class SoftASTChunker:
    CODE_EXTS = {
        ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java", ".kt",
        ".rs", ".aidl", ".hal", ".dts", ".dtsi", ".py", ".sh",
        ".bp", ".mk", ".rc", ".xml"
    }

    SYMBOL_PATTERNS = [
        r'(?:public|private|protected|static|inline|virtual|extern)?\s*[\w<>:\[\]]+\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{',
        r'(?:class|struct|interface|enum)\s+([A-Za-z0-9_]+)',
        r'([a-z0-9_]+)\s*\{\s*name:\s*"([^"]+)"',
        r'LOCAL_MODULE\s*:=\s*([A-Za-z0-9_]+)'
    ]

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
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return [], [], []

        if not lines:
            return [], [], []

        total_lines = len(lines)
        chunks = []
        symbols = []
        modules = []
        subsystem = cls.get_subsystem(rel_path)

        if full_path.name in ("Android.bp", "Android.mk") or full_path.suffix == ".mk":
            content = "".join(lines)
            import re
            for match in re.finditer(r'([a-z0-9_]+)\s*\{\s*name:\s*"([^"]+)"', content):
                modules.append({
                    "module_name": match.group(2),
                    "module_type": match.group(1),
                    "def_path": rel_path,
                    "subsystem": subsystem
                })
            for match in re.finditer(r'LOCAL_MODULE\s*:=\s*([A-Za-z0-9_]+)', content):
                modules.append({
                    "module_name": match.group(1),
                    "module_type": "makefile",
                    "def_path": rel_path,
                    "subsystem": subsystem
                })

        target_lines = 60
        max_lines = 100
        overlap = 15

        start = 0
        while start < total_lines:
            end = min(start + target_lines, total_lines)

            snap_window = lines[max(0, end - 8): min(total_lines, end + 8)]
            for offset, line in enumerate(snap_window):
                stripped = line.strip()
                if stripped.endswith("}") or stripped.endswith(";"):
                    end = max(0, end - 8) + offset + 1
                    break

            chunk_lines = lines[start:end]
            chunk_text = "".join(chunk_lines).strip()

            if chunk_text:
                symbol_name = ""
                symbol_kind = "block"
                import re
                for pat in cls.SYMBOL_PATTERNS[:2]:
                    m = re.search(pat, chunk_text)
                    if m:
                        symbol_name = m.group(1)
                        symbol_kind = "class" if "class" in m.group(0) or "struct" in m.group(0) else "function"
                        break

                chunk_id = f"evox:{repo_path}:{rel_path}:{start+1}-{end}:{symbol_name or 'block'}"
                chunks.append({
                    "chunk_id": chunk_id,
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
                    symbols.append({
                        "symbol_id": f"{rel_path}:{start+1}:{symbol_name}",
                        "symbol_name": symbol_name,
                        "symbol_kind": symbol_kind,
                        "rel_path": rel_path,
                        "start_line": start + 1,
                        "end_line": end,
                        "subsystem": subsystem
                    })

            if end >= total_lines:
                break
            start = max(start + 1, end - overlap)

        return chunks, symbols, modules

    @classmethod
    def extract_git_commits(cls, repo_dir: Path, repo_path: str) -> List[Dict]:
        cmd = 'git log --pretty=format:"COMMIT_REC%H%x1f%an%x1f%ad%x1f%s%x1f%b%x1e" --name-only'
        res = subprocess.run(cmd, cwd=str(repo_dir), shell=True, capture_output=True, text=True, errors="replace")
        if res.returncode != 0 or not res.stdout:
            return []

        commits = []
        records = res.stdout.split("COMMIT_REC")
        for rec in records:
            if not rec.strip():
                continue
            parts = rec.split("\x1e")
            meta_part = parts[0]
            files_part = parts[1] if len(parts) > 1 else ""

            fields = meta_part.split("\x1f")
            if len(fields) >= 4:
                sha = fields[0].strip()
                author = fields[1].strip()
                date_str = fields[2].strip()
                subject = fields[3].strip()
                body = fields[4].strip() if len(fields) > 4 else ""
                files = [f.strip() for f in files_part.splitlines() if f.strip()]

                commits.append({
                    "commit_sha": sha,
                    "repo": "evox",
                    "project": repo_path,
                    "author": author,
                    "date": date_str,
                    "subject": subject,
                    "body": body,
                    "files_changed_count": len(files),
                    "files_changed": files
                })
        return commits

# ==============================================================================
# HIGH-PRECISION DENSE SHARD WRITER
# ==============================================================================

class ShardWriter:
    def __init__(self, export_dir: Path, model_name: str = DEFAULT_MODEL):
        self.export_dir = export_dir
        self.dense_dir = export_dir / "embeddings" / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ASIS] Initializing FastEmbed model: {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        self.current_shard_idx = self._find_latest_shard_index()
        self.current_shard_chunks = 0
        self.shard_bin_file = None
        self._open_shard(self.current_shard_idx)

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
        self.current_offset = self.shard_bin_file.tell()

    def rotate_shard_if_needed(self):
        if self.current_shard_chunks >= SHARD_CHUNK_LIMIT:
            self.current_shard_idx += 1
            self._open_shard(self.current_shard_idx)
            self.current_shard_chunks = 0

    def embed_and_store(self, chunks: List[Dict], repo_info: Dict[str, Any], commit_sha: str) -> List[Dict]:
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        vectors_gen = self.model.embed(texts, batch_size=EMBEDDING_BATCH_SIZE)

        enriched_chunks = []
        shard_id = self.current_shard_path.name

        for chunk, vector in zip(chunks, vectors_gen):
            self.rotate_shard_if_needed()
            shard_id = self.current_shard_path.name
            offset = self.current_offset

            # High precision float32 raw binary write
            vec_bytes = np.array(vector, dtype=PRECISION_DTYPE).tobytes()
            self.shard_bin_file.write(vec_bytes)
            self.current_offset += len(vec_bytes)
            self.current_shard_chunks += 1

            chunk["repo"] = "evox"
            chunk["project"] = repo_info["path"]
            chunk["branch"] = "cnb"
            chunk["commit_sha"] = commit_sha
            chunk["fast_hash"] = f"{mmh3.hash64(chunk['text'])[0]:x}"
            chunk["content_hash"] = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            chunk["embedding_shard_id"] = shard_id
            chunk["embedding_offset"] = offset
            enriched_chunks.append(chunk)

        self.shard_bin_file.flush()
        return enriched_chunks

    def close(self):
        if self.shard_bin_file and not self.shard_bin_file.closed:
            self.shard_bin_file.close()

# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

class ASISPipeline:
    def __init__(self, data_dir: Path = DATA_DIR, temp_dir: Path = TEMP_CLONE_DIR):
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.resolver = ManifestResolver()
        self.checkpoint = CheckpointManager(self.data_dir / "manifest.sqlite")
        self.graph = KnowledgeGraphDB(self.data_dir / "asis_graph.db")
        self.sharder = ShardWriter(self.data_dir)

    def git_clone_repo(self, clone_url: str, revision: str) -> str:
        clean_dir(self.temp_dir)
        branch = revision.replace("refs/heads/", "").replace("refs/tags/", "")
        
        # 1. Clone with full branch commits history
        ntfs_flag = "-c core.protectNTFS=false" if os.name == 'nt' else ""
        cmd = f'git clone {ntfs_flag} --single-branch --branch "{branch}" "{clone_url}" "{self.temp_dir}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        sha_res = subprocess.run('git rev-parse HEAD', cwd=str(self.temp_dir), shell=True, capture_output=True, text=True)
        if sha_res.returncode == 0 and sha_res.stdout.strip():
            return sha_res.stdout.strip()

        # 2. Fallback to default clone if specific branch name not found
        clean_dir(self.temp_dir)
        cmd = f'git clone {ntfs_flag} --single-branch "{clone_url}" "{self.temp_dir}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        sha_res = subprocess.run('git rev-parse HEAD', cwd=str(self.temp_dir), shell=True, capture_output=True, text=True)
        if sha_res.returncode == 0 and sha_res.stdout.strip():
            return sha_res.stdout.strip()

        clean_dir(self.temp_dir)
        raise RuntimeError(f"Git clone failed: {res.stderr.strip()}")

    def git_push_data(self, commit_msg: str):
        try:
            with sqlite3.connect(self.data_dir / "manifest.sqlite") as c:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            with sqlite3.connect(self.data_dir / "asis_graph.db") as c:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass

        subprocess.run('git add -A', cwd=str(self.data_dir), shell=True, capture_output=True)
        c_res = subprocess.run(f'git commit -m "{commit_msg}"', cwd=str(self.data_dir), shell=True, capture_output=True, text=True)
        
        # Retry push up to 5 times
        for attempt in range(1, 6):
            p = subprocess.run('git push origin asis-data', cwd=str(self.data_dir), shell=True, capture_output=True, text=True)
            if p.returncode == 0:
                return
            print(f"[ASIS] Push attempt {attempt}/5 failed: {p.stderr.strip()}. Retrying in {attempt * 3}s...")
            time.sleep(attempt * 3)

    def process_single_repo(self, repo: Dict[str, Any]) -> Tuple[int, int, int, float]:
        repo_path = repo["path"]
        repo_name = repo["name"]
        clone_url = repo["clone_url"]
        revision = repo["revision"]

        start_time = time.time()
        self.checkpoint.mark_in_progress(repo_path, repo_name, clone_url, revision)

        # 1. Clone repository with full branch commits
        commit_sha = self.git_clone_repo(clone_url, revision)

        # 2. Extract Git Commits History
        all_commits = SoftASTChunker.extract_git_commits(self.temp_dir, repo_path)

        # 3. Scan & parse files
        all_chunks = []
        all_symbols = []
        all_modules = []
        file_records = []

        for root, _, files in os.walk(self.temp_dir):
            if ".git" in root:
                continue
            for f in files:
                fpath = Path(root) / f
                if SoftASTChunker.should_index(fpath):
                    rel = str(fpath.relative_to(self.temp_dir)).replace("\\", "/")
                    stat_info = fpath.stat()

                    with open(fpath, "rb") as bf:
                        content_bytes = bf.read()
                        fast_h = f"{mmh3.hash64(content_bytes)[0]:x}"
                        content_h = hashlib.sha256(content_bytes).hexdigest()

                    file_records.append({
                        "rel_path": f"{repo_path}/{rel}",
                        "repo": "evox",
                        "project": repo_path,
                        "branch": "cnb",
                        "commit_sha": commit_sha,
                        "subsystem": SoftASTChunker.get_subsystem(rel),
                        "category": "tier_1",
                        "size": stat_info.st_size,
                        "mtime": stat_info.st_mtime,
                        "fast_hash": fast_h,
                        "content_hash": content_h
                    })

                    c, s, m = SoftASTChunker.chunk_file(fpath, f"{repo_path}/{rel}", repo_path)
                    all_chunks.extend(c)
                    all_symbols.extend(s)
                    all_modules.extend(m)

        # 4. Vectorize chunks with BGE-Large FP32
        stored_chunks = self.sharder.embed_and_store(all_chunks, repo, commit_sha)

        # 5. Save Parquet Shards
        p_idx = self.sharder.current_shard_idx
        if stored_chunks:
            chunk_table = pa.Table.from_pylist(stored_chunks, schema=CHUNK_SCHEMA)
            pq_chunk_path = self.data_dir / f"chunks-{p_idx:04d}.parquet"
            pq.write_table(chunk_table, pq_chunk_path, compression="ZSTD", compression_level=7)

        if file_records:
            file_table = pa.Table.from_pylist(file_records, schema=FILE_SCHEMA)
            pq_file_path = self.data_dir / f"files-{p_idx:04d}.parquet"
            pq.write_table(file_table, pq_file_path, compression="ZSTD", compression_level=7)

        if all_commits:
            git_table = pa.Table.from_pylist(all_commits, schema=GIT_SCHEMA)
            pq_git_path = self.data_dir / f"git_history-{p_idx:04d}.parquet"
            pq.write_table(git_table, pq_git_path, compression="ZSTD", compression_level=7)

        # 6. Populate SQLite Knowledge Graph
        self.graph.batch_insert(file_records, all_symbols, all_modules, all_commits)

        # 7. Delete temp clone directory immediately
        clean_dir(self.temp_dir)

        duration = time.time() - start_time
        self.checkpoint.mark_completed(repo_path, commit_sha, len(file_records), len(all_chunks), len(all_commits), duration)

        # 8. Git commit & push
        commit_msg = f"[ASIS-A17] Ingested {repo_path} ({len(all_chunks)} chunks, {len(file_records)} files, {len(all_commits)} commits)"
        self.git_push_data(commit_msg)

        return len(file_records), len(all_chunks), len(all_commits), duration

    def run(self, limit: int = None, start_from: str = None):
        projects = self.resolver.resolve()
        completed = self.checkpoint.get_completed_repos()

        print("=" * 80)
        print("  ASIS HIGH-PRECISION INGESTION ENGINE — EVOLUTION-X ANDROID 17")
        print(f"  Model: {DEFAULT_MODEL} (1024-dim, Float32)")
        print(f"  Total Repos: {len(projects)} | Already Completed: {len(completed)}")
        print(f"  Data Target: {self.data_dir} (branch: asis-data)")
        print("=" * 80)

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
                print(f"[ASIS] Reached session limit of {limit} repos. Halting cleanly.")
                break

            free_gb = get_free_disk_gb(self.data_dir)
            if free_gb < MIN_FREE_DISK_GB:
                print(f"\n[ASIS CRITICAL] Free disk space low: {free_gb:.2f} GB (< {MIN_FREE_DISK_GB} GB). Stopping to avoid out-of-disk error!")
                break

            stats = self.checkpoint.get_aggregate_stats()
            total_completed = stats["completed_count"]
            progress_pct = (total_completed / len(projects)) * 100
            avg_duration = (stats["total_duration"] / max(1, total_completed))
            remaining_repos = len(projects) - total_completed
            overall_eta_sec = remaining_repos * avg_duration

            print(f"\n[{total_completed + 1} / {len(projects)}] ({progress_pct:.1f}%) "
                  f"Syncing: {repo_path} | Overall ETA: {format_duration(overall_eta_sec)} | Free Disk: {free_gb:.1f}GB")

            try:
                files_cnt, chunks_cnt, commits_cnt, dur = self.process_single_repo(repo)
                cps = chunks_cnt / max(0.1, dur)
                print(f"  [+] Ingested: {files_cnt} files, {chunks_cnt} chunks, {commits_cnt} commits in {dur:.1f}s ({cps:.1f} chunks/s) -> Pushed & Deleted")
                processed_in_session += 1
            except Exception as e:
                print(f"  [-] Error processing {repo_path}: {e}")
                self.checkpoint.mark_failed(repo_path, str(e))
                clean_dir(self.temp_dir)

        self.sharder.close()
        print("\n[ASIS] Ingestion session finished cleanly.")

# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIS Android 17 Sequential Ingestion Engine")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of repositories to process in this run")
    parser.add_argument("--start-from", type=str, default=None, help="Start from a specific repo path")
    parser.add_argument("--status", action="store_true", help="Print current ingestion stats and exit")
    args = parser.parse_args()

    pipeline = ASISPipeline()

    if args.status:
        stats = pipeline.checkpoint.get_aggregate_stats()
        projects = pipeline.resolver.resolve()
        total_p = len(projects)
        comp = stats["completed_count"]
        pct = (comp / total_p) * 100 if total_p else 0
        print(f"ASIS Status: {comp}/{total_p} repos ({pct:.2f}%) | {stats['total_files']} files | {stats['total_chunks']} chunks | {stats['total_commits']} commits | Shards: {pipeline.sharder.current_shard_idx + 1}")
    else:
        pipeline.run(limit=args.limit, start_from=args.start_from)
