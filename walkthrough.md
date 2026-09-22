# ASIS Engineering Walkthrough & Validation Report

## 1. System Summary & Milestones Overview

The **Android Source Intelligence System (ASIS)** has been calibrated, deployed, and verified on the 96-core server (`arcane.serverhive.in`).

| Component | Status | Details |
|---|---|---|
| **Environment** | Verified | `/serverhive/shripad/asis_env` (Python 3.12, PyArrow, FastEmbed, SQLite, DTC) |
| **Intra-Op Calibration** | Benchmark Complete | OpenMP scaling degrades past 8T. Optimal efficiency knee: **4 threads/worker** |
| **Model Bake-Off** | Benchmark Complete | `bge-small-en-v1.5` is **5.2x faster**, uses **52% less RAM**, achieves **100% Recall@1/5/10 & 1.0000 MRR** |
| **Process × Threads Matrix** | Benchmark Complete | **24 Workers × 2 Threads** delivers **53.8 chunks/sec** (+126% over 6×8) |
| **Zero-IPC Sharding Engine** | Verified | Workers write slice files directly to disk; master merges binary arrays into FP16 shards |
| **Checkpoint State Machine** | Verified | `manifest.sqlite` tracks status per file; idempotent resumption on reboot |
| **Knowledge Graph** | Hydrated | `asis_graph.db` with 1,914 symbols and 135 Soong/Make build modules |
| **Agent Tool Interface** | Tested | `query_asis.py` executes sub-500ms hybrid search, symbol lookup, and AST retrieval |

---

## 2. Empirical Benchmark Findings

### 2.1 Concurrency Matrix (48-Core Budget)
```
6 Workers × 8 Threads:   23.8 chunks/sec  (4.1 GB RAM, 118k context switches)
12 Workers × 4 Threads:  41.5 chunks/sec  (8.2 GB RAM, 165k context switches)
16 Workers × 3 Threads:  40.0 chunks/sec  (11.0 GB RAM, 200k context switches)
24 Workers × 2 Threads:  53.8 chunks/sec  (11.8 GB RAM, 270k context switches)  <-- PRODUCTION WINNER
```

### 2.2 Model Performance Comparison
```
BAAI/bge-small-en-v1.5:  156.2 chunks/sec |  549 MB RAM | Recall@1: 100% | MRR: 1.0000
nomic-embed-text-v1.5:    30.2 chunks/sec | 1,137 MB RAM | Recall@1: 100% | MRR: 1.0000
```

---

## 3. Production Shards & Storage Layout

Committed cleanly to `/serverhive/shripad/asis_export/`:

```
/serverhive/shripad/asis_export/
├── asis_graph.db                  (484 KB, SQLite WAL mode)
├── manifest.sqlite                (1.2 MB, Checkpoint State Machine)
├── chunks-0000.parquet            (65 KB, 2,388 chunks)
├── chunks-0001.parquet            (42 KB, 1,668 chunks)
├── files-0000.parquet             (35 KB, 500 files)
├── files-0001.parquet             (17 KB, 200 files)
├── modules-0000.parquet           (8.6 KB, parsed Soong modules)
└── embeddings/dense/
    ├── shard-0000.bin             (1.8 MB raw FP16 vectors)
    └── shard-0001.bin             (1.3 MB raw FP16 vectors)
```

---

## 4. Agent Tool Layer Verification

All tests executed live against the sharded storage engine:

### 4.1 Soong Build Dependency Graph
```bash
python /serverhive/shripad/asis_engine/query_asis.py graph frameworks_base_license
```
```json
{
  "module": "frameworks_base_license",
  "type": "license",
  "path": "frameworks/base/Android.bp",
  "shared_libs": []
}
```

### 4.2 Exact Symbol Lookup
```bash
python /serverhive/shripad/asis_engine/query_asis.py find_definition getDisplay
```
```json
[
  {
    "symbol": "getDisplay",
    "rel_path": "frameworks/base/graphics/java/android/graphics/HardwareRenderer.java",
    "start_line": 1666,
    "end_line": 1725,
    "subsystem": "framework",
    "chunk_id": "evox:frameworks:frameworks/base/graphics/java/android/graphics/HardwareRenderer.java:1666-1725:getDisplay"
  }
]
```

### 4.3 Multi-Shard Dense Vector Search
```bash
python /serverhive/shripad/asis_engine/query_asis.py search "window manager display" --top-k 3
```
Successfully retrieved top semantic matches across multiple Parquet shards and binary vector files with cosine similarity scoring.

---

## 5. Full Tree Ingestion Protocol & Timeline

- **Target Codebase**: `/serverhive/shripad/evox` (~1.6M files)
- **Projected Chunks**: ~2.5M – 3.5M code chunks
- **Execution Concurrency**: 24 Isolated Processes × 2 Threads (`OMP_NUM_THREADS=2`)
- **Throughput**: 53.8 chunks/sec (~193,000 chunks/hour)
- **Total Ingestion Duration**: **~14–18 hours** (runs completely unattended with checkpoint/resume)
- **Launch Command**:
```bash
nohup /serverhive/shripad/asis_env/bin/python -u /serverhive/shripad/asis_engine/full_ingest.py \
  --workers 24 --threads 2 > /serverhive/shripad/asis_export/ingest.log 2>&1 &
```
