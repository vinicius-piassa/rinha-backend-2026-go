# rinha-backend-2026-go

Submission for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) written in **pure Go** — no cgo, no framework, no `.s` files. The AVX2 SIMD in the k-NN hot path is written in Go itself via the experimental `simd/archsimd` intrinsics (`GOEXPERIMENT=simd`, Go 1.26+), compiled inline by the toolchain.

## Architecture

```
client → :9999 (TCP) → lb ──SCM_RIGHTS──▶ server (×2) ──TCP──▶ client
                        │                     │
                        ▼                     ▼
                  epoll accept           epoll recv + HTTP frame
                  round-robin fd          parse → vectorize → IVF k-NN → reply
```

- **`lb`** — TCP listener (epoll, edge-triggered). For each accepted client fd it round-robins a `sendmsg(SCM_RIGHTS)` to one of the API workers over a Unix socket, then closes its own copy. No byte proxying — the API owns the connection end-to-end.
- **`server`** — API worker. Receives client fds from the LB, then a single-thread epoll loop frames HTTP, parses the JSON, vectorizes to a 14-dim `int16` query, routes by partition tag, and runs the IVF k-NN search. Responses are pre-rendered.
- **`build_index`** — offline tool. Parses the reference corpus, runs k-means (k=2048) per partition, builds the IVF index with per-cluster axis-aligned bbox lower bounds, writes the binary files.

## Fraud model

IVF k-NN, K=5 neighbours, partitioned by a 4-bit tag `(card_present<<3 | is_online<<2 | unknown_merchant<<1 | has_last_tx)`. These four are extreme-valued (0 or SCALE) features, so two vectors differing in any of them are far apart — the true top-5 never straddle a partition boundary (validated at 0/5000 verdict changes vs exact full-sweep search). The query routes to its matching sub-index; the fraud verdict is the count of fraud labels among the 5 nearest neighbours.

Per partition: 14-dim `int16` quantized query; phase 1 computes a packed per-cluster bbox squared lower bound (8 clusters per `VPMADDWD`); phase 2 picks the next probe by lowest bound; phase 3 scans the cluster exactly with an early-termination gate; top-5 is a branchless packed `(dist<<22)|idx` array; a repair pass extends to a full sweep only when the initial probe budget leaves the verdict ambiguous.

All SIMD (`Sub`, `DotProductPairs` = VPMADDWD, `Min`/`Max`, `Less`) is `simd/archsimd` — inline AVX2, no call boundary.

## Build & run

```bash
GOEXPERIMENT=simd go build ./cmd/server ./cmd/lb     # requires Go 1.26+
docker compose up -d                                  # 1 lb + 2 api, cgroup-limited
```

`build_index <corpus.json[.gz]> <out.bin> <tag 0..15>` regenerates one partition.

## Resource budget (Rinha 2026)

| Service | CPU | Memory |
|---|---|---|
| `lb` | 0.05 | 8 MB |
| `api1` | 0.475 | 171 MB |
| `api2` | 0.475 | 171 MB |

The hot path is allocation-free and the GC is disabled (`SetGCPercent(-1)` + a `SetMemoryLimit` backstop), so there are no GC pauses at the per-core budget.
