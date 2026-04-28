# CLAUDE.md — Helix

## What This Is

DNA-inspired gradient descent optimizer. Models the molecular geometry of B-form DNA (Watson & Crick, 1953) to couple parameter pairs and drive training without softmax. Forward-only training mode eliminates the backward pass entirely.

## Build

```bash
CGO_ENABLED=1 go build ./...   # Accelerate-accelerated on macOS
CGO_ENABLED=0 go build ./...   # pure Go
go test -v ./...
```

## Architecture

- `helix.go` — HelixOptimizer: DNA rung geometry, immune system (checkpoint/restore/CRISPR), Fibonacci stride, elliptic curve modulation (secp256k1), signal chain conductivity, forward-only training
- `helix_conductor.go` — Conductor: tracks active embedding rows via charge decay, sparse gradient dispatch
- `helix_arena.go` — HelixArena: zero-copy checkpoint/restore via pointer swap
- `helix_accel.go` — Accelerate framework (Apple AMX) accelerated DNA step and Adam step
- `helix_accel_stub.go` — fallback for non-macOS

## Key Concepts

**DNA Geometry:** Each optimizer step advances 36° around the double helix (10 base pairs per turn). The 6-point Rung structure (Backbone1, Glyco1, Hbond1, Hbond2, Glyco2, Backbone2) couples paired parameters via hydrogen bond strengths: A↔T (2 bonds, weaker) and G↔C (3 bonds, stronger).

**Immune System:** Checkpoints healthy weights at loss floors. If loss worsens beyond threshold, restores to checkpoint (like DNA damage repair). CRISPR memory tracks the absolute best floor ever observed.

**Forward-Only:** `ForwardOnlyStep()` collects signal without updating weights. `PrepareStep()` returns the Rung for GPU kernel dispatch. Eliminates the backward pass — the optimizer drives training through the forward path alone.

**Conductor:** Tracks which embedding rows are "hot" (recently observed in training data). Only hot rows get gradient updates — 49 active rows instead of 4M threads.

## Key Types

```go
h := helix.NewHelixOptimizer(lr, beta1, beta2, eps, wd)
h.PairAT(strand1, strand2)  // couple with A↔T bond (2 H-bonds)
h.PairGC(strand1, strand2)  // couple with G↔C bond (3 H-bonds)
h.Step(step, loss, lr)       // one DNA rung advance

c := helix.NewConductor(vocabSize, window)
c.Observe(tokens)
hotRows := c.HotRows()
```

## Related Packages

- `github.com/tensorwire/mongoose` — GPU compute engine
- `github.com/tensorwire/needle` — Fused INT8 kernels (performance layer for helix)
