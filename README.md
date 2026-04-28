# helix

DNA-inspired gradient descent optimizer. Models the molecular geometry of B-form DNA to couple parameter pairs and drive training. Forward-only training mode eliminates the backward pass entirely.

## Install

```bash
go get github.com/tensorwire/helix
```

## Usage

```go
h := helix.NewHelixOptimizer(lr, beta1, beta2, eps, weightDecay)
h.Register(param)
h.PairAT(strand1, strand2)  // A-T bond (2 H-bonds, weaker coupling)
h.PairGC(strand1, strand2)  // G-C bond (3 H-bonds, stronger coupling)
h.Step(step, loss, lr)       // advance one DNA rung (36 degrees)
```

## Key Ideas

- **DNA Geometry**: Each step advances 36 degrees around the double helix. The 6-point rung structure couples paired parameters via hydrogen bond strengths.
- **Immune System**: Checkpoints weights at loss floors. Restores on regression (DNA damage repair). CRISPR memory tracks absolute best.
- **Forward-Only**: Collects signal without a backward pass. The optimizer drives training through the forward path alone.
- **Conductor**: Sparse gradient dispatch — only hot embedding rows get updates.

## Architecture

- `helix.go` — optimizer core: DNA rung geometry, immune checkpoint, Fibonacci stride
- `helix_arena.go` — zero-copy checkpoint/restore via pointer swap
- `helix_accel.go` — Apple AMX acceleration via Accelerate framework
- `helix_accel_stub.go` — pure Go fallback

## Related

- [mongoose](https://github.com/tensorwire/mongoose) — GPU compute engine
- [needle](https://github.com/tensorwire/needle) — fused INT8 kernels for helix

## License

MIT
