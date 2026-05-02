package helix

import "math"

// helixSqrt computes sqrt(x) in pure float32 via fast inverse sqrt.
func helixSqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}
	i := math.Float32bits(x)
	i = 0x5f375a86 - (i >> 1)
	y := math.Float32frombits(i)
	y = y * (1.5 - (x*0.5)*y*y)
	y = y * (1.5 - (x*0.5)*y*y)
	return x * y
}

// =============================================================================
// DNA Gradient Descent
// =============================================================================
//
// A novel optimizer modeled on the molecular geometry of B-form DNA.
//
// Real DNA structure (B-form, Watson & Crick, 1953):
//
//   Diameter:           20 Å
//   Rise per base pair: 3.4 Å
//   Base pairs per turn: 10
//   Twist per step:     36° (2π/10 = 0.6283 rad)
//   Helix angle:        arctan(34 / (π·20)) = 28.4° = 0.496 rad
//   Major groove:       22 Å (where proteins read the code)
//   Minor groove:       12 Å (structural stabilization)
//   Groove ratio:       22/12 = 1.833:1
//
// Each rung (base pair) of the ladder has 6 connection points:
//
//   [Strand 1 backbone]──[glycosidic]──[base···H-bonds···base]──[glycosidic]──[Strand 2 backbone]
//        ①                    ②              ③ ④ (⑤)                ⑤(⑥)           ⑥
//
//   ① Strand 1 backbone (phosphodiester) — anchor: weight value
//   ② Strand 1 glycosidic bond           — signal: weight's gradient
//   ③ Hydrogen bond 1                    — coupling: momentum transfer
//   ④ Hydrogen bond 2                    — coupling: velocity transfer
//   ⑤ Strand 2 glycosidic bond           — signal: paired weight's gradient
//   ⑥ Strand 2 backbone (phosphodiester) — anchor: paired weight value
//
// Base pairing rules encode bond strength:
//   A↔T (adenine-thymine):  2 hydrogen bonds — weaker coupling (more independence)
//   G↔C (guanine-cytosine): 3 hydrogen bonds — stronger coupling (tight co-evolution)
//
// In parameter space:
//   - wq ↔ wk: A↔T pairing (query-key need independence to learn different projections)
//   - gate ↔ up: G↔C pairing (SwiGLU gate and up must co-evolve tightly)
//   - bq ↔ bk: A↔T pairing (bias coupling, looser)
//
// The gradient flows THROUGH the rung, not around the helix:
//   backbone → glycosidic → H-bonds → glycosidic → backbone
//
// Each optimizer step advances one base pair along the helix (36° twist),
// and the 6-point rung structure computes the parameter update.
//
// The major/minor groove asymmetry (1.833:1) creates a dominant gradient
// channel (major groove = primary update direction) and a stabilizing
// channel (minor groove = regularization).

// =============================================================================
// DNA geometry constants — not hyperparameters, physical reality
// =============================================================================

const (
	dnaBasePairsPerTurn = 10
	dnaTwistPerStep     = 2.0 * math.Pi / dnaBasePairsPerTurn // 36° = 0.6283 rad
	dnaHelixAngle       = 0.496                                // arctan(34/(π·20)) rad = 28.4°
	dnaDiameter         = 20.0                                 // Å (normalized to 1.0 in param space)
	dnaRisePerBP        = 3.4                                  // Å per base pair
	dnaMajorGroove      = 22.0                                 // Å
	dnaMinorGroove      = 12.0                                 // Å
	dnaGrooveRatio      = dnaMajorGroove / dnaMinorGroove      // 1.833:1
)

// BondStrength encodes the hydrogen bond count between base pairs.
type BondStrength int

const (
	// AT has 2 hydrogen bonds — weaker coupling, more parameter independence.
	// Use for pairs that need to learn different representations (wq↔wk).
	AT BondStrength = 2

	// GC has 3 hydrogen bonds — stronger coupling, tighter co-evolution.
	// Use for pairs that must stay synchronized (gate↔up in SwiGLU).
	GC BondStrength = 3
)

// HelixOptimizer implements DNA gradient descent.
type HelixOptimizer struct {
	lr          float32
	beta1       float32
	beta2       float32
	eps         float32
	weightDecay float32

	pairs   []helixPair
	singles []HelixParam

	// Helix phase — current angular position on the double helix.
	// Advances by dnaTwistPerStep (36°) each optimizer step.
	phase float64

	// Elliptic curve for non-repeating phase modulation.
	// y² = x³ + 7 (secp256k1 form). Prevents phase-locking
	// by adding a deterministic but non-periodic perturbation.
	curveX float64
	curveY float64

	// Loss landscape tracking for adaptive groove weighting
	prevLoss     float64
	lossMomentum float64 // EMA of |dL/dt|

	// === Immune system ===
	//
	// When the optimizer contacts a new loss floor (virus), the immune
	// response activates:
	//   1. Kill:    snapshot the floor value (antibody captures the virus shape)
	//   2. Collect: rewind weights to the pre-contact checkpoint (healthy DNA)
	//   3. CRISPR:  store the floor in memory (the DNA now knows it's reachable)
	//   4. Resume:  keep exploring from healthy state, primed with knowledge
	//
	// The key insight: the minimum and the gradient AT the minimum are different.
	// The value 1.94 is real. The gradient there is the virus — it's what pushed
	// the loss back up. We keep the approach trajectory, not the exit.

	// Checkpoint: the last known healthy state (weights just before the floor)
	checkpoint       [][]float32 // deep copy of all param data at checkpoint
	checkpointLoss   float32     // loss at checkpoint time
	checkpointStep   int         // step when checkpoint was taken

	// CRISPR memory: the best floor ever seen
	bestFloor        float32     // lowest loss ever observed
	bestFloorStep    int         // when it was observed

	// Immune state
	immuneActive     bool        // true when we've detected a floor and are monitoring
	preFloorLoss     float32     // loss just before we hit the floor
	floorContactStep int         // step when floor was first contacted
	recoveryCount    int         // how many times we've rewound from this floor

	// === Fibonacci stride (golden spiral exploration) ===
	//
	// The step increment follows the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13...
	// On conductive paths, stride grows — covering more ground each step.
	// On immune rewind, stride resets to (1, 1) — fresh exploration from scratch.
	//
	// The golden ratio (φ = 1.618...) emerges naturally as fib[n]/fib[n-1] → φ.
	// This is the same geometry as DNA's groove ratio (1.833), galaxy arms,
	// sunflower seeds — nature's optimal space-filling spiral.
	//
	// Three fractal scales breathe simultaneously:
	//   - Base pair level: individual step stride (fibA, fibB)
	//   - Turn level: groove weight modulation every 10 bp (fibTurnA, fibTurnB)
	//   - Supercoil level: immune sensitivity every 100 bp (fibCoilA, fibCoilB)
	fibA, fibB           int // base pair stride: advance by fibB base pairs
	fibTurnA, fibTurnB   int // turn-level Fibonacci
	fibCoilA, fibCoilB   int // supercoil-level Fibonacci
	fibStepsOnConductor  int // consecutive conductive steps (legacy, unused with EMA)
	condEMA              float64 // EMA of conductivity for Fibonacci stride decisions

	// Disk checkpoint callback — set by the training loop.
	// Called at most once per checkpointInterval steps.
	onCheckpoint         func(step int, loss float32, bestFloor float32)
	checkpointInterval   int // minimum steps between disk writes
	lastDiskCheckpoint   int // step of last disk write

	// Arena for zero-copy checkpoint/restore via pointer swap.
	arena     *HelixArena
	arenaSync func() // called before swap to ensure async GPU work is done

	// Restore callback — called after arena swap on restore.
	// Mode 4 uses this to write arena data back into graph variables.
	onRestore func()

	// Last computed rung + signal — exported for CUDA kernel dispatch
	lastRung         Rung
	lastSignalScale  float32 // +1 = improving, -ratio = worsening

	// Thresholds
	floorWindow      int         // steps to wait after floor contact before judging
	maxRecoveries    int         // max rewinds before accepting the state
	rewindThreshold  float32     // fractional rebound that triggers rewind (0.15 = 15%)

	// === Signal chain (charge transfer through the pi stack) ===
	//
	// Real DNA conducts electricity through overlapping pi orbitals of
	// stacked base pairs. Charge propagates along the helix axis.
	// Conductivity depends on sequence — A-T vs G-C, mismatches disrupt.
	//
	// In the optimizer: each base pair (step) records its loss and gradient
	// direction into the signal chain. The signal propagates through the
	// stack with distance-dependent attenuation (closer rungs = stronger signal).
	//
	// The signal chain is a circular buffer of the last N rungs. Each rung
	// stores a condensed "charge" — the loss value and a hash of the gradient
	// direction. The optimizer reads the full chain to find conductive paths
	// (directions that consistently led downward across multiple rungs).

	signal       []signalRung    // circular buffer, one per base pair
	signalHead   int             // write position in circular buffer
	signalLen    int             // how many rungs have been written
	signalCap    int             // buffer capacity (number of rungs to remember)
}

// signalRung is one rung of the signal chain — the charge stored at one base pair.
// Records the loss, gradient magnitude, and a directional fingerprint.
type signalRung struct {
	loss     float32   // loss at this step
	gradNorm float32   // global gradient norm (signal strength)
	dLoss    float32   // loss change from previous rung (current direction)
	phase    float64   // helix phase at this rung
}

// HelixParam is the interface for parameters in the helix optimizer.
type HelixParam interface {
	GradHelix() []float32
	DataHelix() []float32
	MomHelix() []float32
	VelHelix() []float32
	SizeHelix() int
}

type helixPair struct {
	strand1  HelixParam
	strand2  HelixParam
	strength BondStrength
}

// SimpleHelixParam wraps raw slices for the helix optimizer.
type SimpleHelixParam struct {
	D, G, M, V []float32
	N           int
}

func (p *SimpleHelixParam) GradHelix() []float32 { return p.G }
func (p *SimpleHelixParam) DataHelix() []float32 { return p.D }
func (p *SimpleHelixParam) MomHelix() []float32  { return p.M }
func (p *SimpleHelixParam) VelHelix() []float32  { return p.V }
func (p *SimpleHelixParam) SizeHelix() int       { return p.N }

// NewHelixOptimizer creates a DNA gradient descent optimizer.
func NewHelixOptimizer(lr, beta1, beta2, eps, weightDecay float32) *HelixOptimizer {
	return &HelixOptimizer{
		lr:            lr,
		beta1:         beta1,
		beta2:         beta2,
		eps:           eps,
		weightDecay:   weightDecay,
		phase:         0,
		curveX:        1.0,
		curveY:        math.Sqrt(8.0), // y = sqrt(x³ + 7) at x=1
		bestFloor:      math.MaxFloat32,
		checkpointLoss: math.MaxFloat32,
		floorWindow:     10,
		maxRecoveries:   3,
		rewindThreshold: 0.15,
		signalCap:           100,
		signal:              make([]signalRung, 100),
		checkpointInterval:  88, // double helix 🧬
		fibA: 1, fibB: 1,         // base pair: start at (1, 1)
		fibTurnA: 1, fibTurnB: 1, // turn level
		fibCoilA: 1, fibCoilB: 1, // supercoil level
	}
}

// OnCheckpoint sets a callback that fires every time the immune system
// snapshots healthy weights. Use this to persist checkpoints to disk.
func (h *HelixOptimizer) OnCheckpoint(fn func(step int, loss float32, bestFloor float32)) {
	h.onCheckpoint = fn
}

// SetArena enables zero-copy checkpoint/restore via pointer swapping.
// syncFn is called before each swap to ensure async GPU work is done.
func (h *HelixOptimizer) SetArena(a *HelixArena, syncFn func()) {
	h.arena = a
	h.arenaSync = syncFn
}

// OnRestore sets a callback that fires after the arena swap on restore.
// Used by mode 4 to write restored weights back into graph variables.
func (h *HelixOptimizer) OnRestore(fn func()) {
	h.onRestore = fn
}

// PairAT couples two parameters with A↔T bond strength (2 hydrogen bonds).
// Weaker coupling — parameters maintain more independence.
// Use for: wq↔wk, bq↔bk
func (h *HelixOptimizer) PairAT(strand1, strand2 HelixParam) {
	h.pairs = append(h.pairs, helixPair{strand1, strand2, AT})
}

// PairGC couples two parameters with G↔C bond strength (3 hydrogen bonds).
// Stronger coupling — parameters co-evolve tightly.
// Use for: gate↔up (SwiGLU)
func (h *HelixOptimizer) PairGC(strand1, strand2 HelixParam) {
	h.pairs = append(h.pairs, helixPair{strand1, strand2, GC})
}

// Register adds an unpaired parameter (single strand, no base pairing).
func (h *HelixOptimizer) Register(p HelixParam) {
	h.singles = append(h.singles, p)
}

// Step runs one optimizer step — one base pair advance along the double helix.
func (h *HelixOptimizer) Step(step int, loss float32, lr float32) {
	// === Immune system: checkpoint and floor detection ===
	rewound := h.immuneResponse(step, loss)
	if rewound {
		return // weights restored to checkpoint, skip this step's gradient update
	}

	// Clip gradients globally (max norm = 1.0)
	var gradNorm float32
	if useHelixAccel {
		gradNorm = helixClipGradsAccel(h, 1.0)
	} else {
		gradNorm = h.clipGradients(1.0)
	}

	// Track loss momentum for immune system
	h.updateLossMomentum(float64(loss))

	// Record this rung into the signal chain
	h.recordSignal(loss, gradNorm)

	// Read signal chain conductivity to decide Fibonacci stride
	conductivity := h.SignalConductivity()

	// EMA Fibonacci stride: low-pass filter on conductivity.
	// α=0.1 smooths over ~10 steps, ignoring step-to-step noise.
	h.condEMA = 0.1*conductivity + 0.9*h.condEMA
	if h.condEMA > 0.005 {
		h.fibA, h.fibB = h.fibB, h.fibA+h.fibB
		if h.fibB > 21 { h.fibB = 21 }
	} else if h.condEMA < -0.005 {
		if h.fibB > 1 {
			h.fibA, h.fibB = 1, h.fibA
			if h.fibA < 1 { h.fibA = 1 }
		}
	}

	// Advance helix phase by Fibonacci stride × base pair twist
	stride := float64(h.fibB)
	h.phase += dnaTwistPerStep * stride

	// Fractal breathing: turn-level Fibonacci modulates groove emphasis
	basePairNum := int(h.phase / dnaTwistPerStep)
	if basePairNum > 0 && basePairNum%dnaBasePairsPerTurn == 0 {
		// Completed a full turn — advance turn-level Fibonacci
		h.fibTurnA, h.fibTurnB = h.fibTurnB, h.fibTurnA+h.fibTurnB
		if h.fibTurnB > 21 { h.fibTurnB = 21 }
	}
	if basePairNum > 0 && basePairNum%(dnaBasePairsPerTurn*10) == 0 {
		// Completed 10 turns — advance supercoil-level Fibonacci
		h.fibCoilA, h.fibCoilB = h.fibCoilB, h.fibCoilA+h.fibCoilB
		if h.fibCoilB > 21 { h.fibCoilB = 21 }
	}

	// Advance elliptic curve for phase modulation
	h.advanceCurve()
	curvePhase := math.Atan2(h.curveY, h.curveX)

	// Groove weights from signal chain, modulated by fractal Fibonacci scales
	majorWeight, minorWeight := h.grooveWeightsFromSignal()

	// Fractal modulation: golden ratio breathing across scales.
	// φ = (1+√5)/2 ≈ 1.618. Each Fibonacci scale adds a harmonic.
	phi := (1.0 + math.Sqrt(5.0)) / 2.0
	turnBreath := math.Sin(float64(h.fibTurnB) * h.phase / phi)
	coilBreath := math.Sin(float64(h.fibCoilB) * h.phase / (phi * phi))
	// Blend: subtle modulation, ±10% at turn level, ±5% at supercoil level
	majorWeight *= 1.0 + 0.10*turnBreath + 0.05*coilBreath
	minorWeight *= 1.0 - 0.05*turnBreath + 0.10*coilBreath

	// Bias correction
	bc1 := float32(1.0 - math.Pow(float64(h.beta1), float64(step)))
	bc2 := float32(1.0 - math.Pow(float64(h.beta2), float64(step)))

	// The 6-point rung structure at this phase angle
	rung := h.computeRung(curvePhase, majorWeight, minorWeight)
	h.lastRung = rung

	// Update paired parameters through the DNA rung
	if useHelixAccel {
		for _, pair := range h.pairs {
			helixDNAStepAccel(pair, rung, lr, bc1, bc2, h)
		}
		for _, p := range h.singles {
			helixAdamStepAccel(p, lr, bc1, bc2, h)
		}
	} else {
		for _, pair := range h.pairs {
			h.dnaStep(pair, rung, lr, bc1, bc2)
		}
		for _, p := range h.singles {
			h.adamStep(p, lr, bc1, bc2)
		}
	}
}

// immuneResponse implements the cellular immune system.
//
// The optimizer maintains a rolling checkpoint of "healthy" weights.
// When a new loss floor is hit (virus contact), the immune system activates:
//
//   1. Snapshot the floor value (antibody captures virus shape)
//   2. Monitor for floorWindow steps — if loss rebounds significantly,
//      the gradient at the floor was the virus
//   3. Rewind to the pre-floor checkpoint (excise the bad DNA)
//   4. Store the floor in CRISPR memory (we know it's reachable)
//   5. Resume exploration from healthy state
//
// Returns true if a rewind occurred (caller should skip gradient update).
func (h *HelixOptimizer) immuneResponse(step int, loss float32) bool {
	// --- Update rolling checkpoint every N steps during healthy exploration ---
	// Checkpoint when loss is stable or improving (no active immune response)
	if !h.immuneActive && step > 1 && loss > 0 {
		// Take checkpoint when current loss is within 10% of best seen
		if h.checkpoint == nil || loss < h.checkpointLoss*1.1 {
			h.checkpointLoss = loss
			h.checkpointStep = step
			h.saveCheckpoint()
		}
	}

	// --- Floor detection: did we just hit a new minimum? ---
	// Skip zero loss (uninitialized first step)
	if loss > 0 && loss < h.bestFloor {
		h.bestFloor = loss
		h.bestFloorStep = step

		if !h.immuneActive {
			// First contact with this floor — activate immune monitoring
			h.immuneActive = true
			h.preFloorLoss = h.checkpointLoss
			h.floorContactStep = step
			h.recoveryCount = 0
		}
	}

	// --- Immune monitoring: watch for rebound (virus damage) ---
	if h.immuneActive {
		stepsSinceContact := step - h.floorContactStep

		if stepsSinceContact >= h.floorWindow {
			// Judgment time: did the loss rebound?
			rebound := loss - h.bestFloor
			threshold := h.bestFloor * h.rewindThreshold

			if rebound > threshold && h.recoveryCount < h.maxRecoveries && h.checkpoint != nil {
				// === IMMUNE RESPONSE: kill, collect, rewind ===
				h.restoreCheckpoint()
				h.recoveryCount++
				h.immuneActive = false
				// Reset Fibonacci stride — fresh exploration from scratch
				h.fibA, h.fibB = 1, 1
				h.fibStepsOnConductor = 0
				h.condEMA = 0
				// Don't reset bestFloor — CRISPR remembers the target
				return true // signal to caller: skip this step
			}

			// Loss stayed near the floor — the floor was real, not a virus.
			// Accept the current state and deactivate immune monitoring.
			h.immuneActive = false
			h.checkpointLoss = loss
			h.saveCheckpoint()
			h.checkpointStep = step
		}
	}

	return false
}

// saveCheckpoint snapshots the current healthy weights.
// With arena: pointer swap (nanoseconds). Without: deep copy (milliseconds).
func (h *HelixOptimizer) saveCheckpoint() {
	// Fire disk checkpoint callback (throttled)
	if h.onCheckpoint != nil && h.checkpointStep-h.lastDiskCheckpoint >= h.checkpointInterval {
		h.lastDiskCheckpoint = h.checkpointStep
		h.onCheckpoint(h.checkpointStep, h.checkpointLoss, h.bestFloor)
	}

	if h.arena != nil {
		// Arena mode: swap pointers. Old live becomes checkpoint, old checkpoint becomes live.
		// Sync first — the previous step's async rung kernel may still be writing
		// to the current live buffer. We must wait for it to finish before swapping.
		if h.arenaSync != nil {
			h.arenaSync()
		}
		h.arena.Swap()
		return
	}

	// Fallback: deep copy
	nParams := len(h.singles)
	for range h.pairs { nParams += 2 }
	if h.checkpoint == nil || len(h.checkpoint) != nParams {
		h.checkpoint = make([][]float32, nParams)
	}
	idx := 0
	for _, pair := range h.pairs {
		d1 := pair.strand1.DataHelix()
		if h.checkpoint[idx] == nil || len(h.checkpoint[idx]) != len(d1) {
			h.checkpoint[idx] = make([]float32, len(d1))
		}
		helixCheckpointCopy(h.checkpoint[idx], d1)
		idx++
		d2 := pair.strand2.DataHelix()
		if h.checkpoint[idx] == nil || len(h.checkpoint[idx]) != len(d2) {
			h.checkpoint[idx] = make([]float32, len(d2))
		}
		helixCheckpointCopy(h.checkpoint[idx], d2)
		idx++
	}
	for _, p := range h.singles {
		d := p.DataHelix()
		if h.checkpoint[idx] == nil || len(h.checkpoint[idx]) != len(d) {
			h.checkpoint[idx] = make([]float32, len(d))
		}
		helixCheckpointCopy(h.checkpoint[idx], d)
		idx++
	}
}

// restoreCheckpoint reverts weights to the saved checkpoint.
// With arena: pointer swap (nanoseconds). Without: deep copy (milliseconds).
func (h *HelixOptimizer) restoreCheckpoint() {
	if h.arena != nil {
		if h.arenaSync != nil {
			h.arenaSync()
		}
		h.arena.Swap()
		if h.onRestore != nil {
			h.onRestore()
		}
		return
	}

	// Fallback: deep copy
	if h.checkpoint == nil { return }
	idx := 0
	for _, pair := range h.pairs {
		helixCheckpointCopy(pair.strand1.DataHelix(), h.checkpoint[idx])
		idx++
		helixCheckpointCopy(pair.strand2.DataHelix(), h.checkpoint[idx])
		idx++
	}
	for _, p := range h.singles {
		helixCheckpointCopy(p.DataHelix(), h.checkpoint[idx])
		idx++
	}
}

// SetRewindThreshold sets the fractional rebound that triggers immune rewind.
// Default 0.15 (15%). Lower = more aggressive rewinding.
func (h *HelixOptimizer) SetRewindThreshold(t float32) { h.rewindThreshold = t }

// ImmuneActive returns true if the immune system is currently monitoring.
func (h *HelixOptimizer) ImmuneActive() bool { return h.immuneActive }

// BestFloor returns the lowest loss ever observed (CRISPR memory).
func (h *HelixOptimizer) BestFloor() float32 { return h.bestFloor }

// RecoveryCount returns how many times the immune system has rewound.
func (h *HelixOptimizer) RecoveryCount() int { return h.recoveryCount }
func (h *HelixOptimizer) CurrentRung() Rung          { return h.lastRung }
func (h *HelixOptimizer) SignalScale() float32        { return h.lastSignalScale }

// Rung holds the 6 connection point weights for one base pair.
// Exported so the Metal dispatch path in train.go can read the values.
type Rung struct {
	Backbone1  float32 // ① strand 1 anchor weight
	Glyco1     float32 // ② strand 1 signal weight
	Hbond1     float32 // ③ hydrogen bond 1 (momentum coupling)
	Hbond2     float32 // ④ hydrogen bond 2 (velocity coupling)
	Glyco2     float32 // ⑤ strand 2 signal weight
	Backbone2  float32 // ⑥ strand 2 anchor weight
}

// computeRung derives the 6 connection point weights from the current
// helix geometry: phase angle, groove asymmetry, and curve modulation.
func (h *HelixOptimizer) computeRung(curvePhase, majorWeight, minorWeight float64) Rung {
	// The helix phase determines which groove faces "outward" (accessible).
	// Major groove: protein binding site = primary gradient channel.
	// Minor groove: structural = stabilization channel.
	//
	// The two grooves alternate as the helix rotates. At any phase angle,
	// one groove dominates. The groove ratio (1.833:1) sets the asymmetry.

	// Groove exposure at current phase (sinusoidal modulation)
	// Major groove spans ~220° of the helix, minor spans ~140°.
	grooveAngle := h.phase + curvePhase*0.1 // curve adds subtle non-periodicity
	majorExposure := float32(0.5 + 0.5*math.Cos(grooveAngle))
	minorExposure := 1.0 - majorExposure

	// Scale by landscape-adaptive groove weights
	major := majorExposure * float32(majorWeight)
	minor := minorExposure * float32(minorWeight)

	// Backbone points: anchoring strength from helix angle.
	// At 28.4°, the backbone has both axial (rise) and rotational components.
	// cos(helixAngle) = axial component (stability)
	// sin(helixAngle) = rotational component (exploration)
	axial := float32(math.Cos(dnaHelixAngle))
	rotational := float32(math.Sin(dnaHelixAngle))

	return Rung{
		Backbone1: axial,                          // ① anchor: stability-weighted
		Glyco1:    rotational + major,             // ② signal: rotation + major groove
		Hbond1:    major,                          // ③ momentum coupling via major groove
		Hbond2:    minor,                          // ④ velocity coupling via minor groove
		Glyco2:    rotational + minor,             // ⑤ signal: rotation + minor groove
		Backbone2: axial,                          // ⑥ anchor: stability-weighted
	}
}

// grooveWeights is the legacy single-step version. Kept for immuneResponse
// which needs to track prevLoss independently of the signal chain.
func (h *HelixOptimizer) updateLossMomentum(loss float64) {
	if h.prevLoss == 0 {
		h.prevLoss = loss
		return
	}
	dLoss := math.Abs(loss - h.prevLoss)
	h.prevLoss = loss
	h.lossMomentum = 0.05*dLoss + 0.95*h.lossMomentum
}

// dnaStep applies the 6-point rung update to a paired parameter.
//
// The gradient flows through the rung:
//   backbone₁ → glyco₁ → H-bonds → glyco₂ → backbone₂
//
// Each connection point modulates a different aspect of the update:
//   backbone: weight decay / anchoring
//   glycosidic: gradient signal strength
//   H-bonds: cross-strand momentum/velocity coupling
func (h *HelixOptimizer) dnaStep(pair helixPair, r Rung, lr, bc1, bc2 float32) {
	g1, g2 := pair.strand1.GradHelix(), pair.strand2.GradHelix()
	d1, d2 := pair.strand1.DataHelix(), pair.strand2.DataHelix()
	m1, m2 := pair.strand1.MomHelix(), pair.strand2.MomHelix()
	v1, v2 := pair.strand1.VelHelix(), pair.strand2.VelHelix()
	n1, n2 := pair.strand1.SizeHelix(), pair.strand2.SizeHelix()

	b1, b2 := h.beta1, h.beta2
	ob1, ob2 := 1-b1, 1-b2
	eps, wd := h.eps, h.weightDecay

	// Bond coupling strength: AT=2 bonds, GC=3 bonds.
	// Normalized to [0,1] range: AT=0.4, GC=0.6
	bondStrength := float32(pair.strength) / 5.0

	// Coupled region: min of both param sizes
	coupled := n1
	if n2 < coupled {
		coupled = n2
	}

	for i := 0; i < coupled; i++ {
		// === The 6-point rung gradient computation ===

		// ① Backbone 1: weight anchoring (weight decay modulated by axial component)
		wd1 := wd * r.Backbone1

		// ② Glycosidic 1: gradient signal into the rung
		signal1 := g1[i] * r.Glyco1

		// ③④ Hydrogen bonds: cross-strand coupling
		// Bond 1 (momentum): strand 2's gradient influences strand 1's momentum
		// Bond 2 (velocity): strand 1's gradient influences strand 2's velocity
		// Bond strength determines how much cross-talk occurs.
		crossMom := g2[i] * r.Hbond1 * bondStrength  // ③ strand2 → strand1 momentum
		crossVel := g1[i] * r.Hbond2 * bondStrength  // ④ strand1 → strand2 velocity

		// ⑤ Glycosidic 2: gradient signal from the other side
		signal2 := g2[i] * r.Glyco2

		// ⑥ Backbone 2: weight anchoring
		wd2 := wd * r.Backbone2

		// === Adam update for strand 1 ===
		// Effective gradient: own signal + cross-strand momentum coupling
		effGrad1 := signal1 + crossMom
		mi1 := b1*m1[i] + ob1*effGrad1
		vi1 := b2*v1[i] + ob2*effGrad1*effGrad1
		m1[i] = mi1
		v1[i] = vi1
		mh1 := mi1 / bc1
		vh1 := vi1 / bc2
		d1[i] -= lr * (mh1/(helixSqrt(vh1)+eps) + wd1*d1[i])

		// === Adam update for strand 2 ===
		// Effective gradient: own signal + cross-strand velocity coupling
		effGrad2 := signal2 + crossVel
		mi2 := b1*m2[i] + ob1*effGrad2
		vi2 := b2*v2[i] + ob2*effGrad2*effGrad2
		m2[i] = mi2
		v2[i] = vi2
		mh2 := mi2 / bc1
		vh2 := vi2 / bc2
		d2[i] -= lr * (mh2/(helixSqrt(vh2)+eps) + wd2*d2[i])
	}

	// Uncoupled remainder: standard Adam
	for i := coupled; i < n1; i++ {
		g := g1[i]
		mi := b1*m1[i] + ob1*g
		vi := b2*v1[i] + ob2*g*g
		m1[i] = mi
		v1[i] = vi
		d1[i] -= lr * (mi/bc1/(helixSqrt(vi/bc2)+eps) + wd*d1[i])
	}
	for i := coupled; i < n2; i++ {
		g := g2[i]
		mi := b1*m2[i] + ob1*g
		vi := b2*v2[i] + ob2*g*g
		m2[i] = mi
		v2[i] = vi
		d2[i] -= lr * (mi/bc1/(helixSqrt(vi/bc2)+eps) + wd*d2[i])
	}
}

// adamStep applies standard AdamW to an unpaired parameter.
func (h *HelixOptimizer) adamStep(p HelixParam, lr, bc1, bc2 float32) {
	g, d, m, v := p.GradHelix(), p.DataHelix(), p.MomHelix(), p.VelHelix()
	n := p.SizeHelix()
	b1, b2 := h.beta1, h.beta2
	ob1, ob2 := 1-b1, 1-b2
	eps, wd := h.eps, h.weightDecay

	for i := 0; i < n; i++ {
		gi := g[i]
		mi := b1*m[i] + ob1*gi
		vi := b2*v[i] + ob2*gi*gi
		m[i] = mi
		v[i] = vi
		mh := mi / bc1
		vh := vi / bc2
		d[i] -= lr * (mh/(helixSqrt(vh)+eps) + wd*d[i])
	}
}

// ForwardOnlyStep runs a DNA optimizer step WITHOUT backpropagation.
// Instead of reading per-element gradients, it uses momentum as the synthetic gradient:
//   - Loss decreased → momentum direction was correct → amplify
//   - Loss increased → momentum was wrong → dampen/reverse (immune system handles big reversals)
//
// The first step is a noop (no momentum history). After that, momentum IS the gradient.
// The DNA rung coupling, immune system, signal chain, and Fibonacci stride all work unchanged.
// Only the gradient source changes: from backprop to momentum-guided loss-delta.
func (h *HelixOptimizer) ForwardOnlyStep(step int, loss float32, lr float32) {
	// Immune system: checkpoint and floor detection (unchanged)
	rewound := h.immuneResponse(step, loss)
	if rewound {
		return
	}

	// Loss-delta signal: the only feedback from the forward pass
	dLoss := float64(loss) - h.prevLoss
	h.updateLossMomentum(float64(loss))

	// First step: no history, no update (noop forward)
	if h.prevLoss == 0 || step <= 1 {
		h.prevLoss = float64(loss)
		h.recordSignal(loss, 0)
		return
	}

	// Signal direction: loss going down = positive signal, up = negative
	// Normalized by loss magnitude to prevent scale issues
	var signalScale float32
	if dLoss < 0 {
		// Improving — momentum was right. Scale by how much we improved.
		signalScale = 1.0
	} else {
		// Worsening — momentum was wrong. Dampen proportional to worsening.
		// Small worsening (noise): gentle correction. Big worsening: hard reverse.
		ratio := float32(dLoss / math.Max(math.Abs(h.prevLoss), 1e-6))
		signalScale = -ratio // negative = reverse momentum direction
		if signalScale < -1.0 { signalScale = -1.0 } // cap reversal
	}

	// Gradient norm from momentum (for signal chain recording)
	var momNormSq float64
	for _, pair := range h.pairs {
		for _, m := range pair.strand1.MomHelix() { momNormSq += float64(m) * float64(m) }
		for _, m := range pair.strand2.MomHelix() { momNormSq += float64(m) * float64(m) }
	}
	for _, p := range h.singles {
		for _, m := range p.MomHelix() { momNormSq += float64(m) * float64(m) }
	}
	gradNorm := float32(math.Sqrt(momNormSq))

	h.recordSignal(loss, gradNorm)

	// EMA Fibonacci stride: low-pass filter on conductivity
	conductivity := h.SignalConductivity()
	h.condEMA = 0.1*conductivity + 0.9*h.condEMA
	if h.condEMA > 0.005 {
		h.fibA, h.fibB = h.fibB, h.fibA+h.fibB
		if h.fibB > 21 { h.fibB = 21 }
	} else if h.condEMA < -0.005 {
		if h.fibB > 1 {
			h.fibA, h.fibB = 1, h.fibA
			if h.fibA < 1 { h.fibA = 1 }
		}
	}

	stride := float64(h.fibB)
	h.phase += dnaTwistPerStep * stride

	basePairNum := int(h.phase / dnaTwistPerStep)
	if basePairNum > 0 && basePairNum%dnaBasePairsPerTurn == 0 {
		h.fibTurnA, h.fibTurnB = h.fibTurnB, h.fibTurnA+h.fibTurnB
		if h.fibTurnB > 21 { h.fibTurnB = 21 }
	}
	if basePairNum > 0 && basePairNum%(dnaBasePairsPerTurn*10) == 0 {
		h.fibCoilA, h.fibCoilB = h.fibCoilB, h.fibCoilA+h.fibCoilB
		if h.fibCoilB > 21 { h.fibCoilB = 21 }
	}

	h.advanceCurve()
	curvePhase := math.Atan2(h.curveY, h.curveX)
	majorWeight, minorWeight := h.grooveWeightsFromSignal()
	phi := (1.0 + math.Sqrt(5.0)) / 2.0
	turnBreath := math.Sin(float64(h.fibTurnB) * h.phase / phi)
	coilBreath := math.Sin(float64(h.fibCoilB) * h.phase / (phi * phi))
	majorWeight *= 1.0 + 0.10*turnBreath + 0.05*coilBreath
	minorWeight *= 1.0 - 0.05*turnBreath + 0.10*coilBreath

	bc1 := float32(1.0 - math.Pow(float64(h.beta1), float64(step)))
	bc2 := float32(1.0 - math.Pow(float64(h.beta2), float64(step)))

	rung := h.computeRung(curvePhase, majorWeight, minorWeight)
	h.lastRung = rung
	h.lastSignalScale = signalScale

	// DNA step with momentum as synthetic gradient
	for _, pair := range h.pairs {
		h.dnaStepForwardOnly(pair, rung, lr, bc1, bc2, signalScale)
	}
	for _, p := range h.singles {
		h.adamStepForwardOnly(p, lr, bc1, bc2, signalScale)
	}

	h.prevLoss = float64(loss)
}

// dnaStepForwardOnly — paired update using momentum as synthetic gradient.
// The momentum m[i] encodes the EMA of all past gradients for element i.
// signalScale modulates: +1 = momentum was right (keep going), -ratio = wrong (reverse).
func (h *HelixOptimizer) dnaStepForwardOnly(pair helixPair, r Rung, lr, bc1, bc2, signalScale float32) {
	d1, d2 := pair.strand1.DataHelix(), pair.strand2.DataHelix()
	m1, m2 := pair.strand1.MomHelix(), pair.strand2.MomHelix()
	v1, v2 := pair.strand1.VelHelix(), pair.strand2.VelHelix()
	n1, n2 := pair.strand1.SizeHelix(), pair.strand2.SizeHelix()

	b1, b2 := h.beta1, h.beta2
	ob1, ob2 := 1-b1, 1-b2
	eps, wd := h.eps, h.weightDecay
	bondStrength := float32(pair.strength) / 5.0

	coupled := n1
	if n2 < coupled { coupled = n2 }

	for i := 0; i < coupled; i++ {
		// Synthetic gradient from momentum: m[i] * signalScale
		// If improving: synth = m[i] (same direction as accumulated gradient history)
		// If worsening: synth = -ratio * m[i] (partial reversal)
		sg1 := m1[i] * signalScale
		sg2 := m2[i] * signalScale

		wd1 := wd * r.Backbone1
		signal1 := sg1 * r.Glyco1
		crossMom := sg2 * r.Hbond1 * bondStrength
		signal2 := sg2 * r.Glyco2
		crossVel := sg1 * r.Hbond2 * bondStrength
		wd2 := wd * r.Backbone2

		effGrad1 := signal1 + crossMom
		mi1 := b1*m1[i] + ob1*effGrad1
		vi1 := b2*v1[i] + ob2*effGrad1*effGrad1
		m1[i] = mi1
		v1[i] = vi1
		d1[i] -= lr * (mi1/bc1/(helixSqrt(vi1/bc2)+eps) + wd1*d1[i])

		effGrad2 := signal2 + crossVel
		mi2 := b1*m2[i] + ob1*effGrad2
		vi2 := b2*v2[i] + ob2*effGrad2*effGrad2
		m2[i] = mi2
		v2[i] = vi2
		d2[i] -= lr * (mi2/bc1/(helixSqrt(vi2/bc2)+eps) + wd2*d2[i])
	}

	for i := coupled; i < n1; i++ {
		sg := m1[i] * signalScale
		mi := b1*m1[i] + ob1*sg
		vi := b2*v1[i] + ob2*sg*sg
		m1[i] = mi; v1[i] = vi
		d1[i] -= lr * (mi/bc1/(helixSqrt(vi/bc2)+eps) + wd*d1[i])
	}
	for i := coupled; i < n2; i++ {
		sg := m2[i] * signalScale
		mi := b1*m2[i] + ob1*sg
		vi := b2*v2[i] + ob2*sg*sg
		m2[i] = mi; v2[i] = vi
		d2[i] -= lr * (mi/bc1/(helixSqrt(vi/bc2)+eps) + wd*d2[i])
	}
}

// adamStepForwardOnly — single parameter update using momentum as synthetic gradient.
func (h *HelixOptimizer) adamStepForwardOnly(p HelixParam, lr, bc1, bc2, signalScale float32) {
	d, m, v := p.DataHelix(), p.MomHelix(), p.VelHelix()
	n := p.SizeHelix()
	b1, b2 := h.beta1, h.beta2
	ob1, ob2 := 1-b1, 1-b2
	eps, wd := h.eps, h.weightDecay

	for i := 0; i < n; i++ {
		sg := m[i] * signalScale
		mi := b1*m[i] + ob1*sg
		vi := b2*v[i] + ob2*sg*sg
		m[i] = mi; v[i] = vi
		d[i] -= lr * (mi/bc1/(helixSqrt(vi/bc2)+eps) + wd*d[i])
	}
}

// clipGradients applies global gradient norm clipping. Returns the pre-clip norm.
func (h *HelixOptimizer) clipGradients(maxNorm float32) float32 {
	var normSq float64
	for _, pair := range h.pairs {
		for _, g := range pair.strand1.GradHelix() {
			normSq += float64(g) * float64(g)
		}
		for _, g := range pair.strand2.GradHelix() {
			normSq += float64(g) * float64(g)
		}
	}
	for _, p := range h.singles {
		for _, g := range p.GradHelix() {
			normSq += float64(g) * float64(g)
		}
	}
	norm := float32(math.Sqrt(normSq))
	if norm > maxNorm {
		scale := maxNorm / norm
		for _, pair := range h.pairs {
			for i := range pair.strand1.GradHelix() {
				pair.strand1.GradHelix()[i] *= scale
			}
			for i := range pair.strand2.GradHelix() {
				pair.strand2.GradHelix()[i] *= scale
			}
		}
		for _, p := range h.singles {
			for i := range p.GradHelix() {
				p.GradHelix()[i] *= scale
			}
		}
	}
	return norm
}

// recordSignal writes the current step's charge into the signal chain.
func (h *HelixOptimizer) recordSignal(loss, gradNorm float32) {
	var dLoss float32
	if h.signalLen > 0 {
		// Previous rung
		prevIdx := (h.signalHead - 1 + h.signalCap) % h.signalCap
		dLoss = loss - h.signal[prevIdx].loss
	}

	h.signal[h.signalHead] = signalRung{
		loss:     loss,
		gradNorm: gradNorm,
		dLoss:    dLoss,
		phase:    h.phase,
	}
	h.signalHead = (h.signalHead + 1) % h.signalCap
	if h.signalLen < h.signalCap {
		h.signalLen++
	}
}

// grooveWeightsFromSignal reads the signal chain to compute groove weights.
//
// Charge transfer through DNA's pi stack attenuates with distance.
// Closer base pairs have stronger signal overlap. The attenuation
// follows an exponential decay: signal ∝ exp(-d/λ) where d is the
// distance in base pairs and λ is the characteristic length (~10 bp,
// matching real DNA charge transfer measurements).
//
// The signal chain reveals the loss landscape's local structure:
//   - Consistent downward dLoss across many rungs = conductive path
//     → major groove dominates (strong directional gradient)
//   - Mixed/oscillating dLoss = resistive region
//     → minor groove grows (exploration mode)
//   - All rungs near the same loss = insulating plateau
//     → minor groove maximal (full exploration)
func (h *HelixOptimizer) grooveWeightsFromSignal() (major, minor float64) {
	if h.signalLen < 2 {
		return 1.0, 1.0 / dnaGrooveRatio
	}

	// Read the signal chain with distance-dependent attenuation.
	// λ = 10 base pairs (one full helix turn), matching real DNA.
	lambda := 10.0
	var conductivity float64 // positive = conductive (loss decreasing), negative = resistive
	var totalWeight float64

	for i := 0; i < h.signalLen; i++ {
		// Distance from current position (most recent rung = distance 0)
		idx := (h.signalHead - 1 - i + h.signalCap*2) % h.signalCap
		distance := float64(i)

		// Charge attenuation: exp(-d/λ)
		attenuation := math.Exp(-distance / lambda)
		totalWeight += attenuation

		// dLoss < 0 means loss decreased = conductive (good direction)
		// dLoss > 0 means loss increased = resistive (bad direction)
		conductivity += float64(-h.signal[idx].dLoss) * attenuation
	}

	if totalWeight > 0 {
		conductivity /= totalWeight // normalize
	}

	// Map conductivity to groove weights.
	// High conductivity (consistent descent) → major groove dominates.
	// Low/negative conductivity (plateau/rebound) → minor groove grows.
	//
	// Use tanh to bound the response.
	conductivityNorm := math.Tanh(conductivity * 2.0) // scale factor 2.0 for sensitivity

	// conductivityNorm ∈ [-1, 1]
	//   +1 = perfectly conductive (all rungs descending)
	//    0 = neutral (mixed)
	//   -1 = perfectly resistive (all rungs ascending)

	// Major groove: strong when conductive, reduced when resistive
	major = 0.5 + 0.5*conductivityNorm // range [0, 1]
	if major < 0.1 {
		major = 0.1 // always some gradient signal
	}

	// Minor groove: inverse — exploration when resistive
	minor = (0.5 - 0.3*conductivityNorm) / dnaGrooveRatio // range [0.1, 0.44]

	return major, minor
}

// SignalConductivity returns the current conductivity reading from the signal chain.
// Positive = loss consistently decreasing. Negative = rebounding/plateau.
func (h *HelixOptimizer) SignalConductivity() float64 {
	if h.signalLen < 2 {
		return 0
	}
	lambda := 10.0
	var conductivity, totalWeight float64
	for i := 0; i < h.signalLen; i++ {
		idx := (h.signalHead - 1 - i + h.signalCap*2) % h.signalCap
		att := math.Exp(-float64(i) / lambda)
		totalWeight += att
		conductivity += float64(-h.signal[idx].dLoss) * att
	}
	if totalWeight > 0 {
		conductivity /= totalWeight
	}
	return conductivity
}

// advanceCurve computes the next point on the elliptic curve y² = x³ + 7
// via point doubling. Provides non-repeating phase modulation.
func (h *HelixOptimizer) advanceCurve() {
	x, y := h.curveX, h.curveY
	if math.Abs(y) < 1e-15 {
		h.curveX, h.curveY = 2.0, math.Sqrt(15.0)
		return
	}
	lambda := (3 * x * x) / (2 * y)
	xNew := lambda*lambda - 2*x
	yNew := lambda*(x-xNew) - y
	xNew = math.Mod(xNew, 1e6)
	if math.IsNaN(xNew) || math.IsInf(xNew, 0) {
		xNew, yNew = 1.0, math.Sqrt(8.0)
	}
	h.curveX, h.curveY = xNew, yNew
}

// PrepareStep runs the helix bookkeeping and returns the computed Rung.
//
// Two goroutines run in parallel:
//   goroutine 1 (AMX): gradient clipping via cblas_snrm2 + cblas_sscal
//   goroutine 2 (CPU): signal chain, Fibonacci stride, groove weights, rung geometry
//
// AMX and CPU cores execute simultaneously — the coprocessor clips gradients
// while the CPU computes the rung. Both finish before the Metal kernel dispatches.
//
// Returns (rung, bc1, bc2, rewound).
func (h *HelixOptimizer) PrepareStep(step int, loss float32, lr float32) (Rung, float32, float32, bool) {
	// Immune response (must be serial — modifies checkpoints)
	rewound := h.immuneResponse(step, loss)
	if rewound {
		return Rung{}, 0, 0, true
	}

	// --- Parallel: AMX clips gradients while CPU computes the rung ---
	// In graph mode (mode 4), gradients live inside MPSGraph — no external
	// buffers to clip. Skip the AMX goroutine, just compute the rung.
	hasParams := len(h.pairs) > 0 || len(h.singles) > 0

	normCh := make(chan float32, 1)
	if hasParams {
		go func() {
			var gn float32
			if useHelixAccel {
				gn = helixClipGradsAccel(h, 1.0)
			} else {
				gn = h.clipGradients(1.0)
			}
			normCh <- gn
		}()
	}

	// CPU: signal chain + rung computation (parallel with AMX when params exist)

	h.updateLossMomentum(float64(loss))

	// Conductivity from previous signal (before recording this step)
	conductivity := h.SignalConductivity()

	// EMA Fibonacci stride
	h.condEMA = 0.1*conductivity + 0.9*h.condEMA
	if h.condEMA > 0.005 {
		h.fibA, h.fibB = h.fibB, h.fibA+h.fibB
		if h.fibB > 21 { h.fibB = 21 }
	} else if h.condEMA < -0.005 {
		if h.fibB > 1 {
			h.fibA, h.fibB = 1, h.fibA
			if h.fibA < 1 { h.fibA = 1 }
		}
	}

	// Phase advance
	h.phase += dnaTwistPerStep * float64(h.fibB)

	// Fractal Fibonacci breathing
	basePairNum := int(h.phase / dnaTwistPerStep)
	if basePairNum > 0 && basePairNum%dnaBasePairsPerTurn == 0 {
		h.fibTurnA, h.fibTurnB = h.fibTurnB, h.fibTurnA+h.fibTurnB
		if h.fibTurnB > 21 { h.fibTurnB = 21 }
	}
	if basePairNum > 0 && basePairNum%(dnaBasePairsPerTurn*10) == 0 {
		h.fibCoilA, h.fibCoilB = h.fibCoilB, h.fibCoilA+h.fibCoilB
		if h.fibCoilB > 21 { h.fibCoilB = 21 }
	}

	// Elliptic curve phase
	h.advanceCurve()
	curvePhase := math.Atan2(h.curveY, h.curveX)

	// Groove weights from signal chain
	majorWeight, minorWeight := h.grooveWeightsFromSignal()

	// Fractal golden ratio modulation
	phi := (1.0 + math.Sqrt(5.0)) / 2.0
	turnBreath := math.Sin(float64(h.fibTurnB) * h.phase / phi)
	coilBreath := math.Sin(float64(h.fibCoilB) * h.phase / (phi * phi))
	majorWeight *= 1.0 + 0.10*turnBreath + 0.05*coilBreath
	minorWeight *= 1.0 - 0.05*turnBreath + 0.10*coilBreath

	// Bias correction
	bc1 := float32(1.0 - math.Pow(float64(h.beta1), float64(step)))
	bc2 := float32(1.0 - math.Pow(float64(h.beta2), float64(step)))

	// Compute the 6-point rung
	r := h.computeRung(curvePhase, majorWeight, minorWeight)

	// --- Sync: wait for AMX gradient clipping to finish ---
	var gradNorm float32
	if hasParams {
		gradNorm = <-normCh
	}

	// Record signal
	h.recordSignal(loss, gradNorm)

	return r, bc1, bc2, false
}

// Phase returns the current helix phase angle in radians.
func (h *HelixOptimizer) Phase() float64 { return h.phase }

// BasePair returns which base pair number we're on (1-indexed, wraps every 10).
func (h *HelixOptimizer) BasePair() int {
	bp := int(h.phase/dnaTwistPerStep) % dnaBasePairsPerTurn
	return bp + 1
}

// Turn returns how many full helix turns have completed.
func (h *HelixOptimizer) Turn() int {
	return int(h.phase / (2 * math.Pi))
}

// LossMomentum returns the EMA of loss change magnitude.
func (h *HelixOptimizer) LossMomentum() float64 { return h.lossMomentum }

// Stride returns the current Fibonacci stride (base pairs per step).
func (h *HelixOptimizer) Stride() int { return h.fibB }

// DefaultHelixAngle returns the DNA helix angle (28.4°). Not configurable.
func DefaultHelixAngle(_ int) float32 { return float32(dnaHelixAngle) }
