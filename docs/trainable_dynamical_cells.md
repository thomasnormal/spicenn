# Trainable Dynamical Cells

SPICENN experiments should be organized around `TrainableDynamicalCell`, not a
single narrow `NeuronCell` interface.

A trainable dynamical cell is any SPICE-emittable learning block with:

- public ports
- internal state nodes
- access policies for those state nodes
- a learning protocol
- capabilities and role metadata
- characterization expectations
- tagged emitted elements for experiment-mode linting

This lets the same infrastructure cover behavioral local-feature cells,
transistorized tanh cells, leaky hard-tanh cells, DFA cells, crossbar tiles,
EqProp tiles, reservoirs, and spiking eligibility cells.

## Protocols

`ProtocolFamily` is only the coarse classification.  The real protocol is a
`LearningProtocol` object containing required phases, required ports, required
state roles, and default characterization tests.

Examples:

- `BACKPROP_LOCAL`: `x`, `h`, `learning_in`, `pact`, `pbwd`, `pacc`, `eta`
- `DFA`: `x`, `h`, `output_error`, `pact`, `pbwd`, `pacc`, `eta`
- `EQPROP`: `input_clamp`, `output_nodes`, `target_nudge`, free/nudged/sample/update phases
- `SPIKING_ELIGIBILITY`: `event_in`, `learning_signal`, trace/update phases

Experiment runners should check protocol-family compatibility first with
`ExperimentSpec`, then call
`cell.contract().protocol.validate_contract(cell.contract())` before emitting a
large deck:

```python
experiment.validate_contract(cell.contract())
```

## Access Policy

Experiment decks may:

- connect public ports
- set declared initial conditions
- include passive probes
- include cell-owned core/write elements

Experiment decks may not:

- force internal state after initialization
- include debug-only or characterization-only adapters
- attach external active sources to internal state nodes
- use internal readout to drive the controller or update rule

Characterization harnesses may add state forcers, internal sweeps, noise/fault
injectors, and debug probes.  Those adapters should be tagged so
`lint_experiment_elements` rejects them in headline experiments.

## Promotion

Large MNIST-style experiments should only use cells that have passed protocol
specific characterization:

1. contract-valid
2. single-cell characterized
3. learning-aligned
4. tiny integration
5. MNIST smoke
6. full experiment

The most important promotion metric for transistor replacements is update
alignment, not exact tanh equality.  A noisy but consistently aligned update is
more useful than a precise-looking update that flips sign under offsets,
saturation, or phase overlap.
