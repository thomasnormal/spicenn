# spicenn

`spicenn` is the small component library for the SPICE-native neural-network
experiments.  The boundary is intentionally conservative: components describe
physical circuit blocks and render SPICE, while experiment scripts still choose
datasets, clock schedules, and sweeps.

The core contract is:

- `Component` is the base for anything renderable to SPICE.
- `CapState` owns a capacitor-backed state node, optional leakage, and optional
  reset switch.
- `Synapse`, `Neuron`, and `Layer` inherit from `Component`.
- Components expose `input_nodes()`, `output_nodes()`, and `state_nodes()` so
  experiments can inspect connectivity without parsing generated netlists.
- `NonnegativeSignal` names a compact single-ended ReLU-style signal, while
  `SignedSignal` names an explicit positive/negative rail pair.  This is a
  guardrail: signed inputs, score differences, and error/delta rails should not
  be silently routed through a one-wire activation bus.
- `FanInTopology` describes dense or sparse fan-in/fan-out wiring.
- `NetlistBuilder` is the low-level renderer used by every component.

## Rail convention

The compact convention is only valid at ReLU-style neuron outputs:

- input rails and ReLU activations are `NonnegativeSignal` values and can use
  one wire;
- synapse sums, output scores, backward errors, and deltas are signed quantities
  and must use explicit `SignedSignal` positive/negative rail pairs;
- a signed synapse driven by a nonnegative activation routes its positive
  weight branch into the downstream `+` rail and its negative weight branch into
  the downstream `-` rail;
- a ReLU neuron compares the downstream preactivation rails and writes only a
  compact nonnegative activation capacitor for the next layer.

So the hidden-layer path is:

```text
nonnegative activation -> signed synapse branches -> u+ / u- -> ReLU -> nonnegative activation
```

The readout path stays differential at its output:

```text
nonnegative activation gate -> signed readout branches -> score+ / score-
```

This means a ReLU output does not need `x+`/`x-`, but each synapse contribution
and every score/error/delta signal still does.

The default sparse readout uses the activation as a MOS gate, not as a pass
terminal.  This matters because a pass-source readout branch can back-drive a
zero activation toward roughly `V_weight - V_th` during the same forward
window, creating a false hidden activation floor.  `pass_act_source` remains
available as an explicit experiment, but the default `gate_stack` readout keeps
the hidden activation capacitor isolated from score/readout charge injection.

Biases follow the same rule as ordinary nonnegative activations: a hidden-layer
bias can be a small analog source selected by signed hidden synapses, while the
readout/output bias should use a full-swing constant activation source when it
feeds the same pretrace/write circuit as hidden activations.  Its writer width
is now independently scalable through the readout update builder, so a bias edge
can be physically smaller or larger than feature edges without changing the
mathematical topology.  This represents transistor sizing of that local writer,
not a Python-side update.

Backward transport uses the same sign convention.  A signed error
`e = e+ - e-` passing through a signed weight `w = w+ - w-` needs four physical
branches:

```text
w+ and e+ -> delta+
w+ and e- -> delta-
w- and e+ -> delta-
w- and e- -> delta+
```

The reusable `DifferentialToDifferentialSynapse` block renders those branches
so reverse error flow can reuse the forward weight capacitor nodes without
collapsing the signed error in Python.

Hidden-weight writes use the same local product shape as readout writes:

```text
stored pre activation / trace + signed post delta -> local Cw+ / Cw- update
```

The `DifferentialSignalGate` block provides the ReLU backward mask by gating
raw hidden deltas with the stored ReLU activation capacitor before they reach
the hidden writer.  The current sparse train deck can optionally instantiate
this for the second hidden layer.

The hidden writer now also has a regenerative differential selector mode.  It
senses `delta+` versus `delta-`, produces high-true `hwpos`/`hwneg` write rails,
and exports a local `active` gate.  The senseamp-CMOS variant gates its PMOS
charge path with that local `active` rail and disables the ambiguous NMOS
discharge leg, so equal `delta+ == delta-` is nearly quiet instead of eroding
both weight capacitors.  In the current sparse transient tests the useful
operating point is a small selector width, about `2u`: that is weak enough that
the cross-coupled keepers can resolve one high rail and one low rail.  With a
programmed nonzero readout separator this produces signed hidden-weight motion
in both directions.  A centered readout still produces `delta+ == delta-`, so
the hidden layer correctly remains almost unchanged until the readout weights
have developed a signed difference.

Current reusable blocks include capacitor arrays, differential weight-state
arrays, nonnegative-to-differential hidden synapses, differential ReLU neurons,
signed score-error cells, differential-to-differential reverse-flow synapses,
ReLU delta gates, regenerative hidden-delta write selectors, hidden weight
update layers, the
`make_sparse_differential_relu_layer`,
`make_sparse_signed_readout_layer`,
`make_sparse_differential_error_transport_layer`, and
`make_sparse_readout_update_layer` topology builders, per-synapse
pre-activation traces, node parasitic anchors, and the diffpair-bleed write
selector used by the direct-flow experiments.

For classification experiments, the score-error cell can be driven with either
a positive target rail only,

```text
dp - dn ~= target_pos + score- - score+
```

or with differential target rails,

```text
dp - dn ~= target_pos - target_neg + score- - score+
```

The second form is useful as a hardware-native way to test explicit false-class
depression.  It is not a softmax/cross-entropy implementation; it is a signed
residual generated by transistor stacks.

Going forward, new reusable transistor-level blocks should be added here first,
with unit tests that lock down the rendered SPICE contract.  Experiment drivers
should compose these blocks and keep only experiment-specific schedules,
measurements, and command-line options locally.
