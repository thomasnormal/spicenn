# Parameter Derivation Rules

The SPICE experiments should not rely on a large flat list of tuned widths,
capacitors, and waveform amplitudes.  Most of those values are local circuit
ratios.  They should be derived from a small set of global scales after the
architecture is chosen.

## Principle

For each circuit family, separate parameters into three groups.

## Simulator Status

The experiment drivers can use either ngspice or Xyce through the shared
spicelib-backed netlist adapter.  `--simulator auto` keeps the historical
ngspice-first order, while `--simulator auto-fast` searches for Xyce/XyceNF
first and falls back to ngspice.

1. **Topology constants**

   These define the actual circuit: output head, write-cell type, pretrace
   storage mode, error-cell type, supply rails, and transistor ratios needed
   for correct sign selection.  They should not be tuned for every dataset.

2. **Derived local values**

   These are SPICE widths and capacitances calculated from the topology
   constants and a few global scales.  Examples are output capacitance,
   score capacitance, target/non-target error widths, write-selector guard
   width, and center-pull defaults.

3. **Global scales**

   These are the only values a Bayesian search should normally tune:

   ```text
   learning_rate_scale  - global write mobility
   error_drive_scale    - global error-current strength
   score_tau_scale      - global score/output integration time constant
   ```

The tuner may still keep broad exploratory profiles, but architecture
comparisons should use theory-derived profiles unless there is a concrete
reason to break a local ratio.

## Derived-Profile Contract

For architecture experiments, the driver should receive a circuit choice and a
topology, then calculate local values from the theory module.  The intended
contract is:

```text
architecture/topology
    -> class_count
    -> effective_readout_fan_in
    -> derived local SPICE widths/caps
```

The tunable surface should stay small:

```text
learning_rate_scale
error_drive_scale
score_tau_scale
```

For the class-evidence normalizer, the same rule applies. The local circuit
values are derived from the required handoff:

```text
low-common score delta
  -> observable contrast cap delta
  -> writer-domain err+/err- rails
```

The derived profile sets:

```text
score_common_resistance = target score-common tau / score_common_cap
mass/error capacitance  = required C for the target writer rail step
target error width      = anchor error width * error_drive_scale
nontarget error width   = target error width / (class_count - 1)
```

So Bayesian search should normally see only the high-level scales:

```text
error_drive_scale
target margin / dead-zone scale
learning_rate_scale
```

and not individual common-reference resistors, score-mass capacitors, or
target/non-target width ratios.

For target-vs-impostor correction, the primitive contract is:

```text
score_opponent + target_margin > score_target
  -> target-positive writer rail
  -> offending-opponent-negative writer rail
```

The local sizing is derived as:

```text
observable_margin_ratio = min(target_margin, score_delta) / score_delta_floor
margin_penalty_width    = pairwise_pulldown_width * target_margin / score_window
error_clock_high        = target_writer_rail + sense_threshold
error_capacitance       = C from I*T/V for the writer rail target
pairwise widths         = anchored to the measured low-gain comparator family
```

In the continuous block this is implemented as a weak physical discharge branch
on the target-wins pairwise decision node while `targetp` and `scoredec` are
active. The branch creates the margin/dead-zone without Python-shifting score
sources. The default writer handoff uses a `0.45 V` score-error clock and
`0.5 fF` error caps because the one-sample integrated writer test showed lower
rails were directionally correct but did not move the readout capacitors.

That leaves the intended Bayesian surface at:

```text
target_margin
error_drive_scale
learning_rate_scale
```

rather than individual pairwise comparator widths, error node capacitances, and
per-class branch widths.

## Normalization Candidate Contract

The hardware normalization candidates are kept as standalone subcircuits before
they are eligible for end-to-end training integration.  The current library
tests these ten score-to-writer handoff approaches:

```text
current-sum
common-mode
pairwise-margin
fixed-total-feedback
soft-wta
charge-share
time-domain
pulse-density
log-domain
learned-calibration
```

Each candidate has the same external contract:

```text
low-millivolt class score rails
one-hot target rails
reset/normalize phase
  -> writer-domain err+ / err- rails for every class
```

The important analytical point is that low-millivolt score rails are not valid
writer gates by themselves.  The subcircuits therefore include an internal
low-gain PMOS score-lift stage before class normalization.  Local values should
be derived from the handoff requirements:

```text
raw score delta           -> low-gain readable score separation
normalizer capacitance    -> stable class-contrast storage
writer error capacitance  -> target writer-domain voltage step
error branch widths       -> I = C * dV / dt for the chosen phase window
```

The ngspice primitive suite currently requires each candidate to pass the same
minimum behavioral guarantees before integration:

```text
wrong target score:
    target class gets positive writer pressure
    nontarget classes get negative writer pressure

larger nontarget score mass:
    target-positive pressure is no weaker than in a clear-target case

flat scores:
    still produce dense bootstrap pressure for the labeled class
```

These tests deliberately do not prove that all ten are equally good.  They prove
that each candidate is electrically active at the right scale and sign, so later
end-to-end experiments can compare a small number of higher-level choices
instead of retuning local device widths.

The continuous multiclass block can instantiate the same library with:

```text
error_mode = normalizer-<approach>-descent
```

where `<approach>` is one of the ten primitive names.  The block-level
integration deliberately reuses the subcircuit instead of copying its internals:

```text
class score capacitors
targetp/targetn rails
scoreerr/scoregaterst phases
  -> norm_<approach>
  -> c<i>_errp / c<i>_errn
  -> existing local readout writer
```

The first plugged-in controls (`current-sum` and `soft-wta`) both learn the
one-hot continuous transient but do not yet improve the real
`mnist3fixed8_12` plateau.  That means the reusable interface is electrically
valid, while the best next experiment should compare approaches with a sharper
normalizer-specific diagnostic rather than treating the first two as solved
training paths.

The all-candidate fixed8 screen now records train-time error-rail statistics.
This matters because the first full screen showed:

```text
different writer-domain error amplitudes
same final fixed8 plateau
```

So `error_drive_scale` is not currently the most promising tuning axis.  A
candidate can produce stronger `errp-errn` rails and still land at the same
final class margins.  Before using Bayesian optimization on the normalizer
surface, the next analytical step should measure whether the score/readout
features are class-separable before the normalizer.  If they are not, tuning
normalizer gain only amplifies the same weak evidence.

## Signal Representation Contract

The circuit should distinguish the mathematical sign of a value from the
physical number of wires used to carry it.

```text
NonnegativeSignal(node):
    one voltage wire; valid for ReLU activations, one-hot rails, and clocks

SignedSignal(pos, neg):
    two voltage wires; valid for signed features, scores, errors, and deltas
```

ReLU lets the forward hidden activation bus stay single-ended because
`h = max(0, u)` is nonnegative.  That does not remove the need for signed rails
elsewhere.  A signed input feature must be split before entering this ReLU
network, for example `x+ = max(x, 0)` and `x- = max(-x, 0)`.  Score differences,
target-minus-output errors, and hidden delta/error streams should remain
explicit positive/negative rail pairs.  The `spicenn` library now exposes
`NonnegativeSignal`, `SignedSignal`, `require_nonnegative_signal`, and
`require_signed_signal` so future reusable components can make this contract
testable instead of implicit in node names.

For a hidden layer with ReLU neurons, the compact forward contract is therefore:

```text
activation h_i:
    one nonnegative wire

synapse i -> j:
    positive weight branch charges u_j+
    negative weight branch charges u_j-

next neuron:
    compares/stores the signed preactivation u_j+ - u_j-
    emits one nonnegative ReLU activation wire
```

The first `spicenn`-generated two-hidden-layer forward primitive now follows
this contract end to end:

```text
input/bias wires
-> sparse differential ReLU hidden layer 1
-> sparse differential ReLU hidden layer 2
-> sparse signed readout score rails
```

The default ngspice smoke (`run_device_spicenn_sparse_forward.py`) produces
nonzero signals through both hidden layers and a deliberately asymmetric
programmed readout:

```text
h1 activations      ~= 0.505 V
h2 activations      ~= 0.363 V
score0+ - score0-   ~= +0.363 V
score1+ - score1-   ~= -0.363 V
score0-score1 margin ~= 0.727 V
```

This is not a training result.  It is the current forward-path proof that the
new reusable `spicenn` layer builders preserve the intended signed-rail
architecture in a transient SPICE deck before backward and update cells are
attached.

The first reusable local readout-update primitive has also been moved into
`spicenn`.  A `SignedScoreErrorCell` generates class-local error rails with the
hardware surrogate:

```text
dp - dn ~= target + score_neg - score_pos
```

and `make_sparse_readout_update_layer` instantiates one `DirectFlowWeightCell`
per sparse readout edge.  The ngspice smoke
`run_device_spicenn_readout_update_smoke.py` verifies the local update sign:

```text
positive error:
    dp ~= 1.044 V, dn ~= 0
    signed weight delta ~= +0.249 V

negative error:
    dp ~= 0, dn ~= 1.044 V
    signed weight delta ~= -0.249 V
```

The same smoke shows about `-0.145 V` branch-common movement in both directions.
That is acceptable as an isolated sign test, but it is not acceptable as the
final training writer.  The next integration step should therefore keep the
sign-correct reusable builder while replacing or compensating the update cell
with the complementary/common-mode-controlled writer before MNIST-scale runs.

The sparse two-hidden `spicenn` forward deck now has a readout train-step mode
that composes the same pieces:

```text
sparse hidden forward
-> signed readout score rails
-> SignedScoreErrorCell
-> sparse DirectFlowWeightCell readout updates
```

With centered readout weights and one output row, ngspice verifies the composed
loop rather than an isolated writer:

```text
h2 activation       ~= 0.363 V
dp error rail       ~= 1.045 V
dn error rail       ~= 0.324 V
row signed delta    ~= +21 mV
row common delta    ~= -122 mV
```

This establishes that the `spicenn` component graph can now perform a complete
forward/error/readout-update step in SPICE.  It also reinforces the current
blocker: the reusable topology is no longer the problem for readout learning;
the write cell still needs common-mode control before repeated training cycles.

The same train-step deck can now use the CMOS complementary writer with
diffpair-selected write rails and per-edge spike pretraces.  Because this
sparse forward path produces smaller activations than the older direct-flow
readout experiments, the spike reference had to be lowered from `0.3 V` to
`0.1 V`; otherwise the low-true pretrace never opened.  With charge/discharge
widths inspired by the primitive ratio:

```text
charge_width    = 5e-4 u
discharge_width = 5e-6 u
spike_ref       = 0.1 V
```

ngspice gives:

```text
row signed delta ~= +8.13 mV
row common delta ~= +1.03 mV
```

So the integrated `spicenn` train step now has sign-correct readout learning
with common-mode motion smaller than the useful signed update.  This should be
the default writer direction for repeated readout-training experiments; the old
simple writer remains useful only as a sign/polarity control.

A repeated readout-training smoke now carries the measured `vw+`/`vw-`
capacitor voltages from one ngspice train step into the next deck as capacitor
initial conditions.  This keeps the sample-to-sample update physics in SPICE
while Python only sequences the input/target values and the next cap ICs.  In a
four-step alternating-label smoke with the CMOS complementary writer:

```text
row0 signed sum: 0      -> +15.87 mV
row1 signed sum: 0      -> +16.02 mV
row0 common sum: 3.300V -> 3.286V
row1 common sum: 3.300V -> 3.286V
```

This proves readout weight state can persist across repeated train steps in the
new component graph.  It also exposes the next readout problem: the forward
score rails are not yet sensitive enough to these small signed weight changes,
so prediction remains tied even while the capacitors move correctly.  The next
architecture/calibration task is therefore readout gain and score-cap sizing,
not more preprocessing.

Changing from dense to sparse readout, or from 3 to 5 to 10 classes, should not
open new pulse-width knobs.  It should change the derived fan-in and one-vs-rest
class-count formulas.  If a result needs a local width to be hand-tuned after a
topology change, that is evidence that the sizing rule or circuit family is
wrong, not evidence that the new topology merely needs another search.

Raw local knobs are still useful while developing a new cell.  Once a cell is a
candidate architecture, its raw knobs should be collapsed into topology
constants plus global scales before comparing it to other candidates.

## Current MNIST3 Random-Readout Family

The best current from-random 3-class family is:

```text
fixed8 features
-> rail-buffer hidden activations
-> split_score_caps output head
-> onehot_limited error rails
-> bounded_pmos_charge_only readout write
-> diffpair_bleed write guard
-> synapse_spike pretrace eligibility gate
```

The derived profile keeps these circuit choices fixed.  It derives the actual
driver parameters as:

```text
readout_update_width_u
    = 5e-4 * learning_rate_scale / readout_fanin_scale

readout_write_error_exclusion_width_u
    = 8.0 * learning_rate_scale / readout_fanin_scale

residual_target_width_u
    = 96.0 * error_drive_scale

residual_output_width_u
    = residual_target_width_u / (class_count - 1)

score_cap_f
    = 10.0 * score_tau_scale * readout_fanin_scale

output_cap_f
    = 2.0 * score_cap_f

output_bias_update_width_u
    = fixed_bias_mobility_ratio * readout_update_width_u
```

The PMOS write cell has two derived variants:

```text
mnist3fixed8_random_pmos_chargeonly_derived:
    readout_flow_write_mode = bounded_pmos_charge_only

mnist3fixed8_random_pmos_cd_derived:
    readout_flow_write_mode = bounded_pmos_charge_discharge
```

The charge/discharge variant keeps the same global scales and local sizing
rules, but adds the opposite-branch discharge leg.  This is a circuit change,
not a new hyperparameter.  Early 5-class probes suggest it gives a stronger
class-specific update than charge-only, although common-mode is still too high.

where:

```text
readout_fanin_scale = effective_readout_fan_in / 8
```

The anchor circuit has 8 hidden activations feeding each output row.  If a
topology has more readout inputs per output, the score capacitor grows so the
same forward current sum stays in the same voltage range.  The write pulse
shrinks by the same factor because a row update with more active inputs has a
larger effective feature norm; this is the circuit analogue of the linear
stability bound `eta < 2 / ||h||^2`.

The effective readout fan-in is derived from topology:

```text
dense:
    hidden_cells

random_fanin:
    min(readout_fan_in, hidden_cells)

random_fanout:
    hidden_cells * min(readout_fan_out, class_count) / class_count

balanced_random_fanout:
    hidden_cells * min(readout_fan_out, class_count) / class_count
```

The target/non-target ratio is derived from the class count.  In a balanced
epoch, each row sees one target exposure and `class_count - 1` non-target
exposures, so this keeps their nominal total drive balanced before data- and
device-dependent mobility effects.  The fixed score/output capacitor ratio
preserves the readout head's settling shape.  The write guard scales with the
write mobility so a larger update current does not also increase
overlap/common-mode writes.

These rules live in `spice/parameter_theory.py`, not just in the Optuna driver.
The tuner uses those formulas through `derive_multiclass_readout_sizing`, so
unit tests can catch accidental drift between the theory and the experiment
launcher.

## Error-Rail Mobility

The one-vs-rest error widths balance logical exposure counts:

```text
target_width = (class_count - 1) * nontarget_width
```

That is necessary, but it is not sufficient for a MOS write cell.  In the PMOS
charge/discharge writer, target (`dp`) and non-target (`dn`) rails pass through
different selector paths before they move the positive and negative weight
capacitors.  Equal logical drive can therefore produce unequal physical write
mobility.

The right abstraction is:

```text
effective_target_mobility
    = target_error_drive * target_selector_mobility

effective_nontarget_mobility
    = nontarget_error_drive * nontarget_selector_mobility
```

The class-count rule balances the first factor.  The selector mobility ratio
should be characterized per write-cell family, then fixed as a circuit constant:

```text
readout_dp_gate_update_width_u = dp_selector_ratio * readout_update_width_u
readout_dn_gate_update_width_u = dn_selector_ratio * readout_update_width_u
```

This ratio is not meant to be a per-topology hyperparameter.  It is closer to
transistor sizing inside the synapse/write-selector cell.  Recent MNIST5 fixed8
probes show that exposing this ratio matters: after fixing the generator so
`dp/dn` overrides affect the diffpair-selected PMOS charge devices, increasing
target-side selector width improved the 10-sample 5-class probe from 0.3 to 0.5
raw score accuracy.

The important measured quantity is not just the voltage on the error rails.  It
is the signed movement of the weight capacitors after the complete
error-selector/write stack.  The readout primitive now reports:

```text
mean_target_row_delta
mean_nontarget_row_delta
measured_one_vs_rest_balance_ratio
measured_one_vs_rest_epoch_delta
mean_target_row_common_delta
mean_nontarget_row_common_delta
measured_one_vs_rest_common_epoch_delta
measured_common_drift_to_signed_step_ratio
```

For `class_count = K`, the epoch-drift diagnostic is:

```text
measured_one_vs_rest_epoch_delta
    = target_delta + (K - 1) * nontarget_delta
```

and the balanced ratio is:

```text
measured_one_vs_rest_balance_ratio
    = target_delta / ((K - 1) * abs(nontarget_delta))
```

The common-mode epoch diagnostic is:

```text
measured_one_vs_rest_common_epoch_delta
    = target_common_delta + (K - 1) * nontarget_common_delta
```

This catches a failure mode that signed balance alone hides.  A charge-only
writer can have a nearly balanced signed row update while both physical
branches keep moving toward the same rail.  In that case the classifier may
look reasonable for a few samples, then lose mobility as the branch capacitors
walk out of their useful range.

The current 5-class PMOS charge/discharge primitive with
`dp_selector_ratio = 4` and `dn_selector_ratio = 1` is sign-correct but not
balanced: the target row moves by about `+0.063 V`, while each non-target
exposure moves by about `-0.044 V`.  Over a balanced 5-class epoch that is
approximately:

```text
+0.063 + 4 * (-0.044) = -0.113 V
```

so each row has a systematic downward drift even before data ordering and
feature activity enter.  Reducing the non-target selector width moves the
non-target write closer to the 5-class target of about `-0.016 V` per exposure.
This is a circuit-side scaling problem, not a preprocessing problem.

The first end-to-end 5-class checks also show that this balance condition is
only necessary, not sufficient.  With `dn_selector_ratio = 0.4`, the total
signed row drift over the 10-sample balanced run was near zero, but score
accuracy stayed at `50%` and the accumulated common-mode movement was still
about `10.6x` the largest signed weight movement.  Pulling both branches back
toward the center during the primitive write did not fix the issue; it reduced
target mobility, strengthened non-target dominance, and increased the
primitive common/signed ratio.  The next circuit change should therefore attack
write common-mode at the branch-update topology itself, not rely on a weak
post-hoc center pull.

The newer `bounded_charge_only` reversed writer with `synapse_spike` pretrace
confirms the distinction.  In the matched 5-class primitive it is sign-correct:

```text
target row delta       ~= +18.7 mV
nontarget row delta    ~=  -5.36 mV
signed epoch drift     ~=  -2.7 mV
forward-read common    ~=  18 uV max
```

So the full classifier failure is not explained by preprocessing, simulator
choice, or forward-read disturb.  The problem is that each useful write has
same-sign common movement:

```text
target common delta    ~= -18.7 mV
nontarget common delta ~=  -5.38 mV
common epoch drift     ~= -40.2 mV per balanced 5-class row
```

Over multiple epochs that common-mode drift is large enough to dominate the
useful signed motion.  Weak center-pull variants from `1e-5` to `2e-4 u` barely
changed this because the branch movement happens inside the selected update
stack.  The next readout cell should therefore be a genuinely complementary
writer: target events should increase one branch while decreasing the other,
and non-target events should do the opposite, with each event sized for near
zero branch-common movement.  A center pull can still be useful for retention
and recovery, but it should not be the main common-mode cancellation mechanism.

The first complementary writer primitive confirms that branch-common cancellation
is the right local diagnostic.  With a 5-class loop, `charge = 5e-4 u` and
`discharge = 5e-6 u` gave low common-mode motion:

```text
target signed delta       ~= +32.5 mV
nontarget signed delta    ~= -32.4 mV
target common delta       ~=  +2.4 mV
nontarget common delta    ~=  +2.3 mV
row common/signed ratio   ~= 0.074
```

That cell is not yet class-count balanced: for one-vs-rest training the
nontarget signed step is too large by roughly the class-count factor.  Attempts
to correct only the non-target selector charge made the signed epoch drift
better, but reintroduced large negative common-mode drift.  The next design
rule is therefore stricter: class-count scaling must preserve per-event
branch-common cancellation, not just signed one-vs-rest balance.

## Bias Updates

Output bias capacitors should be modeled as readout synapses with a constant
pre-activation feature.  That means the bias update should not become an
independent topology-specific tuning surface.  Once the bias-write cell is
stable enough to keep enabled, its width should be derived from the same
readout mobility rule:

```text
output_bias_update_width_u
    = fixed_bias_mobility_ratio * readout_update_width_u
```

where `fixed_bias_mobility_ratio` is a circuit-family constant, not an
experiment-by-experiment hyperparameter.  The reason for allowing a fixed ratio
is physical: the bias cell does not have the same hidden-activation pass device
as an ordinary readout synapse, so equal drawn width may not mean equal mobility.
But the ratio should be characterized once for the cell family and then held
fixed.

The current PMOS charge/discharge derived profile uses
`fixed_bias_mobility_ratio = 1.0` for the 3-class anchor.  This was promoted
from the previous disabled-bias default after the small MNIST3 fixed8 probe
improved from a 0.583 raw/centered mean objective to 0.625 when output-bias
capacitor writes were enabled at the same local mobility as ordinary readout
weights.  The 30-sample MNIST3 check also improved slightly.

For class counts above the 3-class anchor, the profile currently sets the
default bias mobility ratio to 0.0 unless explicitly overridden.  The first
MNIST5 fixed8 probes degraded when the same bias mobility was used, even at
half strength, so larger output banks need a separate bias-cell characterization
before bias writes become the default there.  Centered-score accuracy remains
useful diagnostic evidence, but the target architecture needs the correction to
come from on-chip state, so bias capacitors should remain part of the derived
write family rather than becoming an ad hoc preprocessing correction.

The class count is part of the circuit topology.  The tuner infers it from the
effective dataset name, including command-line dataset overrides.  For example:

```text
mnist3fixed8_30 -> residual_output_width_u = 96 / 2
mnist5fixed8_25 -> residual_output_width_u = 96 / 4
mnistfixed8_20  -> residual_output_width_u = 96 / 9
```

This lets the same derived profile scale from 3 to 5 to 10 classes without
opening a new hyperparameter.

## What This Buys Us

With a broad tuner, changing topology changes the meaning of many knobs, so a
good result can be an accidental fit to one netlist.  With derived profiles,
the question becomes sharper:

```text
Does this circuit family learn over a useful range of global gain, error drive,
and integration time?
```

If it does not, the next move should usually be a circuit change, not another
large hyperparameter search.
