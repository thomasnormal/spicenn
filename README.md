# Local Neural Hardware Simulator

Reproducible simulation and optimization stack for locally connected neural hardware that is robust to circuit noise. Stochastic operation is allowed, but it is not required; the core requirement is that the architecture remains trainable and accurate when realistic noise, drift, and mismatch are present.

The stack separates three layers:

1. SPICE characterization of charge-domain neuron primitives, including analog/multilevel states, local ADCs, and noisy/dithered comparators.
2. PyTorch simulation using expected, sampled, or multi-level circuit-derived activation curves.
3. Architecture/training/energy sweeps across dense, local, hierarchical, relay, and small-world models.

Dense networks are included only as baselines. The main metrics are accuracy, inference energy, update energy, wire length, routing proxies, and robustness.

## Environment

Preferred:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This workspace currently has `ngspice` installed via Homebrew. Check with:

```bash
python3 spice/run_spice_sweep.py --check-only
```

If no SPICE simulator is available, the higher-level simulator still works with analytic probit/logistic activations.

## Quick Start

Run the fast digits smoke test:

```bash
python3 experiments/00_smoke_digits.py --epochs 3
```

Run the behavioral SPICE activation sweep:

```bash
python3 experiments/01_spice_activation_curve.py --trials 25 --points 17
```

Train and evaluate the current hardware-oriented MNIST candidate:

```bash
python3 experiments/07_hardware_mnist_candidate.py --download --epochs 5
```

This model uses only local convolutions plus class evidence maps and global
averaging, then evaluates expected and finite-cycle stochastic activations.

Compare binary, bipolar, thermometer, and multi-bit charge/value activations:

```bash
python3 experiments/09_value_activation_sweep.py --epochs 2
```

Run behavioral SPICE for a charge-domain multi-level ADC neuron:

```bash
python3 spice/run_charge_adc_sweep.py --bits 2,3,4 --trials 4
```

Train the local MNIST model with the measured ngspice charge-ADC transfer curve
used as the PyTorch activation LUT:

```bash
python3 experiments/17_spice_lut_mnist_calibration.py --epochs 4 --train-limit 10000 --test-limit 2000 --include-ideal-baseline
```

The SPICE-LUT path is closer to the proposed circuit than the ideal quantized
ReLU6 model, but it is not a full SPICE MNIST implementation. Convolution,
pooling, BatchNorm, readout, and training updates still run in PyTorch.

Run behavioral SPICE for a time-coded charge neuron:

```bash
python3 spice/run_time_neuron_sweep.py --trials 4
```

Run behavioral SPICE for a pulse-width charge neuron:

```bash
python3 spice/run_pulse_width_sweep.py --trials 3
```

Evaluate robustness of the saved best full-MNIST candidate:

```bash
python3 experiments/11_best_candidate_robustness.py
```

Try a layer-less recurrent local sheet where all cells update in parallel for a
small number of ticks:

```bash
python3 experiments/12_layerless_recurrent_sheet.py
```

Export the best checkpoint as differential conductance weights:

```bash
python3 experiments/13_export_conductance_weights.py
```

Run a small network whose forward pass, backward pass, and weight updates all
happen inside behavioral SPICE:

```bash
python3 spice/run_spice_backprop_xor.py --epochs 80
```

Run an end-to-end behavioral SPICE training demo on real MNIST samples
downsampled to a small input vector:

```bash
python3 spice/run_spice_mnist_train.py --train-samples 80 --image-size 4 --epochs 5
```

Run the chunked version of the all-SPICE trainer, which carries programmable
weight capacitor states from one ngspice chunk to the next:

```bash
python3 spice/run_spice_mnist_stream_train.py --train-samples 100 --test-samples 100 --image-size 8 --epochs 2 --chunk-size 25
```

This keeps forward/error/update equations inside ngspice for each chunk. It is
still too slow and too inaccurate for the final goal; the saved `8x8` run took
about 329 s and reached 42% held-out accuracy.

Run a faster batch operating-point trainer, where ngspice computes a whole
batch's forward pass, error signals, and programmable-state update in one DC
solve:

```bash
python3 spice/run_spice_mnist_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 8 --epochs 20 --batch-size 50
```

The current saved `8x8` batch-op run reaches 68.5% best held-out accuracy on
200 held-out samples. It is not the final architecture, but it is the fastest
all-SPICE training path currently in the repo.

Run the first local all-SPICE batch-op trainer, where each class has trainable
local block evidence and ngspice computes local nonlinear evidence plus updates:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 8 --block-size 4 --epochs 20 --batch-size 50
```

The saved local block-evidence run reaches 68.0% on 200 held-out samples. It is
local and trainable in SPICE, but still far below the full-MNIST target.
This is an analog/multilevel model, not a 0/1 stochastic-bit model: local
evidence is represented by continuous voltage states. The default local
nonlinearity is `tanh`, and the same script can also emit SPICE algebraic
`relu`, `clipped-relu`, and `diff-clipped-relu` local activations. The
`diff-clipped-relu` mode is a signed bounded voltage implemented as two
rectifier branches, `clip_relu(a) - clip_relu(-a)`, which is a more natural
electronics replacement for tanh than an ideal sigmoid/tanh primitive. The
script can inject SPICE-side input noise, weight mismatch, local offset, and
output offset during training/evaluation:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 8 --block-size 4 --epochs 20 --batch-size 50 --train-weight-mismatch-sigma 0.01 --eval-weight-mismatch-sigma 0.01 --eval-local-offset-sigma 0.01 --robustness-sigmas 0,0.01,0.03
```

The ReLU forms are implemented directly in the generated ngspice behavioral
netlist as `0.5*(a+abs(a))`; clipped ReLU subtracts a second shifted ReLU.
That avoids simulator-specific `if(...)` expressions and is closer to an
analog rectifier/saturation circuit. A controlled 14x14, four-block,
200-train/200-held-out subset check with two epochs and batch size 100 produced
these all-SPICE results:

```text
tanh + tanh output:                  best 35.5%, final 32.5%
relu + tanh output:                  best 23.5%, final 23.5%
clipped-relu + tanh output:          best 23.5%, final 23.5%
relu + softmax class competition:    best 41.0%, final 41.0%
clipped-relu + softmax competition:  best 41.0%, final 41.0%
diff-clipped-relu + tanh output:     best 34.5%, final 32.5%
diff-clipped-relu + softmax:         best 55.5%, final 55.5%
```

Continuing the `diff-clipped-relu + softmax` checkpoint for three more
200-sample epochs reached 66.5% on 200 held-out images and 69.0% when evaluated
on 1,000 held-out images. Continuing on 1,000 train / 1,000 held-out samples
reached 78.0%, 79.9%, then 80.9% over three one-epoch continuation runs. These
are preliminary all-SPICE subset results, not full-MNIST evidence. They show
that the differential rectifier is trainable and hardware-plausible, but the
older tanh-style path still has the best saved local all-SPICE accuracy.

Increasing the same architecture to 14x14 inputs with four 7x7 local blocks is
slower but currently gives the best local all-SPICE batch-op result. The saved
run reached 66.0% after 5 epochs, then 71.0% after 3 resumed epochs from the
saved programmable state:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 14 --block-size 7 --epochs 5 --batch-size 25 --lr 0.2
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 14 --block-size 7 --epochs 3 --batch-size 50 --lr 0.2 --init-weights spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e5_lr02_final_weights.npz
```

The same checkpoint can be evaluated without another update pass:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 14 --block-size 7 --epochs 0 --eval-only --batch-size 50 --init-weights spice/results/spice_mnist_local_block_local_block_14x14_b7_200_cont3_lr02_b50_final_weights.npz --robustness-sigmas 0,0.01 --robustness-repeats 2
```

That saved robustness check held at 71.0% with zero perturbation and about
70.25% mean accuracy at normalized perturbation sigma 0.01.

Scaling the same checkpoint to 1,000 training samples and 1,000 held-out
samples continues to improve accuracy, but at about 8.5 minutes per epoch on
this machine:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 1000 --test-samples 1000 --image-size 14 --block-size 7 --epochs 1 --batch-size 50 --lr 0.1 --init-weights spice/results/spice_mnist_local_block_local_block_14x14_b7_200_cont3_lr02_b50_final_weights.npz
```

The resumed 1,000-sample sequence reached 76.4%, 81.0%, 84.1%, then 85.0%
held-out accuracy over four one-epoch continuation runs. Continuing from that
checkpoint on 2,000 training samples reached 86.0%, 86.9%, 87.5%, then 87.7%
on the 1,000-sample held-out set. Enabling trainable per-class/per-block output
gains on top of that checkpoint reached 87.9%; a second conservative gain-update
continuation at `lr=0.005` stayed at 87.9%.
I also added an optional centered overlapping block (`--add-center-block`) as a
minimal capacity increase over the four regular 7x7 blocks. Expanding the 87.9%
checkpoint into the five-block model reproduced 87.9% on the 1,000-image held-out
set, proving the new block starts neutral, but one full 2,000/1,000 training
epoch with the centered block also stayed at 87.9%.
Expanding the same checkpoint into a two-channel local-template model, with the
original template in channel 0 and a neutral trainable template in channel 1,
reproduced 87.9% and then reached 88.0% after one all-SPICE epoch. A second
two-channel continuation stayed at 88.0%, and each epoch took about 35 minutes,
so this is now the best subset result but not a practical route to full MNIST by
itself. I then added class-chunking to the multichannel trainer for independent
tanh/linear class outputs; the chunked two-channel continuation preserved 88.0%
while reducing one epoch to about 17.6 minutes.
Continuing that class-chunked two-channel checkpoint on 4,000 training samples
with a lower learning rate took about 33.1 minutes and dropped to 87.2%, so
adding more samples alone did not break the 88.0% subset plateau.
Expanding the two-channel 88.0% checkpoint to three local-template channels also
started neutral and stayed at 88.0% after one class-chunked epoch, so extra
within-block template capacity alone is not breaking the plateau.
Continuing the same best checkpoint with SPICE-computed softmax class-competition
updates at `lr=0.001` also stayed at 87.9%, so changing the output loss/update
signal alone did not break the plateau.
I also added an eval-only SPICE ensemble diagnostic that sums branch scores
inside ngspice. A one-branch control reproduced 87.9%, but combining the 87.9%
tanh branch with the 80.9% differential-clipped-ReLU softmax branch fell to
86.7%. Sweeping the second-branch gain using SPICE-computed branch scores found
the best gain was 0.0, so simple replicated local classifiers are not a shortcut
past this plateau.
I also tested whether the plateau is only a readout-calibration problem by
freezing the 87.9% local block evidence circuit and training a small programmable
10x10 class mixer inside ngspice:

```bash
python3 spice/run_spice_mnist_local_readout_calibrator.py --checkpoint spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e6_frome5_traingains_lr0005_b100_final_weights.npz --train-samples 2000 --test-samples 1000 --image-size 14 --block-size 7 --stride 7 --epochs 1 --batch-size 100 --lr 0.02 --identity-scale 4.0
```

The eval-only calibrator reproduced 87.9%; one conservative mixer epoch with
identity scale 1.0 fell to 87.7%, and identity scale 4.0 stayed at 87.9%. This
small global readout is programmable and SPICE-trained, but it did not break the
plateau. I also exposed all 40 class/block local evidence features to a 10x40
trainable mixer; that also fell from 87.9% to 87.7%. Results are in
`results/tables/spice_local_readout_calibration.csv`.
This is the current best all-SPICE local result, but still not full MNIST and
still below 90%.
Continuing the 87.7% checkpoint for one epoch on 4,000 training samples with
`lr=0.03` reached 87.5%, so simply adding more samples to this four-block model
did not improve the current best checkpoint.
One-draw eval-only robustness checks held the 85.0% checkpoint at 84.8% and the
87.7% checkpoint at 87.7% under normalized perturbation sigma 0.01.

An overlapping-block variant with stride 3 creates 9 local 7x7 blocks on the
same 14x14 input. Batch size 100 timed out at the default 90 s per ngspice solve;
batch size 50 completed one 200-train / 200-held-out epoch in about 272 s and
reached only 21% held-out. Overlap may still be useful, but the current batch-op
netlist becomes too large before it gives enough learning benefit.

A first full-resolution local feasibility run used 28x28 inputs with four
non-overlapping 14x14 blocks. It completed one 100-train / 100-held-out
all-SPICE epoch with batch size 25 and reached 32% held-out, but required about
538 s. This proves the generated SPICE path can use full MNIST resolution, but
the direct batch-op netlist is too slow to scale naively to full 60k/10k MNIST.
The local block trainer can now initialize a 28x28 / 14x14-block model from a
14x14 / 7x7-block checkpoint by 2x upsampling each local filter and scaling the
replicated weights by 1/4. Eval-only checks using the 87.9% 14x14 checkpoint
reached 88.0% on a 200-image held-out sample and 87.9% on the 1,000-image
held-out split. A bounded full-resolution fine-tune, using 200 shuffled training
samples from the same 2,000-sample split and evaluating on the same 1,000-image
held-out split, reached 88.1%; a second bounded epoch stayed at 88.1%. This is
the current best all-SPICE local subset result, but still far below the full
MNIST >90% target and very slow at about 16.5 minutes per bounded epoch.
For independent tanh/linear class-evidence outputs, the trainer also supports
class chunking:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 100 --test-samples 100 --image-size 28 --block-size 14 --epochs 1 --batch-size 25 --class-chunk-size 1
```

The class-chunked 28x28 run reproduced 32% held-out accuracy while reducing
wall time from about 538 s to 305 s. This is not enough for full MNIST yet, but
it is the first useful factorization of the SPICE solve size.
With batch size 50, the same 100/100 full-resolution class-chunked setup reached
41% after one epoch in about 307 s, but a second continuation epoch fell to 33%.
That means class chunking improves feasibility, not the underlying four-block
learning capacity.

The local block trainer also supports a simpler linear analog class-evidence
readout and a SPICE-computed softmax class-competition readout:

```bash
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 8 --block-size 4 --epochs 10 --batch-size 50 --linear-output
python3 spice/run_spice_mnist_local_block_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 8 --block-size 4 --epochs 10 --batch-size 50 --lr 0.5 --softmax-output
```

The saved linear-readout run reached 67.0% best held-out accuracy on the same
200-sample 8x8 setup, and the saved softmax-readout run reached 64.0%. Neither
improved the current local baseline.

Run a more biological random sparse hidden network, where each hidden cell has
a random sparse local input fan-in plus optional shortcut inputs, a simple
bounded activation, and SPICE-computed backprop through both trainable layers:

```bash
python3 spice/run_spice_mnist_sparse_random_train.py --train-samples 100 --test-samples 100 --image-size 8 --hidden 32 --fan-in 16 --radius 4 --shortcut-fraction 0.2 --epochs 5 --batch-size 20 --lr 1.0 --activation diff-clipped-relu --output-mode softmax
```

This sparse trainer also lets the generated SPICE netlist encode gradients in
different ways:

```bash
--gradient-mode analog
--gradient-mode clipped --gradient-clip 1.0
--gradient-mode quantized --gradient-bits 4 --gradient-clip 1.0
--gradient-mode symmetric-quantized --gradient-bits 4 --gradient-clip 1.0
--gradient-mode pulse-count --gradient-bits 4 --gradient-clip 1.0
--gradient-mode pulse-dithered --gradient-bits 4 --gradient-clip 1.0
--gradient-mode pulse-residual --gradient-bits 4 --gradient-clip 1.0
```

On a 100-train / 100-held-out 8x8 subset, the initial sparse result was weak
with analog gradients: 34% best held-out for a stronger initialization. The
same sparse network with 4-bit quantized gradients reached 72% best held-out,
while 8-bit quantized gradients reached 34% and analog `lr=5.0` reached 19%.
I then split the finite-update idea into more electronic sign/magnitude
encodings: symmetric 4-bit quantization peaked at 29%, plain 4-bit pulse-count
updates peaked at 29%, and deterministic dithered 4-bit pulse-count updates
reached 47%. Resuming the dithered-pulse checkpoint for five more epochs at
`lr=0.5` reached 50%. I also added a more hardware-natural residual pulse mode,
where each synapse keeps a local analog gradient residue and emits integer
programming pulses only when the residue crosses the pulse quantum; in the same
100/100 setup it reached 30%. So the early result is not “more gradient precision
is always better”, and not even “any 4-bit pulse code works”; the legacy quantizer's
offset or dead-zone behavior is probably acting like a learning bias. The
comparison is saved in `results/tables/spice_sparse_random_gradient_precision.csv`.
I also added direct feedback alignment to this sparse trainer:

```bash
python3 spice/run_spice_mnist_sparse_random_train.py --hidden-error-rule dfa --feedback-scale 0.3
```

In DFA mode, each hidden cell receives a fixed random mix of the 10 class-error
voltages instead of transporting the trainable output weights backward. On the
same 100/100 8x8 sparse setup, DFA with the legacy 4-bit quantized gradient path
reached 61% after five epochs. A five-epoch continuation at `lr=0.5` peaked at
75% and finished at 70%, roughly matching the older 72% exact-backprop sparse
baseline but with a more hardware-plausible hidden error path. Continuing that
DFA checkpoint with deterministic dithered pulse updates reached 79%, the best
sparse hardware-plausible update result so far, but scaling the same checkpoint
to a 200/200 split reached only 71.5%. The sparse trainer now saves best
checkpoints for future unstable runs.
The current design note for this direction is
`results/biological_sparse_precision_notes.md`: it argues for sparse recurrent
sheets, simple rectifying/saturating cells, and gradient precision through
pulse-count, dither, mini-batch charge, and residual-charge accumulation rather
than 4-bit floating-point wires.

Run a more layerless/biological random sparse recurrent sheet. Each hidden cell
has local random input fan-in, local recurrent fan-in plus optional shortcuts,
and all cells update in parallel for a small number of local ticks. The generated
ngspice netlist unrolls those ticks, computes softmax error, computes recurrent
backprop through time or direct feedback-alignment hidden errors, and updates
the programmable weights:

```bash
python3 spice/run_spice_mnist_recurrent_sparse_sheet_train.py --train-samples 100 --test-samples 100 --image-size 8 --hidden 16 --input-fan-in 8 --recurrent-fan-in 4 --ticks 3 --epochs 5 --batch-size 20 --lr 1.0 --activation diff-clipped-relu --gradient-mode quantized --gradient-bits 4
```

For the more biological hidden-error path, add:

```bash
--hidden-error-rule dfa --feedback-scale 0.3
```

Initial all-SPICE results are promising enough to keep but not yet competitive:
the analog-gradient recurrent sheet reached 29% on a 100/100 8x8 subset, while
the 16-cell 4-bit quantized-gradient version reached 68% after a continuation
and 69% after continuing on a 200/200 subset. A 32-cell version reached 74% best
on 100/100 and 73% on a 200/200 scale check. Adding fixed self-memory 0.3 and
local inhibition 0.1 improved the best 100/100 result to 76% and reached 75.5%
best on 200/200, but the 200/200 final epoch fell to 68%, so that branch is
useful but unstable. A residual-pulse smoke run also executed, but reached only
20% on 40/40. A proper 100/100 residual-pulse run with self-memory and local
inhibition reached 29%, and increasing pulse density with a smaller fixed pulse
quantum reached only 30%. That suggests residual pulse programming is functional
but not yet solving recurrent credit assignment. Recurrent DFA is more promising:
the 16-cell DFA branch reached 56% after five epochs and 65% best after a
continuation, compared with 55% and 68% for exact BPTT. The stronger 32-cell
self-memory/local-inhibition DFA branch peaked at 72%, below the 76% exact-BPTT
recurrent best but close enough to keep as a hardware-plausible training rule.
Scaling the best 32-cell recurrent DFA checkpoint to 200/200 peaked at 67.5%,
so this branch currently loses stability with more samples.
New recurrent-sheet runs save both final and best weight checkpoints. These
results are saved in
`results/tables/spice_recurrent_sparse_sheet_comparison.csv`.

Run a shared local class-evidence variant, where class-specific local kernels
are reused across scanned sheet positions and ngspice computes the shared weight
updates:

```bash
python3 spice/run_spice_mnist_shared_local_evidence_train.py --train-samples 100 --test-samples 100 --image-size 8 --kernel-size 3 --stride 2 --channels 2 --epochs 5 --batch-size 20
```

This path is closer to a local convolutional tile, but early 8x8 checks were
slow and weak. It remains an experimental variant, not a candidate result.

Run a multichannel unshared local-block variant, where each class/block owns
multiple analog evidence cells:

```bash
python3 spice/run_spice_mnist_local_block_multichannel_train.py --train-samples 100 --test-samples 100 --image-size 8 --block-size 4 --channels 2 --epochs 5 --batch-size 25
```

The trainer now supports trainable gains, softmax class competition, and the
same local activation options as the single-channel local trainer. The saved
two-channel fixed-gain checks stalled around 30% held-out accuracy; the patched
trainable-gain `diff-clipped-relu + softmax` check reached 37% on a 100/100 8x8
subset. So naive channel duplication is still worse than the one-cell-per-class
block baseline. Results are summarized in
`results/tables/spice_multichannel_local_comparison.csv`.

A local feature/readout variant is also available:

```bash
python3 spice/run_spice_mnist_local_feature_batch_op_train.py --train-samples 200 --test-samples 200 --image-size 8 --block-size 4 --channels 4 --epochs 10 --batch-size 50 --lr 1.0
```

It learns local features shared across class readout. The original tanh run
reached only 46.5% held-out accuracy on a 200-train / 200-held-out 8x8 subset.
After adding softmax output, rectified local activation options, and checkpoint
resume support, a `diff-clipped-relu + softmax` run reached 67% on 100/100 and
a 200/200 continuation reached 73% best, 71.5% final. Scaling the same shared
feature formulation to 14x14 with four 7x7 blocks reached 75% best, 74.5% final
on a 200/200 subset. Doubling the 14x14 feature channels from 4 to 8 peaked at
74.5% and took much longer, so width alone did not improve this branch. That is
still below the best 14x14 local block path, but it is a better capacity signal
than the multichannel class-specific branch. Results are summarized in
`results/tables/spice_local_feature_comparison.csv`.

Run SPICE-evaluated forward inference on downsampled MNIST with offline-trained
programmable weights:

```bash
python3 spice/run_spice_mnist_inference.py --hidden 32 --test-samples 200 --tag mlp14x14_h32_200_op
```

The current saved run reaches `92.0%` on 200 held-out MNIST samples with exact
Python/SPICE prediction agreement. This is not SPICE training and not the full
10k MNIST test set.

Run a smaller sklearn digits benchmark whose held-out forward pass is evaluated
by ngspice:

```bash
python3 spice/run_spice_digits_inference.py --tag logreg
```

Run a hidden-layer MLP version, also trained/evaluated by ngspice:

```bash
python3 spice/run_spice_mnist_mlp_train.py --train-samples 80 --test-samples 80 --image-size 4 --hidden 8 --epochs 8
```

Summarize SPICE-only MNIST training runs:

```bash
python3 experiments/15_summarize_spice_mnist_training.py
```

Explore the MNIST architecture frontier between training wall time, accuracy,
and hardware energy/wire proxies:

```bash
python3 experiments/16_mnist_time_accuracy_frontier.py
```

Estimate training sample-cycling and programmable-conductance update energy:

```bash
python3 experiments/14_training_cycle_energy.py
```

Run tests:

```bash
python3 -m pytest tests
```

## Outputs

Generated artifacts are written under:

- `spice/results/`
- `results/tables/`
- `results/figures/`
- `results/raw/`

Important generated files include:

- `results/tables/smoke_digits.csv`
- `results/figures/smoke_digits_accuracy.png`
- `spice/results/activation_curve.csv`
- `spice/results/activation_curve_fit.json`
- `spice/results/activation_curve.png`
- `results/tables/spice_lut_mnist_calibration.csv`
- `results/figures/spice_lut_accuracy_vs_input_scale.png`

## Current Scope

The repository includes:

- behavioral SPICE templates and a simulator-detecting sweep runner
- fitted probit/logistic activation extraction
- expected and sampled stochastic threshold activations
- analog/multilevel local voltage-state training with explicit noise/mismatch injection in SPICE
- dense, shared-conv, unshared-local, hierarchical, relay, and small-world model scaffolds
- absolute-unit energy and wire proxy model
- smoke experiments and placeholder/full experiment entry points
- tests for activation, topology, energy, and training

The real-comparator PDK netlist is intentionally a placeholder. Add SKY130/GF180 model includes and a latch implementation when the local PDK path is known.

## Reproducibility

Scripts fix and log seeds. Result tables include configuration columns where practical. Do not treat a reported result as real unless the corresponding script actually generated the CSV/plot in this checkout.

## Candidate Tile Direction

A first physical prototype target to evaluate with the generated sweeps:

- 2D local analog/multilevel tile
- 3x3 or 5x5 local receptive field
- 4-16 local channels
- conductance-weighted summing into a 3-30 fF integration capacitor
- saturating charge/voltage state with an optional local ADC or comparator
- stochastic cycles only if they improve energy/robustness; deterministic multi-level charge/voltage codes are the default direction
- optional local inhibition and sparse trainable shortcuts
- local auxiliary or direct-feedback-alignment update path for on-device learning experiments
