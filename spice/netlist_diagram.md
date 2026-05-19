# SPICE Netlist Diagrams

These are node-level diagrams for the behavioral SPICE circuits currently used
by the simulator.

## Binary Noisy Comparator

File: `spice/templates/noisy_comparator_behavioral.cir`

```text
          Vs
    sig o---( DC S )---o 0
        |
        |   V(sig)
        v
   +-----------------------------+
   | behavioral comparator Bcmp  |
   | out = VDD *                 |
   |       (V(sig)+V(n) > VTH)   |
   +-----------------------------+
        ^                 |
        | V(n)            v
    n o---( TRNOISE )---o out
      Vn                 |
                         |
                         o 0
```

Extracted object:

```text
P(out = VDD | S)
```

This is the one-bit stochastic activation. The comparator threshold is the hard
nonlinearity; the transient noise turns it into a graded probability curve.

## Multi-Comparator Thermometer Activation

File: `spice/templates/multivalue_thermometer_behavioral.cir`

```text
          Vs
    sig o---( DC S )---o 0
        |
        | V(sig)
        v
  +--------------------------------------------------+
  | Bcode behavioral thermometer ADC                 |
  |                                                  |
  | code = (1/(levels-1)) * sum_k                    |
  |        (V(sig)+V(n) > threshold_k)               |
  |                                                  |
  | out = VDD * code                                 |
  +--------------------------------------------------+
        ^                                      |
        | V(n)                                 v
    n o---( TRNOISE )---o 0                  out
      Vn                                      |
                                              o 0
```

For `bits = b`:

```text
levels = 2^b
number of comparators = levels - 1
normalized output code in [0, 1]
bipolar code = 2 * code - 1
```

This models an ADC-like local neuron output using multiple noisy thresholds
instead of a single binary comparator.

## Charge Integrator + Multi-Level ADC

File: `spice/templates/charge_adc_behavioral.cir`

```text
                    Vdrive
    in o------------( PULSE 0 -> VIN for TAU )------------o 0
       |
       | V(in)
       v
  +------------------------------------------------+
  | Gsyn behavioral conductance/current source     |
  | I = GSUM * V(in)                               |
  +------------------------------------------------+
       |
       | injected current
       v
    int o---------------------+
        |                     |
        |                     |
       === Cint               |
       === CINT               |
        |                     |
        o 0                   |
                              |
                              | V(int)
                              v
  +--------------------------------------------------+
  | Bcode behavioral noisy ADC                       |
  |                                                  |
  | code = (1/(levels-1)) * sum_k                    |
  |        (V(int)+V(n) > threshold_k)               |
  |                                                  |
  | out = VDD * code                                 |
  +--------------------------------------------------+
        ^                                      |
        | V(n)                                 v
    n o---( TRNOISE )---o 0                  out
      Vn                                      |
                                              o 0
```

Approximate signal scale:

```text
Q_sig ~= TAU * GSUM * VIN
V_int ~= Q_sig / CINT
```

Measured quantities:

```text
vint = V(int) at TDECIDE
y    = V(out) at TDECIDE
ecap = 0.5 * CINT * vint^2
```

This is the current best behavioral SPICE abstraction for a wider-value local
neuron: local conductance-weighted integration followed by a small local
multi-threshold ADC.

## Time-To-Threshold Charge Neuron

File: `spice/templates/time_to_threshold_neuron.cir`

```text
                    Vdrive
    in o------------( PULSE 0 -> VIN )-------------------o 0
       |
       v
  +------------------------------------------------+
  | Bsyn current source                            |
  | I = GSUM * V(in)                               |
  +------------------------------------------------+
       |
       v
    int o---------------------+
        |                     |
       === Cint               |
       === CINT               |
        |                     |
        o 0                   |
                              |
                              | V(int)
                              v
  +--------------------------------------------------+
  | Bspk threshold comparator                        |
  | spk = VDD * (V(int)+V(noise) > VTH)              |
  +--------------------------------------------------+
```

Measured code:

```text
tfire = first threshold crossing time
activation ~= fast spike -> large value
```

This avoids a multi-bit value wire. The activation can be represented by pulse
timing, pulse width, or a local downstream integration window.

## Partial-Sharing Phase-Resolved Local Feature Network

Files:

- `spice/run_spice_mnist_partial_sharing_phase_transient.py`
- `spice/run_spice_mnist_partial_sharing_phase_train.py`

Generated architecture overview:

![Current best SPICE NN architectures](../results/figures/current_best_spice_nn_architectures_imagegen.png)

This is the compact accelerator branch used for the 784-state and 854-state
partial-sharing candidates. It is not the purest form of the local-synapse
architecture: a shared kernel capacitor is a single physical state node whose
voltage is broadcast to several block-local multiplier/synapse devices, or
equivalently a kernel bank that would be time-multiplexed across blocks in a
more sequential implementation. The fully private local-feature decks remain
the cleaner match for "each neuron/synapse owns its own weight capacitor."

The diagram below is a two-block, two-class toy netlist with the same topology
rule as the larger partial-sharing MNIST decks:

- shared local kernel capacitor banks are reused by every image block
- private local kernel capacitor banks, when present, are per block
- activations, output scores, output deltas, hidden deltas, gradient
  accumulators, weights, readout weights, and biases are capacitor nodes
- Python only drives phase waveforms and restarts chunks from checkpoint ICs

```text
Python phase sources
--------------------
Vpact   pact   0  PWL(...)    forward local activation store
Vpscore pscore 0  PWL(...)    forward score store
Vperr   perr   0  PWL(...)    output error/delta store
Vpbwd   pbwd   0  PWL(...)    hidden-delta store
Vpacc   pacc   0  PWL(...)    gradient accumulation
Vpapply papply 0  PWL(...)    weight update
Vpclear pclear 0  PWL(...)    gradient capacitor clear


State capacitors for one shared channel, one private channel, two blocks
-----------------------------------------------------------------------
Input samples:

  Vpix0 pix0 0 PWL(sample pixels...)
  Vpix1 pix1 0 PWL(sample pixels...)
  Vpix2 pix2 0 PWL(sample pixels...)
  Vpix3 pix3 0 PWL(sample pixels...)

Shared local kernel S0, physically reused across B0 and B1:

  Csw0_0  sw0_0  0 {CW}     IC=wS0[0]
  Csw0_1  sw0_1  0 {CW}     IC=wS0[1]
  Cshb0   shb0   0 {CW}     IC=bS0
  Cgsw0_0 gsw0_0 0 {CGRAD}  IC=0
  Cgsw0_1 gsw0_1 0 {CGRAD}  IC=0
  Cgshb0  gshb0  0 {CGRAD}  IC=0

Private local kernel P0 for B0:

  Cpw0_0_0  pw0_0_0  0 {CW}     IC=wP[B0,0]
  Cpw0_0_1  pw0_0_1  0 {CW}     IC=wP[B0,1]
  Cphb0_0   phb0_0   0 {CW}     IC=bP[B0]
  Cgpw0_0_0 gpw0_0_0 0 {CGRAD}  IC=0
  Cgpw0_0_1 gpw0_0_1 0 {CGRAD}  IC=0
  Cgphb0_0  gphb0_0  0 {CGRAD}  IC=0

Private local kernel P0 for B1:

  Cpw1_0_0  pw1_0_0  0 {CW}     IC=wP[B1,0]
  Cpw1_0_1  pw1_0_1  0 {CW}     IC=wP[B1,1]
  Cphb1_0   phb1_0   0 {CW}     IC=bP[B1]
  Cgpw1_0_0 gpw1_0_0 0 {CGRAD}  IC=0
  Cgpw1_0_1 gpw1_0_1 0 {CGRAD}  IC=0
  Cgphb1_0  gphb1_0  0 {CGRAD}  IC=0

Activation and delta storage:

  Ch0_0  h0_0  0 {CSTATE} IC=0      B0 shared-channel activation
  Ch0_1  h0_1  0 {CSTATE} IC=0      B0 private-channel activation
  Ch1_0  h1_0  0 {CSTATE} IC=0      B1 shared-channel activation
  Ch1_1  h1_1  0 {CSTATE} IC=0      B1 private-channel activation
  Cdh0_0 dh0_0 0 {CSTATE} IC=0      B0 shared-channel hidden delta
  Cdh0_1 dh0_1 0 {CSTATE} IC=0      B0 private-channel hidden delta
  Cdh1_0 dh1_0 0 {CSTATE} IC=0      B1 shared-channel hidden delta
  Cdh1_1 dh1_1 0 {CSTATE} IC=0      B1 private-channel hidden delta

Readout, output, and output-gradient state for two classes:

  Cv0_0_0  v0_0_0  0 {CW}     IC=r[Y0,B0,S0]
  Cv0_0_1  v0_0_1  0 {CW}     IC=r[Y0,B0,P0]
  Cv0_1_0  v0_1_0  0 {CW}     IC=r[Y0,B1,S0]
  Cv0_1_1  v0_1_1  0 {CW}     IC=r[Y0,B1,P0]
  Cob0     ob0     0 {CW}     IC=bo[Y0]
  Cscore0  score0  0 {CSTATE} IC=0
  Cd0      d0      0 {CSTATE} IC=0
  Cgv0_0_0 gv0_0_0 0 {CGRAD}  IC=0
  ...

  Cv1_0_0  v1_0_0  0 {CW}     IC=r[Y1,B0,S0]
  ...
  Cob1     ob1     0 {CW}     IC=bo[Y1]
  Cscore1  score1  0 {CSTATE} IC=0
  Cd1      d1      0 {CSTATE} IC=0
  Cgv1_0_0 gv1_0_0 0 {CGRAD}  IC=0
```

Forward phase:

```text
Bstore_h0_0 h0_0 0 I =
  V(pact) * {CSTATE}/{TAU} *
  (V(h0_0) - tanh(V(sw0_0)*V(pix0) + V(sw0_1)*V(pix1) + V(shb0)))

Bstore_h1_0 h1_0 0 I =
  V(pact) * {CSTATE}/{TAU} *
  (V(h1_0) - tanh(V(sw0_0)*V(pix2) + V(sw0_1)*V(pix3) + V(shb0)))

Bstore_h0_1 h0_1 0 I =
  V(pact) * {CSTATE}/{TAU} *
  (V(h0_1) - tanh(V(pw0_0_0)*V(pix0) + V(pw0_0_1)*V(pix1) + V(phb0_0)))

Bscore0 scorecalc0 0 V =
  V(v0_0_0)*V(h0_0) + V(v0_0_1)*V(h0_1) +
  V(v0_1_0)*V(h1_0) + V(v0_1_1)*V(h1_1) + V(ob0)

Bstore_score0 score0 0 I =
  V(pscore) * {CSTATE}/{TAU} * (V(score0) - V(scorecalc0))
```

Backward and accumulation phase:

```text
Bstore_d0 d0 0 I =
  V(perr) * {CSTATE}/{TAU} *
  (V(d0) - ((V(target0)-tanh(V(score0))) * (1-tanh(V(score0))^2)))

Bstore_dh0_0 dh0_0 0 I =
  V(pbwd) * {CSTATE}/{TAU} *
  (V(dh0_0) - ((V(v0_0_0)*V(d0) + V(v1_0_0)*V(d1)) * (1-V(h0_0)^2)))

Bacc_sw0_0 gsw0_0 0 I =
  -V(pacc) * {CGRAD}/{TPHASE} *
  (V(dh0_0)*V(pix0) + V(dh1_0)*V(pix2))

Bacc_v0_0_0 gv0_0_0 0 I =
  -V(pacc) * {CGRAD}/{TPHASE} * V(d0)*V(h0_0)
```

Update and clear phase:

```text
Bupd_sw0_0 sw0_0 0 I =
  -V(papply) * {CW}*{LR}/({BS}*{TPHASE}) * V(gsw0_0)

Bupd_v0_0_0 v0_0_0 0 I =
  -V(papply) * {CW}*{LR}/({BS}*{TPHASE}) * V(gv0_0_0)

Bclear_gsw0_0 gsw0_0 0 I =
  V(pclear) * {CGRAD}/{TAU} * V(gsw0_0)
```

The important sharing rule is visible in `Bacc_sw0_0`: one shared gradient
capacitor sums contributions from both blocks before a single `papply` pulse
updates the shared weight capacitor. Private weights use one gradient capacitor
per block, and readout weights remain position-specific.
