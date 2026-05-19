# Behavioral SPICE Backward Pass

File: `spice/run_spice_backprop_xor.py`

This is a minimal proof that training can be represented inside SPICE rather
than only in PyTorch. It trains a 2-2-1 tanh MLP on XOR.

## Data Cycling

Training examples are voltage waveforms:

```text
Vx1     x1     0   PWL(...)
Vx2     x2     0   PWL(...)
Vt      target 0   PWL(...)
Vu      upd    0   PWL(...)
```

Each sample occupies one time slot. The input and target are held steady, then
`upd` goes high after a settle interval. This models:

```text
load sample -> forward settle -> update pulse -> next sample
```

For MNIST-scale hardware this implies an input sequencer or sensor frontend
that cycles data points through tiles. Even if the tile internals become
self-timed, sample loading and train/infer mode still need coarse control.

## Weight Storage

Each trainable weight is a capacitor node in the behavioral training demo:

```text
Cw11 w11 0 {CW} IC=...
```

This is a proxy for programmable conductance state. In physical hardware the
persistent weight should be a conductance/resistance cell or SRAM-controlled
current source; the capacitor representation makes the update differential
equation explicit in SPICE.

## Forward Pass

```text
h1 = tanh(w11*x1 + w12*x2 + bh1)
h2 = tanh(w21*x1 + w22*x2 + bh2)
y  = tanh(v1*h1 + v2*h2 + bo)
```

## Backward Pass

```text
err = target - y
do  = err * (1 - y^2)
dh1 = (1 - h1^2) * v1 * do
dh2 = (1 - h2^2) * v2 * do
```

These are behavioral voltage nodes in ngspice.

## Update Currents

Programming current sources update the weight-state capacitors:

```text
C * dV(w11)/dt = LR * upd * dh1 * x1
C * dV(v1)/dt  = LR * upd * do  * h1
```

This is backpropagation as a mixed-signal circuit abstraction: error voltages
and local activity voltages drive programming pulses.

## Caveat

This is not yet MNIST training in SPICE. Full MNIST entirely in transient SPICE
would be prohibitively slow at the current behavioral-netlist granularity. The
purpose is to validate the circuit-level training primitives: data cycling,
error propagation, and programmable-weight updates.

## MNIST Behavioral SPICE Demo

File: `spice/run_spice_mnist_train.py`

This generator creates a 10-class SPICE classifier for downsampled MNIST:

```text
16 image-voltage inputs -> 10 tanh class outputs
```

Each class output has its own target waveform and local error/update path:

```text
y_k = tanh(sum_i w_ki*x_i + b_k)
e_k = target_k - y_k
d_k = e_k * (1 - y_k^2)
C*dV(w_ki)/dt = LR * update_gate * d_k * x_i
```

The current version is intentionally small so ngspice can run it as a transient
simulation. It is real MNIST data, but downsampled and subsetted.
