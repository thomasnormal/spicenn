# Engineering Log

## 2026-05-21

- Verified local SPICE tools for this workspace: `ngspice` is installed at `/usr/bin/ngspice` and reports version 42; `Xyce` is installed at `/usr/local/bin/Xyce` and reports 7.10.0.
- Focused SPICENN tests initially could not collect because declared Python dependencies were missing (`scipy`, `PySpice`, then CPU-only `torch`/`torchvision` for MNIST dataset loading). Installed those in the user site packages. Aborted the default CUDA `torch` wheel path and purged the pip cache before installing CPU-only wheels.
- The tiny MNIST dataset files are not present under this repo's `data/` path. The loader deliberately uses `download=False`, so MNIST smokes need either a local dataset cache or an explicit dataset-download policy change.
- Reproduced the readout eligibility bottleneck on built-in SPICE smokes: nonzero ReLU activations around 0.06-0.23 V can be useful, but a 0.3 V spike reference leaves readout pretrace gates mostly off, so target errors only update bias or analog fallback paths. A lower reference exposes real hidden eligibility coverage and improves the `sum_extremes` smoke.
- Downloaded MNIST through the existing high-level `load_mnist_sequence` path. The shared `spice/datasets.py` path was the inconsistent one: it hard-coded `download=False`, which blocked sparse/device MNIST smokes on a fresh checkout. Added an explicit download flag instead of silently changing default network behavior.
- After the dataset path was unblocked, `run_device_spicenn_sparse_forward.py --dataset mnist3fixed8_3 --download-dataset ...` completed electrically with full pretrace coverage, but final sparse-device evaluation was still only 1/3. The immediate blocker moved from dataset plumbing back to architecture/training quality.
