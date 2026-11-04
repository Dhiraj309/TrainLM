# Dependency and Compatibility Policy

> **Current core contract:** CPU compatibility CI is configured for the minimum
> and current profiles, including PyTorch 2.14.1. **Native TorchTPU:** a
> supported install profile and end-to-end training remain unverified.

**Matrix:** [`compatibility/dependency_matrix_v1.json`](../compatibility/dependency_matrix_v1.json)

TrainLM separates portable model/trainer dependencies from accelerator-specific
runtime and kernel packages. Core imports must work without CUDA, PyTorch/XLA,
JAX, libtpu, or profiling packages installed.

## Native TorchTPU status

The source-level salvage classification is recorded in
[the T0.1 inventory](TORCHTPU_SALVAGE_INVENTORY.md).

[The T0–T11 rebaseline](TORCHTPU_REBASELINE.md) targets native TorchTPU rather
than PyTorch/XLA as the long-term TPU backend. The owner has reported a v5e-1
eager allocation, forward/backward and optimizer-update smoke with Python
3.13.18, torch 2.14.1+cpu, torch_tpu 0.2.0 and libtpu 0.0.50. This does not
validate TrainLM's full trainer, compilation, lifecycle, distributed execution
or performance.

The current compatibility change widens the core Torch contract to
`>=2.9,<2.15`, raises the Transformers floor to `5.15` and the Hub floor to
`1.5`, and adds CPU CI for both the minimum and current profiles. The current
profile uses Python 3.13 and the owner's observed Torch 2.14.1 package set. Its
first job passed. The initial minimum job exposed tests failing under
Transformers 5.0.0; the first retry then showed Transformers 5.15.0 requires
Hub 1.5.0, so the minimum constraints and project floor now match that resolver
requirement. The corrected matrix run is pending. This does not certify CUDA or
native TPU. No TorchTPU-specific install extra or support claim is added.

The read-only preflight reports environment-wide `pip check` findings. A fully
parsed conflict unrelated to TrainLM's training stack, such as IPython requiring
missing Jedi, remains visible but does not block the TrainLM profile. Conflicts
involving TrainLM, Torch, TorchTPU, Transformers, Hub, PyYAML or datasets, and
unclassified output, still block it. A clean TrainLM environment and a passing
preflight have not yet been re-observed on the owner's runtime.

## Owner-reported native probe profile

The owner supplied repeated passing standalone eager smokes with Python 3.13.18,
torch 2.14.1+cpu, torch_tpu 0.2.0 and libtpu 0.0.50; tpu-info 0.14.2 was
installed. The notebook correctly identifies a source checkout (TrainLM need
not be installed as a distribution to import it). The earlier report's Torch
range mismatch is addressed by the current core contract, while its unrelated
IPython 7.34.0 → missing Jedi finding remains a visible, nonblocking warning.
Explicit compile, lifecycle, distributed and full trainer evidence are pending.

The [utility inventory](TORCHTPU_UTILITIES.md) distinguishes versioned API
documentation from the current public API policy. Experimental utilities
need guarded adapters; monitoring, profiling and kernel extras remain optional.
An experimental `accelerator="native_tpu"` single-device adapter exists behind
explicit opt-in. The widened core range does not create a supported TorchTPU
install profile or certify that adapter on hardware.
The v5e-1 notebook also runs the repository CPU test suite against the runtime's
already installed Python/PyTorch versions. Its result is recorded separately
from TPU discovery and eager execution, and it does not install packages. The
owner run at source `08f02fb` produced 746 passes and 17 failures. Two facade
tests had left `accelerator="auto"` on a TPU host, the wheel smoke required
`venv`/`ensurepip`, and the roadmap contract used an obsolete heading; those
test-harness issues were corrected locally. Twelve loss-layout failures came
from counting ignored targets in the mean denominator, and a scalar-count test
expected one snapshot although the trainer records both loss and gradient
norm. These fixes were made after that run. The initial current-profile CI job
passed. The minimum retry first exposed the Hub lower-bound mismatch; the
minimum lock now uses Hub 1.5.0 alongside Transformers 5.15.0, and the corrected
run is pending.

### Read-only native install-profile check

Run `python examples/torchtpu_t1_install_preflight.py` from a TrainLM checkout,
or use the package-preflight cell in the
[T1 notebook](../notebooks/TorchTPU_T1_v5e1_Probe.ipynb). The command reads
installed distribution metadata, the checkout's `pyproject.toml`, and the
result of `pip check`. It does not import torch, initialize a TPU, install
packages or change the declared dependency range. Its JSON result distinguishes
`metadata_compatible`, `incomplete`, `incompatible` and `check_failed`.
The command exits zero only for `metadata_compatible`; that status validates
package metadata, not device behavior or native trainer support. Its source
requirements now accept Torch 2.14.1. It reports parsed unrelated environment
conflicts without treating them as TrainLM blockers; stack conflicts and
unclassified output remain blocking. Re-run it on the owner runtime to confirm
the current report, and use CI for CPU conformance.

## Existing support profiles (CPU/CUDA and historical XLA)

| Profile | Python | PyTorch | Transformers | Accelerator packages | Status |
|---|---|---|---|---|---|
| Core minimum | 3.10 | 2.9.0 | 5.15.0 | None | Required CPU CI; Hub 1.5.0 |
| Core current | 3.13 | 2.14.1 | 5.20.2 | None | Required CPU CI |
| CUDA portability | 3.13 | 2.14.1 | 5.20.2 | Official matching CUDA wheel | Shared package set; CPU CI does not certify CUDA |
| TPU XLA stable | 3.10–3.13 | 2.9.0 | 5.15.0 | torch-xla 2.9.0, libtpu 0.0.21 | Pending M5 TPU validation |
| TPU XLA + Pallas | 3.10–3.13 | 2.9.0 | 5.15.0 | XLA/libtpu above, JAX/jaxlib 0.7.1 | Pending M5/kernel validation |

The checked-in CI workflow runs the full test suite against the constrained
core-minimum (Python 3.10) and core-current (Python 3.13) profiles. The initial
core-current job passed; the minimum run is being repeated with the compatible
Hub 1.5.0 / Transformers 5.15.0 pair. A passing CPU job verifies only that
profile on CPU; it does not certify CUDA or TPU. The T0.1 CPU baseline was
separately run in Python 3.12.14.

The broad core package contract is PyTorch `>=2.9,<2.15`, Transformers
`>=5.15,<6`, and Hugging Face Hub `>=1.5,<2`. Exact constraint profiles are the
reproducible CI and TPU environments. The minimum profile locks Hub 1.5.0;
core-current locks Hub 1.33.0; the historical TPU/XLA constraints lock Hub
1.16.4. A version inside the broad range is not hardware certified merely
because dependency resolution succeeds.

## Installation profiles

These commands are documentation for the target environments. TPU extras are
Linux TPU VM profiles and are not intended for the local Windows workspace.

### Core minimum

```text
python -m pip install -e ".[dev]" -c constraints/core-minimum.txt
```

### Core current

```text
python -m pip install -e ".[dev]" -c constraints/core-current.txt
```

### Stable PyTorch/XLA TPU

```text
python -m pip install -e ".[tpu-xla]" -c constraints/tpu-xla-2.9.txt
```

The editable install is required when running the TPU launcher from a checkout:
it registers the `trainlm` distribution and places the `src/` package on the
worker import path. The validation notebook runs this same command when
`TRAINLM_INSTALL=1` or when the local distribution metadata is missing.

### Stable PyTorch/XLA TPU with Pallas

```text
python -m pip install -e ".[tpu-pallas]" -c constraints/tpu-pallas-2.9.txt
```

PyTorch/XLA 2.9 officially pairs with PyTorch 2.9. Its package metadata pins
`libtpu==0.0.21` for the TPU extra and `jax==jaxlib==0.7.1` for the Pallas
extra. TrainLM repeats those pins in constraints so a transitive release change
cannot silently alter a benchmark environment.

Do not install `jax[tpu]` independently into the Pallas profile. PyTorch/XLA
owns the active TPU runtime in this backend, while its Pallas extra supplies the
matched JAX/JAXLIB libraries used by the kernel bridge.

## Dependency ownership

| Group | Owns | Must not own |
|---|---|---|
| Core | PyTorch semantics, Transformers model contract | Accelerator initialization |
| `tpu-xla` | XLA runtime and libtpu plugin | Generic trainer/model logic |
| `tpu-pallas` | Matched JAX libraries for custom kernel integration | JAX model/trainer implementation |
| Profiling | Optional TensorBoard-facing artifacts | Required training behavior |
| Development | Tests, schema validation, coverage | Runtime imports |

CUDA wheels are selected from the official PyTorch index for the machine's
CUDA runtime. TrainLM does not encode a CUDA wheel index in package metadata.

## Compatibility CI policy

- Per-change CPU CI runs the minimum and current profiles.
- Transformers v5 compatibility is evaluated at both the accepted floor and
  newest supported minor before widening the core range.
- Accelerator imports are forbidden in core-only import tests.
- Scheduled TPU correctness uses the exact XLA constraint file.
- Pallas tests use the exact Pallas constraint file and record package versions
  in the run manifest.
- Release performance evidence never uses an unversioned nightly dependency.
- Nightly experiments use a dated, separate manifest and cannot replace stable
  certification evidence.

## Upgrade protocol

A dependency profile change requires:

1. a new matrix version or explicit reviewed edit;
2. matching constraint and package-extra updates;
3. CPU conformance for affected core versions;
4. M6/M7 correctness reruns for backend changes;
5. parity/HLO review when the TPU compiler or kernel stack changes;
6. updated support and benchmark manifests before release claims.

PyTorch/XLA remains the selected implementation backend until TorchTPU has a
public installable training contract and passes its future migration gates.

## Official references

- [Transformers v5 installation](https://huggingface.co/docs/transformers/v5.0.0/installation)
- [Transformers releases](https://github.com/huggingface/transformers/releases)
- [Hugging Face Hub downloads](https://huggingface.co/docs/huggingface_hub/main/package_reference/file_download)
- [Hugging Face Hub package metadata](https://pypi.org/pypi/huggingface-hub/json)
- [PyTorch/XLA installation](https://github.com/pytorch/xla)
- [PyTorch/XLA 2.9 package metadata](https://pypi.org/pypi/torch-xla/2.9.0/json)
- [PyTorch package metadata](https://pypi.org/project/torch/)
- [JAX TPU installation](https://docs.jax.dev/en/latest/installation.html)
