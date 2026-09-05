# Dependency and Compatibility Policy

**Matrix:** [`compatibility/dependency_matrix_v1.json`](../compatibility/dependency_matrix_v1.json)

TrainLM separates portable model/trainer dependencies from accelerator-specific
runtime and kernel packages. Core imports must work without CUDA, PyTorch/XLA,
JAX, libtpu, or profiling packages installed.

## Support profiles

| Profile | Python | PyTorch | Transformers | Accelerator packages | Status |
|---|---|---|---|---|---|
| Core minimum | 3.10 | 2.9.0 | 5.0.0 | None | Required CPU CI |
| Core current | 3.13 | 2.13.0 | 5.15.0 | None | Required CPU CI |
| CUDA portability | 3.13 | 2.13.0 | 5.15.0 | Official matching CUDA wheel | Smoke when available |
| TPU XLA stable | 3.10–3.13 | 2.9.0 | 5.15.0 | torch-xla 2.9.0, libtpu 0.0.21 | Pending M5 TPU validation |
| TPU XLA + Pallas | 3.10–3.13 | 2.9.0 | 5.15.0 | XLA/libtpu above, JAX/jaxlib 0.7.1 | Pending M5/kernel validation |

The broad core package contract is PyTorch `>=2.9,<2.14`, Transformers
`>=5.0,<6`, and Hugging Face Hub `>=1.0,<2`. Exact constraint profiles are the
reproducible CI and TPU environments. The minimum profile locks Hub 1.3.5; the
current and TPU profiles lock Hub 1.16.4. A version inside the broad range is
not hardware Certified merely because dependency resolution succeeds.

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
