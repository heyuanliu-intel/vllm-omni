# Model-Level Offloading

Model-level, or sequential, offloading keeps only the pipeline component group
currently executing on the accelerator. It is the simplest offload strategy
and is selected with `--enable-cpu-offload`.

## How it works

Pre-forward hooks enforce mutual exclusion between DiT and encoder modules:

- before an encoder runs, the DiT moves to CPU;
- before a DiT runs, encoders and other DiTs move to CPU; and
- VAE modules remain on the accelerator.

Pinned host memory reduces transfer overhead. Transfers occur at phase
boundaries, so cold-start and encoder-to-denoiser transitions become slower.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_cpu_offload=True,
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni --enable-cpu-offload
```

## Model integration

Pipelines should implement `SupportsComponentDiscovery`:

```python
from typing import ClassVar

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery


class MyPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vision_model"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []
```

All entries may be dotted paths. DiT and encoder lists are both required for
mutual exclusion. VAE modules are pinned but not swapped; resident modules are
small modules that must stay on the accelerator for layerwise paths.

## Split-model components

Some models, such as Cosmos3, split one transformer into mutually exclusive
components that run in different phases. The pipeline exposes
`enable_omni_model_cpu_offload`, and the backend delegates to the model-local
contexts:

```python
class Cosmos3VFMTransformer(nn.Module):
    def forward(self, ...):
        with self._offload_context("reasoner"):
            ...
        with self._offload_context("generator"):
            ...
```

This preserves the same invariant—exactly one component is device resident—
while reusing sequential `.to()` movers.

## Host-staged VAEs (`vae_cpu_offload`)

By default model-level offloading keeps VAEs GPU-resident, because the mutual
exclusion above only swaps the DiT and the encoders. For pipelines that stage
and release their VAEs around each use, holding those weights on the
accelerator for the whole request is pure overhead. `--vae-cpu-offload` lets
such a pipeline park its VAEs in host memory instead.

**Python API:**

```python
from vllm_omni import Omni

m = Omni(model="MiniMaxAI/MiniMax-H3", enable_cpu_offload=True, vae_cpu_offload=True)
```

**CLI:**

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --enable-cpu-offload --vae-cpu-offload
```

The flag only takes effect when model-level offloading is on and the pipeline
declares its VAEs as on-demand components. If any condition fails the VAE stays
resident, which is the previous behavior. Each VAE use then pays a load and an
offload, so it pays off when the VAE is large relative to how often it runs, and
costs throughput when the VAE runs frequently.

Currently supported by: MiniMax-H3 (`video_vae`, `audio_vae`).

## Limitations

- Single device only.
- Higher cold-start latency.
- Transfers between encoder and denoising phases add latency.

See the [model-level design](../../../design/feature/offloader/cpu_offload.md)
for lifecycle and extension invariants.
