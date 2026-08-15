# Layerwise Offloading

Layerwise, or blockwise, offloading keeps one transformer block on the
accelerator and prefetches the next block while the current block computes.
It is best suited to compute-heavy video DiTs whose block execution can hide
host-to-device transfers.

## Execution flow

Each block has a pre-forward and post-forward hook. Parameters are consolidated
in pinned host tensors and rematerialized for execution on a dedicated copy
stream.

| Block | Pre-forward hook | Forward | Post-forward hook |
| --- | --- | --- | --- |
| block 0 | Prefetch block 1 | Compute block 0 | Free block 0 |
| block 1 | Prefetch block 2 | Compute block 1 | Free block 1 |
| ... | ... | ... | ... |
| last block | Prefetch block 0 | Compute last block | Free last block |

Encoders, VAE modules, and non-block DiT modules such as embeddings and norms
remain device resident.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_layerwise_offload=True,
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni --enable-layerwise-offload
```

## Model integration

Transformer classes declare containers of executable blocks:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]


class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = [
        "transformer_blocks",
        "single_transformer_blocks",
    ]
```

See the [layerwise design](../../../design/feature/offloader/layerwise_offload.md) for
the discovery and hook invariants. `OffloadPlan` is a separate declarative
topology path for distributed layerwise offload.

## Selecting which component families are managed

`--layerwise-offload-components` narrows which component families plain
layer-wise offloading may manage, as a comma list drawn from `dit`,
`text_encoder`, and `vae`. A family left out stays fully device-resident for
the whole request lifecycle. Omitting the option selects every family (the
default behavior above).

The load-bearing use is excluding `dit`: when the DiT fits whole on the device
but the encoders and VAEs do not, streaming the DiT would multiply the
per-step denoise time, so run with only the other families managed:

```bash
vllm serve <model> --omni --enable-layerwise-offload \
  --layerwise-offload-components text_encoder,vae
```

Constraints:

- Unknown or empty selections fail config validation.
- Only plain layer-wise offloading consumes the selection. Distributed
  layer-wise offloading ignores it (a warning is logged), and model-level CPU
  offload has its own policy.
- Selecting `text_encoder` or `vae` manages them only on pipelines that
  declare the capability (`OffloadPlan.encoder_block_attrs` /
  `on_demand_component_paths`); components without it stay on GPU either way.

## Limitations

- Single device only.
- Setup consolidates and pins block parameters, increasing cold-start time.
- Performance depends on block compute time and host-to-device bandwidth;
  lightweight blocks may not hide transfers.
