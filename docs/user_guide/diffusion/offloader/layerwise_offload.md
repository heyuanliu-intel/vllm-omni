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

Encoders are device resident by default. A pipeline may declare which of an
encoder's submodules are streamable block stacks, through
`OffloadPlan.encoder_block_attrs`; those stacks are paged the same way DiT
blocks are, and only the encoder's non-block state (norms, embeddings,
projections) is placed on the device. Non-block DiT modules such as embeddings
and norms remain device resident. VAE modules remain device resident too,
except when the pipeline declares the VAE in its
`OffloadPlan.on_demand_component_paths` *and* the VAE exposes an
`offload_to_cpu()` method: such a pipeline loads and releases the VAE itself
around encode/decode, so layer-wise offloading leaves it in host memory instead
of keeping a resident copy that is never used.

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

By default, layerwise offloading may manage every component family. Use
`--layerwise-offload-components` to narrow that set:

```bash
vllm serve MODEL --omni --enable-layerwise-offload \
  --layerwise-offload-components text_encoder,vae
```

The value is a non-empty comma-separated list drawn from `dit`,
`text_encoder`, and `vae`; unknown names fail configuration validation.
Every family is gated the same way in the backend:

- `dit` left out: the complete DiT stays device-resident and no DiT streaming
  hooks are installed.
- `text_encoder` left out: encoders are placed on the device and nothing else
  is done to them -- no block streaming and no host parking, *even if the
  pipeline declares an encoder block stack or declares the encoder on-demand*.
- `vae` left out: VAEs are placed on the device, *even if the pipeline declares
  the VAE on-demand and it exposes `offload_to_cpu()`*.

A family left out of the selection stays fully device-resident, and that is a
contract on both sides. A pipeline that manages its own encoder or VAE
residency must read the selection before it host-stages a component:
otherwise the backend places an unselected component on the device and the
pipeline pulls it back after every use, paying a host round trip for weights
the operator asked to keep resident. `MiniMaxH3Pipeline._stages_component_family`
is the reference implementation.

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

## Limitations

- Single device only.
- Setup consolidates and pins block parameters, increasing cold-start time.
- Performance depends on block compute time and host-to-device bandwidth;
  lightweight blocks may not hide transfers.
