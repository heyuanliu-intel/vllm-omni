# Regional Compilation

Regional compilation applies `torch.compile` to the repeated transformer blocks
declared by a diffusion model. It is the default compilation scope when
diffusion inference runs without `--enforce-eager`.

## Configuration

Dynamic compilation is enabled by default so the compiled regions can handle
mixed resolutions. For a fixed-shape workload, disable it explicitly:

```bash
vllm serve <model> --omni --no-diffusion-compile-dynamic
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_compile_granularity: regional
    diffusion_compile_dynamic: false
```

For an experimental whole-transformer compile scope, set
`--diffusion-compile-granularity full` or use
`diffusion_compile_granularity: full` in the stage configuration. Full scope may
still contain graph breaks; it does not force one graph. It is rejected when
HSDP, sequence parallelism, CPU offload, or layerwise offload is enabled. Use
regional scope with those features.

These settings control the generic model-runner compilation path. Pipelines
that provide their own `setup_compile()` implementation manage their compilation
policy independently. Compilation is lazy, so backend or graph errors can first
surface on the initial request.

## Keeping one compiled shape

Compiled regions are keyed by input shape, so a pipeline whose sequence length
follows the request — a longer prompt, a different reference image — hands the
compiler a shape it has not seen before and pays for a recompilation.

MiniMax-H3 lets a request pin that length with `extra_args["pad_seq_len"]`
(a positive multiple of 64, at least the rows the request actually uses):

```python
sampling_params = SamplingParams(extra_args={"pad_seq_len": 54080})
```

The packed sequence is then padded to that fixed length instead of to the next
64-row boundary, so requests of different prompt lengths share one compiled
shape. The server logs the effective length as
`MiniMax H3 packed sequence: ... pad_seq_len=... used=... seq_len=...`
whenever a request pins it. Pick a bucket that covers the longest prompt the
deployment accepts; the padding rows are masked out, so the extra cost is the
attention and feed-forward work on those rows.

Use `--enforce-eager` to disable the model runner's generic compile setup.
Pipelines that compile internally define their own eager-mode behavior.
