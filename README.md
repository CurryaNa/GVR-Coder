# GVR-Coder

Official supplementary code and sample data for **GVR-Coder: A Visual-Feedback
Framework for Structured SVG Generation in Complex Document and Meeting
Scenarios**.

GVR-Coder targets text-to-SVG generation for information-dense documents and
meeting scenarios. The released material covers the main components of the
framework:

- curriculum-driven rejection-sampling fine-tuning;
- reinforcement learning with visual and structural rewards;
- SVG verification and targeted repair; and
- a small, anonymized sample of the DocMeetSVG data.

> Paths, model identifiers, prompt keys, and environment-specific settings in
> the code are placeholders and must be configured before use. The complete
> dataset and model weights are coming soon.

## Repository layout

```text
.
├── data/
│   └── DocMeetSVG_sample.jsonl
└── gvr-coder_code/
    ├── gvr_agent/
    │   ├── verify.py
    │   ├── repair.py
    │   ├── extract_ap.py
    │   ├── extract_rpcode.py
    │   └── scripts/
    │       ├── common.sh
    │       ├── run_all.sh
    │       └── run_round*.sh
    └── train_scripts/
        ├── cul_rsft.sh
        ├── rlvr.sh
        └── plugin.py
```

For data-collection details, prompt templates, human-evaluation protocol, and
additional experimental results, please refer to the paper.

## Sample data

`data/DocMeetSVG_sample.jsonl` contains seven representative, anonymized
examples. Each line is a JSON object with the following main fields:

| Field | Description |
| --- | --- |
| `data_id` | Unique sample identifier |
| `query_id` | Identifier of the source query |
| `svg_type_2` | Diagram category (one legacy sample uses `svg_type2`) |
| `user_prompt` | Natural-language SVG generation request |
| `reference` | Reference SVG code |
| `messages` | System, user, and assistant messages used for supervised training |
| `difficulty` | Difficulty label: `0.0` (simple), `0.5` (medium), or `1.0` (hard) |

Some examples contain additional provenance or prompt fields. Consumers should
therefore tolerate optional fields and the legacy `svg_type2` spelling.

Load the sample data with:

```python
import json

with open("data/DocMeetSVG_sample.jsonl", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(len(samples))
print(samples[0]["user_prompt"])
```

## Environment

The scripts were developed for Python 3.10+ in a CUDA-enabled distributed
training environment. Core Python dependencies include:

```bash
pip install "pydantic>=2" pandas loguru beautifulsoup4 openai playwright
```

Training additionally requires a compatible installation of
[ms-swift](https://github.com/modelscope/ms-swift), Megatron-LM, PyTorch, and
the CUDA libraries required by the selected attention and inference backends.
Install the browser used by the SVG renderer when applicable:

```bash
playwright install chromium
```

The exact PyTorch, CUDA, ms-swift, Megatron-LM, and vLLM versions should be
matched to the target hardware and cluster environment.

## Training

### Implementation details

We evaluate GVR-Coder on representative model architectures, including
Qwen3-14B and Qwen3-32B, using full-parameter fine-tuning. Supervised training
is conducted on 32 NVIDIA A100 GPUs with a global batch size of 32. During the
reinforcement-learning stage, we use GRPO to train the SFT model through
vLLM's colocate integration mode with a global batch size of 256. All reward
coefficients in the RL stage are set to 1. The complete training pipeline is
implemented with the [ms-swift](https://github.com/modelscope/ms-swift)
framework.

### Curriculum fine-tuning

`gvr-coder_code/train_scripts/cul_rsft.sh` performs three sequential SFT
stages:

1. simple samples;
2. 85% medium and 15% simple samples; and
3. 85% hard and 15% medium samples.

Before launching, update `MODEL_PATH`, `BASE_SAVE_DIR`, the three dataset
paths, and distributed settings in the script. The script expects the
`megatron_sft_tool` command provided by the original training environment.

```bash
cd gvr-coder_code/train_scripts
bash cul_rsft.sh
```

### Visual-reward reinforcement learning

`gvr-coder_code/train_scripts/rlvr.sh` is the GRPO launch template. It combines:

- `svg_scorer`: a vision-language-model score for rendered SVG quality; and
- `svg_complexity_scorer`: a structural reward based on SVG primitive counts
  relative to the reference.

Configure all `/path/to/...` entries, the distributed environment variables,
the model/data locations, and the Conda environment before running:

```bash
cd gvr-coder_code/train_scripts
bash rlvr.sh
```

The reward implementations are registered in `plugin.py` through ms-swift's
`ORM` interface. The public `HtmlRenderer` is intentionally a stub: replace
`HtmlRenderer.start` and `HtmlRenderer.render_to_file` with a real
Playwright/Chromium renderer before training. Also configure the VLM model,
`OPENAI_API_KEY` (or a compatible endpoint/client), output directory, and
worker counts for the local environment.

## Verification and repair

The agent scripts implement the following loop:

```text
SVG JSON → render → verify → extract feedback → repair
         → extract repaired SVG → render → verify again
```

The released `verify.py` and `repair.py` depend on project-specific
`infer.BatchInferOpenAI`, `utils.svg_handler.check_and_format_svg`, and a
`render` command. These infrastructure modules are not included in the
anonymized package; provide compatible implementations or adapt the imports
to the local inference stack.

### 1. Verify generated SVGs

`verify.py` expects a JSON array whose records contain `data_id`,
`md_content`, and `svg_code`. Rendered images must be available at
`<output-dir>/images/<data_id>.jpg`.

```bash
python gvr-coder_code/gvr_agent/verify.py \
  --output-dir outputs/verify \
  --data-path inputs/generated_svgs.json \
  --predict-path outputs/verify/predictions.jsonl \
  --config-path configs/infer.yaml \
  --prompt-key YOUR_VERIFY_PROMPT \
  --enable-infer \
  --excel-path outputs/verify/statistics.xlsx
```

For each prediction, the verifier returns issue attributes, repair advice, and
`needs_repair`. Its score is `max(0, 6 - number_of_attributes)`. The script
also reports best-of-N normalized score and pass rate.

### 2. Extract verifier feedback

```bash
python gvr-coder_code/gvr_agent/extract_ap.py \
  --input outputs/verify/predictions.jsonl \
  --output outputs/verify/critic_feedback.json
```

### 3. Repair failed examples

Only records with `needs_repair=true` are sent to the repair model:

```bash
python gvr-coder_code/gvr_agent/repair.py \
  --input-critic-json outputs/verify/critic_feedback.json \
  --output-dir outputs/repair_round1 \
  --predict-path outputs/repair_round1/predictions.jsonl \
  --config-path configs/infer.yaml \
  --prompt-key YOUR_REPAIR_PROMPT \
  --enable-infer \
  --render
```

### 4. Prepare another verification round

```bash
python gvr-coder_code/gvr_agent/extract_rpcode.py \
  --input outputs/repair_round1/predictions.jsonl \
  --output outputs/repair_round1/repaired_svgs.json \
  --output-dir outputs/repair_round1
```

The resulting JSON can be rendered and passed back to `verify.py`. Repeat the
loop until the desired acceptance criterion or iteration limit is reached.
The shell wrappers under `gvr_agent/scripts/` provide an example of automating
three repair rounds; set their `BASE_DIR` placeholders before use.

```bash
cd gvr-coder_code
bash gvr_agent/scripts/run_all.sh
```

## Notes

- The sample JSONL uses `user_prompt` and `reference`, whereas the verifier and
  repair utilities use the normalized names `md_content` and `svg_code`.
  Convert these fields when moving from training data to the agent pipeline.
- The shell scripts are templates for the original distributed environment;
  review batch sizes, parallelism, sequence lengths, and worker counts before
  running on different hardware.
- Generated SVG or model output must be treated as untrusted input. Use an
  isolated browser context and appropriate resource limits when rendering it.

## Citation

Citation metadata will be added after publication. For now, please refer to
the work by its title:

> *GVR-Coder: A Visual-Feedback Framework for Structured SVG Generation in
> Complex Document and Meeting Scenarios.*

## License

No license is included in this anonymized package. The code and data should
not be assumed to grant reuse or redistribution rights until the authors
provide explicit licensing terms.
