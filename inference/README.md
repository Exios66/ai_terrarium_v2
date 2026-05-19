# Inference Module

Batch inference for digital-twin prompt CSVs using local or API-based models:
- [vLLM](https://github.com/vllm-project/vllm)'s offline `LLM` engine for local open-weight models
- Hugging Face Transformers pipeline for simpler local inference
- Anthropic Claude API for proprietary model comparison

All backends read prompt CSVs produced by `prompt_creation`, apply chat templates,
and write generation results to a CSV with built-in checkpoint-resume support.

## Prerequisites

|| Requirement | Notes |
||---|---|
|| Python >= 3.10 | Required by vLLM |
|| CUDA-capable GPU(s) | Ada / Hopper recommended for fp8 quantisation |
|| `vllm` | `pip install vllm` (see version note below) |
|| `transformers` | Required for both the vLLM tokenizer path and the Transformers backend |
|| `pandas`, `tqdm` | Data I/O and progress bars |

> **Tip:** Use a dedicated conda environment (e.g. `conda activate vllm`) to keep vLLM's heavy CUDA dependencies isolated from the lighter `prompt_creation` stack.

## File overview

```
inference/
├── __init__.py              # Package exports
├── predict_transformers.py  # Core: transformers_predict() + CLI entry point
├── predict_vllm.py          # Core: vllm_predict() + CLI entry point
├── predict_claude.py        # Core: claude_predict() + CLI entry point (Anthropic API)
├── utils.py                 # convert_prompt_to_messages, read_api_key
└── README.md                # This file

scripts/
├── run_vllm.sh              # One-command launcher (foreground or nohup)
└── run_vllm_monitor.sh  # Tail / poll / status for inference logs
```

## Quick start

### 1. CLI (recommended)

```bash
# From the repo root:
python -m inference.predict_vllm \
    --prompt_csv data/prompts/prompts.csv \
    --result_csv data/results.csv \
    --gpu 0 \
    --tensor_parallel_size 2 \
    --quantization fp8
```

All flags have sensible defaults; run `python -m inference.predict_vllm --help` for the full list.

### 2. Shell script

Edit the defaults at the top of `scripts/run_vllm.sh`, or pass them as environment variables:

```bash
PROMPT_PATH=data/prompts/prompts.csv RESULT_PATH=data/results.csv ./scripts/run_vllm.sh
```

Set `BACKGROUND=1` (default) to run under `nohup` with automatic log rotation under `logging/`.

### 3. Python API

```python
from types import SimpleNamespace
import pandas as pd
from inference.predict_vllm import vllm_predict

df = pd.read_csv("data/prompts/prompts.csv")

cfg = SimpleNamespace(
    model_full_name="meta-llama/Llama-3.3-70B-Instruct",
    max_output_tokens=30,
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.0,
    batch_size=16,
    save_freq=200,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    quantization="fp8",
    hf_access_token_file="hf_access_token.txt",
)

vllm_predict(df, "cuda:0", cfg, "data/results.csv")
```

## Transformers path

Use `predict_transformers.py` when you want the simplest possible
"prompt CSV -> local model -> result CSV" flow for quick debugging or smaller runs.
It is intentionally less featureful and less scalable than the vLLM path, but
it uses the same input/output CSV contract so you can swap backends later.

### CLI

```bash
python -m inference.predict_transformers \
    --prompt_csv data/prompts/prompts.csv \
    --result_csv data/results.csv \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --hf_access_token_file hf_access_token.txt \
    --gpu 0
```

For public models, you can leave `--hf_access_token_file` empty. The default
example above uses Llama 3 8B Instruct, which is gated on Hugging Face, so it
typically needs the same token file you would use with vLLM.

### Python API

```python
from types import SimpleNamespace
import pandas as pd
from inference.predict_transformers import transformers_predict

df = pd.read_csv("data/prompts/prompts.csv")

cfg = SimpleNamespace(
    model_full_name="meta-llama/Meta-Llama-3-8B-Instruct",
    max_output_tokens=30,
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.0,
    batch_size=4,
    save_freq=100,
    load_mode="",
    torch_dtype="auto",
    hf_access_token_file="hf_access_token.txt",
)

transformers_predict(df, "cuda:0", cfg, "data/results.csv")
```

### When to use which backend?

|| Backend | Best for | Tradeoff |
||---|---|---|
|| `predict_transformers` | Teaching, small demos, readable code paths | Slower, no built-in multi-GPU serving abstraction |
|| `predict_vllm` | Large prompt sets, repeated experiments, production-ish runs | Heavier environment and more systems complexity |

## Claude (Anthropic) backend

Use `predict_claude.py` when you want to run inference via the Anthropic Claude API instead of a local model. This is useful for comparing with proprietary models or when GPU resources are limited.

### Prerequisites for Claude backend

- Python >= 3.10
- `anthropic` package: `pip install anthropic`
- An Anthropic API key (free trial available; escalate to paid plan for higher rate limits)

### Claude CLI

```bash
python -m inference.predict_claude \
    --prompt_csv data/prompts/prompts.csv \
    --result_csv data/results.csv \
    --model claude-3-5-haiku-20241022 \
    --api_key_file anthropic_api_key.txt \
    --concurrency 8 \
    --requests_per_min 50
```

### API key resolution

The module looks for an API key in this order (first match wins):

1. `--api_key_file` flag (plain-text file, one line)
2. `ANTHROPIC_API_KEY` environment variable
3. `anthropic_api_key.txt` in the repo root

**Setup:**

```bash
# Option 1: Plain-text file
echo "sk_live_YOUR_KEY_HERE" > anthropic_api_key.txt

# Option 2: Environment variable
export ANTHROPIC_API_KEY=sk_live_YOUR_KEY_HERE
```

### Claude Python API

```python
from types import SimpleNamespace
import pandas as pd
from inference.predict_claude import claude_predict

df = pd.read_csv("data/prompts/prompts.csv")

cfg = SimpleNamespace(
    model_full_name="claude-3-5-haiku-20241022",
    api_key_file="anthropic_api_key.txt",  # or "" to use env var
    system_msg="You are a helpful assistant.",
    max_output_tokens=60,
    concurrency=8,
    requests_per_min=50,
    save_freq=100,
)

claude_predict(df, cfg, "data/results.csv")
```

### Claude configuration

| Flag | Config attribute | Default | Description |
|---|---|---|---|
| `--model` | `model_full_name` | `claude-3-5-haiku-20241022` | Anthropic model ID (see [docs](https://docs.anthropic.com/en/docs/about-claude/models/overview)) |
| `--api_key_file` | `api_key_file` | `anthropic_api_key.txt` | Path to plaintext API key file |
| `--system_msg` | `system_msg` | (persona default) | System prompt for all requests |
| `--max_output_tokens` | `max_output_tokens` | `60` | Max tokens per response |
| `--concurrency` | `concurrency` | `8` | Max simultaneous in-flight requests |
| `--requests_per_min` | `requests_per_min` | `50` | Soft rate limit (sleeps to stay under this) |
| `--save_freq` | `save_freq` | `100` | Flush results to CSV every N rows |

Claude also supports checkpoint-resume and rate limiting. Use `--concurrency` and `--requests_per_min` to control throughput and avoid quota limits. Pricing varies by model; Haiku (fast, cheapest) is suitable for high-volume batch work.

## Input / output format

### Input CSV

|| Column | Required | Description |
||---|---|---|
|| `caseid` | Yes | Unique row identifier (used for checkpoint-resume) |
|| `prompt` | Yes | Natural-language persona prompt (produced by `prompt_creation`) |
|| `answer` | No | Ground truth; carried through to results if present |

### Output CSV

|| Column | Description |
||---|---|
|| `caseid` | Echoed from input |
|| `answer` | Echoed from input (if present) |
|| `generated_text` | Model output (newlines replaced with spaces) |

## Checkpoint-resume

If the result CSV already exists when inference starts, rows whose `caseid` appears in the existing output are skipped automatically.  This makes it safe to kill and restart long-running jobs without losing progress.

### Validation and error handling

The checkpoint resume system includes built-in safety checks to prevent appending results to a corrupted checkpoint CSV. Before resuming, the code validates:

1. **CSV header presence** — The file must have column names.
2. **`caseid` column presence** — The output CSV must include a `caseid` column for resume to work.
3. **Duplicate case IDs** — Detects if any case ID appears multiple times (indicates corruption or merge error).

If validation fails, you will see an error like:

```
Invalid existing result CSV: Existing result CSV contains duplicate caseid values (101, 102, 103): results.csv
Delete or repair the file before resuming so new generations are not appended to a corrupt checkpoint.
```

#### Recovery workflow

If you hit a checkpoint validation error:

1. **Investigate the CSV** — Check whether the file is actually corrupt or if it was created by a merge/concatenation mistake:
   ```bash
   head -20 results.csv
   tail -20 results.csv
   ```

2. **Preserve completed work** — If the file is valid but has issues, back it up:
   ```bash
   cp results.csv results.csv.backup
   ```

3. **Remove the checkpoint** to start fresh inference:
   ```bash
   rm results.csv
   ```
   Then restart inference; new results will be written to a clean file.

4. **Or repair and retry** — If only a small section is corrupted, you can:
   - Edit the CSV manually to remove duplicate rows, or
   - Extract only valid rows to a new file:
     ```bash
     python -c "import pandas as pd; pd.read_csv('results.csv').drop_duplicates(subset=['caseid']).to_csv('results_fixed.csv', index=False)"
     ```
   Then rename and resume.

## Key CLI flags

|| Flag | Default | Description |
||---|---|---|
|| `--model` | `meta-llama/Llama-3.3-70B-Instruct` | HuggingFace model id or local path |
|| `--gpu` | `0` | First GPU id (combined with `--tensor_parallel_size`) |
|| `--tensor_parallel_size` | `2` | Number of GPUs for tensor parallelism |
|| `--quantization` | `fp8` | `fp8`, `bitsandbytes`, `awq`, `gptq`, or `none` |
|| `--gpu_memory_utilization` | `0.9` | Fraction of GPU memory vLLM may use |
|| `--max_model_len` | `8192` | Maximum context length |
|| `--max_output_tokens` | `30` | Max new tokens per sample |
|| `--temperature` | `0.0` | 0 = greedy; increase if output degenerates |
|| `--repetition_penalty` | `1.0` | > 1 penalises repeated tokens |
|| `--batch_size` | `16` | Sub-batch size for `llm.generate` |
|| `--save_freq` | `200` | Flush results every N rows |
|| `--hf_access_token_file` | `hf_access_token.txt` | HuggingFace token for gated models |

## Utility functions

### `read_completed_caseids(result_csv: str) → set[str]`

Validates and reads a checkpoint CSV to extract the set of already-completed case IDs.

**Purpose**: Used internally during checkpoint-resume to safely validate the existing results file before appending new results.

**Parameters**:
- `result_csv` (str) — Path to the inference results CSV

**Returns**: `set[str]` of completed case IDs

**Raises ValueError** if:
- The CSV file is empty (no header row)
- The CSV has no `caseid` column
- Duplicate case ID values are detected

**Example**:
```python
from inference.utils import read_completed_caseids

try:
    completed = read_completed_caseids("results.csv")
    print(f"Already completed: {len(completed)} samples")
except ValueError as e:
    print(f"Invalid checkpoint: {e}")
```

## Monitoring

```bash
# Tail the latest log:
./scripts/run_vllm_monitor.sh

# One-shot status:
./scripts/run_vllm_monitor.sh --status

# Poll until a background job finishes:
./scripts/run_vllm_monitor.sh --poll <PID>
```

## HuggingFace model access and tokens

Many high-quality open-weight models (Llama, Mistral, Gemma, etc.) are **gated** on HuggingFace — you must request access before downloading weights.

### Step-by-step

1. **Create a HuggingFace account** at <https://huggingface.co/join>.
2. **Find the model page** — e.g. <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>.
3. **Request access** — click the "Access" or "Agree and access" button and wait for approval (usually instant for Llama models).
4. **Generate an access token** at <https://huggingface.co/settings/tokens>.  Select the *Read* scope.
5. **Save the token** to a plain-text file in the repo root:

   ```bash
   echo "hf_YOUR_TOKEN_HERE" > hf_access_token.txt
   ```

   This file is listed in `.gitignore` and will **not** be committed.

6. **Verify access** (optional):

   ```bash
   conda activate vllm
   python -c "
   from huggingface_hub import HfApi
   api = HfApi(token=open('hf_access_token.txt').read().strip())
   info = api.model_info('meta-llama/Llama-3.3-70B-Instruct')
   print(f'Model: {info.modelId}, Gated: {info.gated}')
   "
   ```

### HF cache sharing

Model weights (~140 GB for Llama-3.3-70B) are stored in the HuggingFace cache directory (`HF_HOME`).  To avoid downloading twice, point both repos at the same cache:

```bash
export HF_HOME=/home/zli2545/ai_terrarium/hf_cache
```

The shell script `scripts/run_vllm.sh` defaults to the `ai_terrarium` cache.  Override via the `HF_HOME` env variable if your cache lives elsewhere.

## Logging

Log files are written to `logging/` with the naming convention:

```
{DATE}_{TASK_NAME}_{PROMPT_COMBO}_{PROMPT_STEM}_{TIME}.log
```

Each log starts with a structured header block containing all run parameters (model, GPU, quantisation, etc.) and ends with a footer showing elapsed runtime and result path.

```bash
# List all logs with summary table:
./scripts/run_vllm_monitor.sh --list

# Latest log status:
./scripts/run_vllm_monitor.sh --status
```

## Environment variables

|| Variable | Effect |
||---|---|
|| `HF_HOME` | HuggingFace cache directory (default: `<repo>/hf_cache`) |
|| `CUDA_VISIBLE_DEVICES` | Overrides GPU selection; the module sets this automatically when needed |
|| `VLLM_GPU_MEMORY_UTIL` | Override for `--gpu_memory_utilization` (shell script only) |
|| `VLLM_MAX_MODEL_LEN` | Override for `--max_model_len` (shell script only) |
|| `VLLM_TP_SIZE` | Override for `--tensor_parallel_size` (shell script only) |
|| `VLLM_QUANTIZATION` | Override for `--quantization` (shell script only) |

## Claude (Anthropic) inference

For cloud-based inference with Anthropic's Claude models (no GPU required):

### CLI

```bash
python -m inference.predict_claude \
    --prompt_csv data/prompts/prompts.csv \
    --result_csv data/results.csv \
    --model claude-3-5-haiku-20241022 \
    --api_key_file anthropic_api_key.txt
```

Key flags:
- `--model` — Anthropic model ID (default: `claude-3-5-haiku-20241022`)
- `--api_key_file` — Path to file containing your Anthropic API key (default: `anthropic_api_key.txt`)
- `--concurrency` — Max simultaneous API calls (default: 8)
- `--requests_per_min` — Soft rate-limit cap (default: 50)
- `--max_output_tokens` — Max new tokens per request (default: 60)
- `--save_freq` — Checkpoint frequency (default: 100 rows)

Run `python -m inference.predict_claude --help` for all options.

### Python API

```python
from types import SimpleNamespace
import pandas as pd
from inference.predict_claude import claude_predict

df = pd.read_csv("data/prompts/prompts.csv")

cfg = SimpleNamespace(
    model_full_name="claude-3-5-haiku-20241022",
    api_key_file="anthropic_api_key.txt",
    system_msg="You are a helpful assistant...",
    max_output_tokens=60,
    concurrency=8,
    requests_per_min=50,
    save_freq=100,
)

claude_predict(df, cfg, "data/results.csv")
```

### API Key setup

1. **Create an Anthropic account** at <https://console.anthropic.com>
2. **Generate an API key** in your account settings
3. **Save the key** to `anthropic_api_key.txt` in the repo root:
   ```bash
   echo "sk_test_YOUR_KEY_HERE" > anthropic_api_key.txt
   ```
   This file is listed in `.gitignore` and will **not** be committed.

Alternatively, set the `ANTHROPIC_API_KEY` environment variable and leave `--api_key_file` empty (or use `anthropic_api_key.txt` as default).

### Features

- **Checkpoint-resume**: If the result CSV exists, rows already processed are skipped automatically.
- **Rate limiting**: Respects `--requests_per_min` soft cap to stay within API quotas.
- **Thread-pool concurrency**: Configurable parallel requests for fast batch processing.
- **Automatic retry**: Transient errors (rate limits, network timeouts) are retried with exponential backoff.
- **Same input/output format**: Input and output CSVs are identical to vLLM/Transformers backends, enabling easy comparison.

### When to use Claude vs other backends

| Backend | Best for | Tradeoff |
|---|---|---|
| `predict_vllm` | Local, unfiltered inference; large volume; total cost control | Requires GPU, heavier setup |
| `predict_transformers` | Simple debugging, educational demos | Single-GPU only, slower |
| `predict_claude` | Managed cloud inference, safety constraints, rapid experiments | Pay-per-token pricing, rate limits |

### Cost and rate limits

- **Pricing**: Variable by model. Check <https://www.anthropic.com/pricing> for current rates.
- **Rate limits**: Default soft cap is 50 requests/minute. Adjust `--requests_per_min` to match your quota.
- **Concurrency**: Default 8 concurrent requests. Increase for higher throughput (respectfully).

Example cost estimate:
- 1000 prompts × 30 output tokens × $3/MTok (Haiku) ≈ $0.09
