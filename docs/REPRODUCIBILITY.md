# Reproducibility

The repository contains one final runtime:

```text
runtime/bad_t012_fire_t050/run.py
```

The root `run.py` must be byte-for-byte identical to this file.

## Artifact verification

From the repository root:

```bash
python scripts/verify_repo.py
```

The script checks SHA256 hashes for:

- `adapters/bad/adapter_model.safetensors`
- `adapters/bad/adapter_config.json`
- `adapters/fire/adapter_model.safetensors`
- `adapters/fire/adapter_config.json`
- `run.py`
- `runtime/bad_t012_fire_t050/run.py`
- `metadata.json`
- `wheels/peft-0.20.0-py3-none-any.whl`

It also checks that the root `run.py` and `runtime/bad_t012_fire_t050/run.py` are byte-for-byte identical.

Expected result:

```text
RESULT: ALL OK
```

## Build the submission

```bash
python scripts/build_submission.py --variant bad_t012_fire_t050
```

Submission contents:

```text
run.py
metadata.json

adapters/
  bad/
    adapter_config.json
    adapter_model.safetensors

  fire/
    adapter_config.json
    adapter_model.safetensors

wheels/
  peft-0.20.0-py3-none-any.whl
```

## Competition runtime

Expected shared models:

```text
/shared_models/Qwen/Qwen3.5-4B
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

Entry point:

```bash
python -u run.py
```

Final thresholds:

```text
Dietary supplements (BAD): 0.12
Flammable products (FIRE): 0.50
```

## Final pre-submission check

```bash
python scripts/verify_repo.py
python scripts/build_submission.py --variant bad_t012_fire_t050
```

Then confirm that the final ZIP contains `run.py`, `metadata.json`, both LoRA adapters, and the local PEFT wheel.
