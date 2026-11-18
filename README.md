# E-CUP 2026 — Task 2: Quality Control

## 2nd place on the Private Leaderboard

This repository contains the final solution for classifying products into two categories:

- Dietary supplements (`БАД`, referred to as `BAD` in the code).
- Flammable products (`Легковоспламеняющиеся`, referred to as `FIRE`).

Final Private Leaderboard result:

```text
2nd place
Private score = 0.8717948718
```

![Private LB](assets/private_lb.png)

## Solution overview

The solution uses the multimodal `Qwen/Qwen3.5-4B` model with two independent LoRA adapters:

- One adapter for dietary supplements.
- One adapter for flammable products.

The model combines the product name, description, up to 5 images, and OCR text extracted from the images.

At inference, images are combined into a contact sheet. `PaddlePaddle/PaddleOCR-VL-1.5` also extracts text from them, which is added to the Qwen prompt as an additional information source.

Pipeline:

```text
name + description
        |
        +--------------------------+
        |                          |
        v                          v
   Qwen prompt                product images
                                   |
                         +---------+---------+
                         |                   |
                         v                   v
                  Qwen contact sheet   OCR contact sheet
                                             |
                                             v
                                  PaddleOCR-VL-1.5
                                             |
                                             v
                                          OCR text
                                             |
                         +-------------------+
                         |
                         v
                  Qwen3.5-4B + LoRA
                         |
                         v
                    logits "0"/"1"
                         |
                         v
                    P(label = 1)
                         |
                         v
                category threshold
                         |
                         v
                  final prediction
```

## Hypotheses and research

The final solution emerged from multiple training runs. The development loop was to fix a working baseline, change one component, evaluate each category separately, analyze errors, and then retain or revert the change. The goal was to understand which component improved performance while avoiding overfitting to the public leaderboard.

### Different categories need different models and decision boundaries

Dietary supplements and flammable products proved to be substantially different tasks.

For supplements, difficult cases mostly involve text semantics: negated or unrelated mentions, sports nutrition, phrases such as “not a dietary supplement,” similar products, and items without explicit labeling.

For flammable products, visual cues, packaging labels, ingredients, warnings, and hazardous components included in a kit matter more.

Two independent LoRA adapters on the same `Qwen/Qwen3.5-4B` backbone therefore replaced a shared head. Training, sampling, OCR, and thresholds could be adjusted independently for each category.

### Controlled adapter ablation

At one iteration, new BAD and FIRE candidates were available simultaneously. Replacing both adapters degraded performance, but that did not establish that the entire new training recipe was unsuccessful.

Controlled combinations changed only one adapter at a time. This showed that the new FIRE adapter improved the system, while the new BAD adapter caused the regression.

The final solution therefore retains the more stable BAD adapter and uses the improved OCR-aware FIRE adapter. This experiment localized the source of the performance change instead of comparing two entirely different submissions.

### OCR as a separate modality

We tested whether product photos contain text missing from `name` and `description` using an ablation with no additional training on a fixed holdout of 600 products:

- `BASE`: original Qwen + LoRA without OCR text in the prompt.
- `OCR`: the same weights, images, and prompt, with additional OCR text.
- Model weights remained unchanged between variants.

This isolated the effect of OCR from that of new training.

OCR then became a separate data pipeline. The entire dataset was processed, and product-level text was used to train the FIRE adapter. At inference, OCR is computed directly from the images.

OCR is treated as a potentially noisy additional signal. The prompt explicitly states that recognition may contain errors; decisions also use the name, description, and original images.

### Full text and full OCR for FIRE

For flammable products, a critical cue may appear late in the description or in only one photo. The final FIRE training pipeline therefore removed artificial text limits:

```text
description truncation = none
OCR truncation         = none
```

The FIRE adapter was trained on full descriptions and full product-level high-quality OCR. Balanced sampling addressed class imbalance: 5,502 unique products produced 10,608 effective training rows.

OCR was part of the training distribution, allowing the model to learn associations between recognized text and target labels, rather than merely receiving more context at runtime.

### Thresholds as separate model parameters

Post-training threshold tuning was a separate research component.

On an early validation set, the same weights produced:

```text
Dietary supplements (BAD):
F1 @ 0.50       = 0.938856
best threshold  = 0.85
best F1         = 0.942346
```

Later, full error mining on a broader diagnostic pool showed a different pattern. BAD had a broad plateau at low thresholds, with the best diagnostic F1 among the leading sweep results near `0.09`:

| BAD threshold | F1 | Precision | Recall |
|---:|---:|---:|---:|
| 0.05 | 0.977267 | 0.977179 | 0.977354 |
| 0.08 | 0.977277 | 0.980637 | 0.973940 |
| 0.09 | 0.977521 | 0.982042 | 0.973041 |
| 0.10 | 0.977421 | 0.982389 | 0.972502 |
| 0.12 | 0.976647 | 0.983771 | 0.969626 |

The optimal threshold depends strongly on the validation slice and is not a property of the weights alone. We therefore treated it as a separate post-training parameter instead of hardcoding the local `argmax` from one set.

Although the diagnostic maximum was near 0.09, the 0.08–0.12 range formed a relatively flat plateau. We selected 0.12 as a more conservative operating point: slightly higher precision and fewer false positives with almost unchanged F1, reducing sensitivity to local evaluation noise.

Final runtime thresholds:

```text
Dietary supplements (BAD) = 0.12
Flammable products (FIRE) = 0.50
```

![BAD threshold F1](assets/research/bad_threshold_f1.png)

![BAD threshold precision recall](assets/research/bad_threshold_precision_recall.png)

### Error mining beyond aggregate scores

After obtaining a strong BAD adapter, we analyzed false positives and false negatives by semantic pattern:

- The word `БАД` (dietary supplement) in a negated or unrelated context.
- Explicit `не является БАД` (“not a dietary supplement”).
- Sports nutrition and fat burners.
- Products without explicit supplement labeling.
- Conflicting product-card text and packaging labels.
- Cases where the useful cue appears only in OCR.

For example, `не является лекарством, БАД` and `не является лекарственным средством и БАД` look similar but can express opposite meanings depending on context. Roughly, the first may distinguish a supplement from a medicine, while the second may negate both categories. Simple keyword rules are unreliable here.

Explicit category rules were added to the prompt, while OCR remained a supporting information source.

### Cleaning conflicting annotations

We tested whether some BAD errors came from exact duplicate groups with conflicting labels by building a cleaned training branch:

```text
BAD rows:     7469 -> 7256
label 0:      1905 -> 1812
label 1:      5564 -> 5444
removed rows: 213
```

All 213 removed rows were recorded in an audit list. The pipeline checked that none returned after OCR joins or sampling.

The same BAD recipe was then retrained with full product-level high-quality OCR. This was a useful data-quality experiment but did not provide sufficient evidence to replace the final BAD adapter.

This negative result matters: cleaner data and richer input do not guarantee better generalization.

### Leakage-resistant validation

A separate research branch examined evaluation reliability using 5-fold `StratifiedGroupKFold`. Identical normalized `name + description` values were assigned to the same fold, preventing near-identical product cards from appearing in both training and validation.

The protocol called for independent Qwen LoRA training per fold, OOF predictions, and cross-fitted thresholds based only on other folds.

This protocol is intended to test whether improvements transfer to new product groups or merely reflect duplicate and similar cards; it is not part of final inference.

### Alternative backbone

We also prepared an independent benchmark branch using `google/gemma-4-E4B-it`. To keep the comparison controlled:

- A balanced benchmark of 400 products was fixed.
- Each category contributed 200 products: 100 with label 0 and 100 with label 1.
- All benchmark IDs were excluded from Gemma training.
- The objective remained binary classification using next-token `0/1` logits.
- Separate LoRA adapters were trained for the two categories.

This branch was not included in the final submission. It tested how strongly the observed errors depended on the backbone.

### Components retained in the final system

| Research hypothesis | Outcome |
|---|---|
| Separate category adapters | Retained |
| Controlled adapter swaps | Preserved the strong BAD adapter and replaced only FIRE |
| OCR as an additional modality | Retained |
| OCR-aware FIRE training | Retained |
| Full description + full OCR for FIRE | Retained |
| Category-specific thresholds | Retained |
| Error mining and explicit category rules | Retained |
| Remove 213 conflicting BAD rows + retrain with HQ OCR | Tested; did not replace the final BAD adapter |
| Group-aware OOF validation | Separate validation branch |
| Gemma 4 E4B | Separate benchmark branch; excluded from the final submission |

The strongest result came from a controlled combination of components: a stable BAD adapter, an independently improved FIRE adapter, multimodal input, supporting OCR, and separately tuned decision thresholds.

## Base model

```text
Qwen/Qwen3.5-4B
```

Competition runtime path:

```text
/shared_models/Qwen/Qwen3.5-4B
```

Each category loads its own PEFT/LoRA adapter:

```text
adapters/
|- bad/
|  |- adapter_config.json
|  \- adapter_model.safetensors
|
\- fire/
   |- adapter_config.json
   \- adapter_model.safetensors
```

## Training the BAD adapter

Training notebook:

```text
notebooks/train_bad_adapter.ipynb
```

This notebook originally trained adapters for both categories. The FIRE adapter was later improved separately, while the BAD adapter remained unchanged.

The adapter was trained on `Qwen/Qwen3.5-4B` using QLoRA. Parameters of the completed run:

```text
LoRA r         = 16
LoRA alpha     = 32
LoRA dropout   = 0.05
learning rate  = 1e-4
epochs         = 1
contact sheet  = 576x576
```

Training is formulated as binary classification: the model learns to select target token `"0"` or `"1"` instead of generating a long response.

Final weights:

```text
adapters/bad/adapter_model.safetensors
SHA256:
5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d
```

Config:

```text
adapters/bad/adapter_config.json
SHA256:
291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391
```

## Training the FIRE adapter

Training notebook:

```text
notebooks/train_fire_adapter.ipynb
```

This category uses a separate OCR-aware pipeline with `Qwen/Qwen3.5-4B`, QLoRA, and FSDP on two NVIDIA T4 GPUs.

Main parameters:

```text
base model           = Qwen/Qwen3.5-4B
training             = FSDP + QLoRA
GPU                  = 2 x NVIDIA T4
LoRA r               = 16
LoRA alpha           = 32
LoRA dropout         = 0.05
learning rate        = 1e-4
epochs               = 1
visual side          = 448
contact sheet        = 576x576
unique training rows = 5502
balanced rows        = 10608
text truncation      = none
OCR truncation       = none
```

Training input includes the name, full description, images, and precomputed OCR text. The pipeline supports checkpointing and resumption. After FSDP training, the adapter is exported in standard PEFT format.

Final weights:

```text
adapters/fire/adapter_model.safetensors
SHA256:
6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01
```

Config:

```text
adapters/fire/adapter_config.json
SHA256:
278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b
```

## OCR for the full dataset

OCR was computed for the entire dataset to train the OCR-aware model. Reproducible notebook:

```text
notebooks/ocr_all_data.ipynb
```

The final cache was produced by sequentially processing images with two OCR backends.

### 1. PaddleOCR-VL-1.5

The first portion of images was processed using:

```text
PaddlePaddle/PaddleOCR-VL-1.5
```

OCR ran in parallel on two NVIDIA T4 GPUs. Per-image results were saved to intermediate CSV files so processing could resume after a session stopped.

### 2. PP-OCRv5 Mobile

The remaining images used a faster PaddleOCR backend:

```text
detector:
PP-OCRv5_mobile_det

recognizer:
eslav_PP-OCRv5_mobile_rec
```

`eslav_PP-OCRv5_mobile_rec` recognized Russian and English text. The second pipeline loaded existing PaddleOCR-VL results, skipped processed `(product_id, image_idx)` pairs, and processed only missing images.

The full dataset therefore combines:

```text
PaddlePaddle/PaddleOCR-VL-1.5
+
PP-OCRv5_mobile_det + eslav_PP-OCRv5_mobile_rec
```

Merging produced two final files.

### Image-level OCR

```text
data/ocr/ocr_all_images_high_quality.csv
```

Contains OCR for individual product photos, for workflows needing per-image text.

### Product-level OCR

```text
data/ocr/ocr_all_products_high_quality.csv
```

Contains text from multiple photos of each product combined into one `ocr_text` field. This product-level OCR was used as a precomputed text feature in OCR-aware training.

Both CSV files use Git LFS. Distinguish the two OCR stages:

- Training data preparation uses the combined cache from both backends.
- Final competition inference computes OCR directly with `PaddlePaddle/PaddleOCR-VL-1.5`.

## OCR at inference

Final runtime model path:

```text
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

Up to 5 images are used per product. Main parameters:

```text
MAX_IMAGES            = 5
OCR contact sheet     = 1024x1024
Qwen contact sheet    = 576x576
MAX_DESCRIPTION_CHARS = 2200
MAX_OCR_CHARS         = 2200
OCR_MAX_NEW_TOKENS    = 160
OCR_BATCH_SIZE        = 32
QWEN_BATCH_SIZE       = 24
```

OCR is particularly useful for packaging text: ingredients, warnings, labels, product properties, substance names, and information absent from the product description.

The runtime is fault tolerant: if OCR fails, classification continues with the available text and images.

## Class probability

The decision does not use free-form autoregressive generation. Logits for tokens `"0"` and `"1"` are taken from Qwen's final position, then softmax gives `P(label = 1)`.

Final decision rule:

```python
threshold = 0.12 if category == BAD else 0.5
pred = int(p1 >= threshold)
```

## Repository structure

```text
.
|- README.md
|- run.py
|- metadata.json
|- manifest.json
|
|- adapters/
|  |- bad/
|  |  |- adapter_config.json
|  |  \- adapter_model.safetensors
|  |
|  \- fire/
|     |- adapter_config.json
|     \- adapter_model.safetensors
|
|- data/
|  \- ocr/
|     |- ocr_all_images_high_quality.csv
|     \- ocr_all_products_high_quality.csv
|
|- notebooks/
|  |- ocr_all_data.ipynb
|  |- train_bad_adapter.ipynb
|  \- train_fire_adapter.ipynb
|
|- scripts/
|  |- verify_repo.py
|  \- build_submission.py
|
|- docs/
|  |- ARTIFACTS.md
|  \- REPRODUCIBILITY.md
|
|- assets/
|  |- private_lb.png
|  \- research/
|     |- bad_threshold_f1.png
|     \- bad_threshold_precision_recall.png
|
\- wheels/
   \- peft-0.20.0-py3-none-any.whl
```

## Artifact verification

The final adapters and runtime were verified using SHA256.

| Artifact | SHA256 |
|---|---|
| BAD adapter weights | `5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d` |
| BAD adapter config | `291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391` |
| FIRE adapter weights | `6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01` |
| FIRE adapter config | `278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b` |
| run.py (entry point) | `a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5` |
| metadata.json | `2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812` |
| PEFT wheel | `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233` |

Verify the local repository:

```bash
python scripts/verify_repo.py
```

Expected result:

```text
RESULT: ALL OK
```

Adapter weights and OCR data are stored with Git LFS. If LFS objects cannot be downloaded, the same files are mirrored on Hugging Face:

- [BAD adapter](https://huggingface.co/large-language-leonid/bad_adapter)
- [FIRE adapter](https://huggingface.co/large-language-leonid/fire_adapter)
- [OCR dataset](https://huggingface.co/datasets/large-language-leonid/e_cup_2026_ocr)

## Build the submission

```bash
python scripts/build_submission.py
```

The archive is created at the repository root:

```text
submission_final.zip
```

Competition entry point:

```text
python -u run.py
```

Competition image:

```text
odsai/ecup26-quality-baseline:1.0
```

## Result

```text
Private Leaderboard = 2nd place
Private score = 0.8717948718
```
