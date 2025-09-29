# Data

## Installation

```bash
conda create -n expressive -y python=3.12
conda activate expressive

pip install uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install jupyter transformers datasets torchcodec
```

## StressTest

`stresstest.jsonl` is the jsonl file for the StressTest dataset.

## ParaSpeechCaps Holdout

`paraspeechcaps_holdout.jsonl` is the jsonl file for the ParaSpeechCaps Holdout dataset.