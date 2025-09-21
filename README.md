# Kannada → English Translator


A ready-to-run project for translating Kannada text to English using Transformer models (Hugging Face). It includes:
- FastAPI REST server with `/translate` endpoint
- Optional Streamlit UI for quick demos
- CLI for batch translation
- Dockerfile, tests, and evaluation utilities


## Quick Start
1. Create a venv and install requirements:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Configure model via env vars (default: Helsinki-NLP/opus-mt-kn-en):
```bash
export HF_BASE_MODEL="Helsinki-NLP/opus-mt-kn-en"
# or NLLB (heavier):
# export HF_BASE_MODEL="facebook/nllb-200-distilled-600M"
# export SRC_LANG="kan_Knda"; export TGT_LANG="eng_Latn"
```

3. Start the API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit http://localhost:8000/docs for Swagger.

4. Streamlit UI (optional):
```bash
streamlit run app/ui.py
```

5. CLI example:
```bash
python scripts/translate_cli.py --text "ನೀವು ಹೇಗಿದ್ದೀರಿ?"
```


## Endpoints
```bash
POST /translate ⇒ { "text": "..." } → { "translation": "..." }
POST /translate/batch ⇒ { "texts": ["..."] } → { "translations": ["..."] }
```

## Evaluation
Compute BLEU on a small dev set:
```bash
python scripts/eval_bleu.py --preds out.txt --refs refs.txt
```

## Training (LoRA scaffold)
If you have parallel data (TSV with kn\ten):

python scripts/train_lora.py \
  --train_tsv data/train.tsv \
  --val_tsv data/val.tsv \
  --base_model "Helsinki-NLP/opus-mt-kn-en" \
  --output_dir outputs/lora-kn-en

This is a minimal scaffold intended to get you started; plug in your own datasets and tune hyperparameters.

## Docker
docker build -t kn-en-translator .
docker run -p 8000:8000 kn-en-translator

## Requirements
See requirements.txt. Python ≥ 3.9 recommended.

## License
MIT
