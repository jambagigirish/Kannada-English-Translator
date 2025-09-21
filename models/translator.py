import os
from typing import List
from dotenv import load_dotenv
load_dotenv()


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_MODEL = os.getenv("HF_BASE_MODEL", "Helsinki-NLP/opus-mt-kn-en")
SRC_LANG = os.getenv("SRC_LANG") # Only used for multilingual models like NLLB
TGT_LANG = os.getenv("TGT_LANG")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


class Translator:
def __init__(self, model_name: str | None = None, max_length: int = 256):
self.model_name = model_name or DEFAULT_MODEL
self.max_length = max_length
self._load()


def _load(self):
self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
self.model.to(DEVICE)


def translate(self, text: str) -> str:
self._maybe_set_lang()
inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
with torch.no_grad():
gen = self.model.generate(**inputs, max_length=self.max_length)
out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
return out[0]


def translate_batch(self, texts: List[str]) -> List[str]:
self._maybe_set_lang()
inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
with torch.no_grad():
gen = self.model.generate(**inputs, max_length=self.max_length)
out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
return out


def _maybe_set_lang(self):
# For NLLB, some tokenizers require src/tgt language codes
if hasattr(self.tokenizer, "src_lang") and SRC_LANG:
self.tokenizer.src_lang = SRC_LANG
if hasattr(self.tokenizer, "tgt_lang") and TGT_LANG:
self.tokenizer.tgt_lang = TGT_LANG
