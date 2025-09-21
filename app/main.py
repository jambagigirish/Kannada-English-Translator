from fastapi import FastAPI
from pydantic import BaseModel
from models.translator import Translator


app = FastAPI(title="Kannada → English Translator", version="0.1.0")
translator = Translator() # Lazy-loads the HF model on first use


class TranslateIn(BaseModel):
text: str


class BatchIn(BaseModel):
texts: list[str]


@app.get("/")
def root():
return {"status": "ok", "model": translator.model_name}


@app.post("/translate")
def translate(body: TranslateIn):
out = translator.translate(body.text)
return {"translation": out}

@app.post("/translate/batch")
def translate_batch(body: BatchIn):
outs = translator.translate_batch(body.texts)
return {"translations": outs}
