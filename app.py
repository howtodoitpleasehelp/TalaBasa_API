from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "smacale/talabasa_war-eng_v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

class Request(BaseModel):
    text: str

@app.post("/translate")
def translate(req: Request):

    if not req.text.strip():
        return {"error": "Empty input"}

    inputs = tokenizer(req.text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"translation": result}