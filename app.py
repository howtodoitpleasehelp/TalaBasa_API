from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# App setup
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Model setup
# ---------------------------
model_name = "smacale/talabasa_war-eng_v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# MEMORY-OPTIMIZED LOADING (CRITICAL FOR RENDER)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,        # reduces memory usage
    low_cpu_mem_usage=True            # prevents RAM spikes
)

# FORCE CPU (safer for Render free tier)
device = torch.device("cpu")
model = model.to(device)
model.eval()

# ---------------------------
# Request schema
# ---------------------------
class TranslateRequest(BaseModel):
    text: str

# ---------------------------
# API endpoint
# ---------------------------
@app.post("/translate")
def translate(req: TranslateRequest):

    if not req.text or not req.text.strip():
        return {"error": "Empty input text"}

    # Tokenize input
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    # Inference (optimized for low memory)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,   # prevents memory explosion
            num_beams=2          # lighter decoding
        )

    # Decode output
    translation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "input": req.text,
        "translation": translation
    }

# ---------------------------
# Health check endpoint
# ---------------------------
@app.get("/")
def home():
    return {"status": "TalaBasa API is running"}