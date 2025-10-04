# use_frs_classifier.py
"""
Load saved ./frs_classifier and classify input text.
- Forces single-token mappings for certain keywords (absent, forgot, frs, issue, etc.)
- Maps other single tokens (like 'unable') to Other
- For multi-token input uses the model and returns (label, confidence, probs)
Run:
  python use_frs_classifier.py
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model
MODEL_REPO = "shaikumar0/text-classifier-model"  # Hugging Face model name
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO, use_auth_token=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO, use_auth_token=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Completed classes (must match index/order used during training)
classes = ["Absent", "Forgot FRS", "FRS Issue", "Other"]

# Preprocess (must match training)
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    text = " ".join(text.split())
    return text

# Single-token forced mapping (keys must be lowercase to match preprocess)
single_token_map = {
    "absent": "Absent",
    # forgot variants
    "forgot": "Forgot FRS",
    "forget": "Forgot FRS",
    "missed": "Forgot FRS",
    "miss": "Forgot FRS",
    "frs": "Forgot FRS",
    # issue variants (lowercase)
    "frs issue": "FRS Issue",
    "frs error": "FRS Issue",
    "frs bug": "FRS Issue",
    "frs crash": "FRS Issue",
    
 
}

# tokens that if single-word should be Other
single_token_other = {"unable", "sorry", "hi", "hello", "ok", "thanks", "please","not absent","not forgot","not issue","not frs", "not working"}

def classify_message(text: str, threshold: float = 0.70, return_probs: bool = True):
    """
    Returns (label, confidence, probs_dict) if return_probs True
    or (label, confidence) otherwise.
    """
    text_clean = preprocess(text)
    text_clean = text_clean.lower()  # ensure lowercase assigned
    text_clean = text_clean.replace("frs","face recognization system")

    if text_clean == "":
        probs = {c: 0.0 for c in classes}
        return ("Other", 0.0, probs) if return_probs else ("Other", 0.0)

    tokens = text_clean.split()

    # 1) Single-token forced mapping
    if " ".join(tokens) in single_token_map:
        label = single_token_map[token.join(" ")]
        probs = {c: (1.0 if c == label else 0.0) for c in classes}
        return (label, 1.0, probs) if return_probs else (label, 1.0)
    if " ".join(tokens) in single_token_other:
        probs = {c: (1.0 if c == "Other" else 0.0) for c in classes}
        return ("Other", 1.0, probs) if return_probs else ("Other", 1.0)
    # default single unknown token -> Other
    if len(tokens) == 1 : 
        probs = {c: (1.0 if c == "Other" else 0.0) for c in classes}
        return ("Other", 1.0, probs) if return_probs else ("Other", 1.0)

    # 2) Multi-token rule overrides (quick fixes)
    # If text mentions frs + words that indicate forgot -> Forgot FRS
    if ("frs" in text_clean or "face recognition system" in text_clean) and any(
        k in text_clean for k in ("forget", "forgot", "miss", "missed", "unable", "didn't", "didnt")
    ):
        probs = {c: (1.0 if c == "Forgot FRS" else 0.0) for c in classes}
        return ("Forgot FRS", 1.0, probs) if return_probs else ("Forgot FRS", 1.0)

    # If text contains 'unable' + 'come'/'attend' -> Absent
    if "unable" in text_clean and any(k in text_clean for k in ("come", "attend", "office", "join")):
        probs = {c: (1.0 if c == "Absent" else 0.0) for c in classes}
        return ("Absent", 1.0, probs) if return_probs else ("Absent", 1.0)

    # 3) Model prediction
    inputs = tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits
        probs_tensor = F.softmax(logits, dim=1).squeeze(0)  # shape (num_labels,)
        conf, pred_idx = torch.max(probs_tensor, dim=0)
        conf = float(conf.cpu().item())
        pred_idx = int(pred_idx.cpu().item())

    probs = {classes[i]: float(probs_tensor[i].cpu().item()) for i in range(len(classes))}

    # If model confidence is low, treat as Other
    if conf < threshold:
        return ("Other", conf, probs) if return_probs else ("Other", conf)

    label = classes[pred_idx]
    return (label, conf, probs) if return_probs else (label, conf)


# Interactive demo
if __name__ == "__main__":
    print("Enter message (or type 'exit'):")
    while True:
        s = input("> ").strip()
        if s.lower() in ("exit", "quit"):
            break
        label, conf, probs = classify_message(s, threshold=0.70, return_probs=True)
        print(f"Predicted: {label}  |  confidence: {conf:.3f}")
        print(f"Probs: {probs}")

