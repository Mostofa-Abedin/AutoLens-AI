import io
from pathlib import Path

MODEL_NAME = "openai/clip-vit-base-patch32"

_processor = None
_model = None
_device = None
_loaded = False


def load_model():
    global _processor, _model, _device, _loaded
    import torch
    from transformers import CLIPModel, CLIPProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model {MODEL_NAME} on {_device}...")
    _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    _model = CLIPModel.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()
    _loaded = True
    print("CLIP model loaded.")


def _embed(pil_image) -> list[float]:
    import torch

    if not _loaded:
        raise RuntimeError("CLIP model not loaded — call load_model() first")
    inputs = _processor(images=pil_image, return_tensors="pt").to(_device)
    with torch.no_grad():
        features = _model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()


def generate_embedding(image_path: str) -> list[float]:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    return _embed(img)


def generate_embedding_from_bytes(image_bytes: bytes) -> list[float]:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _embed(img)


def is_loaded() -> bool:
    return _loaded
