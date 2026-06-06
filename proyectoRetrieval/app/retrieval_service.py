import base64
import io
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import faiss
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = DATA_DIR / "modelo_final_retrieval.pth"
TOKENIZER_DIR = DATA_DIR / "tokenizer_final_retrieval"
CONFIG_PATH = DATA_DIR / "best_config.pkl"


class SimpleFlickrDataset(Dataset):
    def __init__(self, hf_dataset: Any, split: str = "train", limit: int = 200):
        self.data = hf_dataset[split][:limit]

    def __len__(self) -> int:
        return len(self.data["image"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "image": self.data["image"][idx].convert("RGB"),
            "caption": self.data["caption_0"][idx],
        }


class ImageCNN(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(1)
        return F.normalize(self.proj(features), dim=-1)


class ImageTransformer(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()
        self.backbone = vit
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return F.normalize(self.proj(features), dim=-1)


class TextEncoderModel(nn.Module):
    def __init__(self, model_name: str, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        return F.normalize(self.proj(cls_token), dim=-1)


class TextDecoderModel(nn.Module):
    def __init__(self, model_name: str, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_tokens = out.last_hidden_state[torch.arange(batch_size), sequence_lengths]
        return F.normalize(self.proj(last_tokens), dim=-1)


def get_model_image(model_name: str) -> tuple[nn.Module, transforms.Compose]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if model_name == "CNN":
        return ImageCNN().to(DEVICE), transform
    return ImageTransformer().to(DEVICE), transform


def get_model_text(model_name: str, text_arch_type: str) -> tuple[nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if text_arch_type == "encoder":
        return TextEncoderModel(model_name).to(DEVICE), tokenizer
    return TextDecoderModel(model_name).to(DEVICE), tokenizer


def collate_fn(batch: list[dict[str, Any]], tokenizer: AutoTokenizer, img_transform: transforms.Compose) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.stack([img_transform(item["image"]) for item in batch])
    texts = [item["caption"] for item in batch]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=40, return_tensors="pt")
    return images, tokens["input_ids"], tokens["attention_mask"]


def compress_embedding(embedding: np.ndarray) -> dict[str, Any]:
    array = np.asarray(embedding, dtype=np.float16)
    compressed = zlib.compress(array.tobytes(), level=9)
    return {
        "encoding": "base64+zlib",
        "dtype": "float16",
        "shape": list(array.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def image_to_base64(image: Image.Image, max_size: tuple[int, int] = (320, 320), quality: int = 78) -> str:
    image_copy = image.convert("RGB")
    image_copy.thumbnail(max_size)
    buffer = io.BytesIO()
    image_copy.save(buffer, format="JPEG", optimize=True, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class RetrievalArtifacts:
    image_embeddings: np.ndarray
    text_embeddings: np.ndarray
    captions: list[str]
    images: list[Image.Image]
    index_img: faiss.Index
    index_txt: faiss.Index


class RetrievalService:
    def __init__(self) -> None:
        self.config = self._load_config()
        self.tokenizer: AutoTokenizer | None = None
        self.text_model: nn.Module | None = None
        self.image_model: nn.Module | None = None
        self.img_transform: transforms.Compose | None = None
        self.artifacts: RetrievalArtifacts | None = None
        self.dataset_limit = 100

    def _load_config(self) -> dict[str, Any]:
        config = pd.read_pickle(CONFIG_PATH)
        if hasattr(config, "to_dict"):
            return config.to_dict()
        return dict(config)

    def load(self) -> None:
        if self.artifacts is not None:
            return

        image_model_name = self.config["Modelo_Imagen"]
        text_model_name = self.config["Modelo_Texto"]
        text_type = self.config["Tipo_Texto"]

        self.image_model, self.img_transform = get_model_image(image_model_name)
        self.text_model, self.tokenizer = get_model_text(text_model_name, text_type)

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.text_model.load_state_dict(state_dict)
        self.text_model.eval()
        self.image_model.eval()

        dataset = load_dataset("jxie/flickr8k")
        test_ds = SimpleFlickrDataset(dataset, "test", limit=self.dataset_limit)

        image_embeddings: list[np.ndarray] = []
        text_embeddings: list[np.ndarray] = []
        images: list[Image.Image] = []
        captions: list[str] = []

        batch_size = 16
        for start in range(0, len(test_ds), batch_size):
            batch = [test_ds[idx] for idx in range(start, min(start + batch_size, len(test_ds)))]
            imgs, ids, masks = collate_fn(batch, self.tokenizer, self.img_transform)
            imgs = imgs.to(DEVICE)
            ids = ids.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.no_grad():
                img_feat = self.image_model(imgs).cpu().numpy().astype("float32")
                txt_feat = self.text_model(ids, masks).cpu().numpy().astype("float32")

            image_embeddings.append(img_feat)
            text_embeddings.append(txt_feat)
            images.extend([item["image"] for item in batch])
            captions.extend([item["caption"] for item in batch])

        self.artifacts = RetrievalArtifacts(
            image_embeddings=np.concatenate(image_embeddings, axis=0),
            text_embeddings=np.concatenate(text_embeddings, axis=0),
            captions=captions,
            images=images,
            index_img=faiss.IndexFlatIP(EMBED_DIM),
            index_txt=faiss.IndexFlatIP(EMBED_DIM),
        )
        self.artifacts.index_img.add(self.artifacts.image_embeddings)
        self.artifacts.index_txt.add(self.artifacts.text_embeddings)

    def retrieve(self, query_text: str, top_k: int = 3) -> dict[str, Any]:
        self.load()
        assert self.artifacts is not None
        assert self.text_model is not None
        assert self.tokenizer is not None

        tokens = self.tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=40,
            return_tensors="pt",
        )
        ids = tokens["input_ids"].to(DEVICE)
        masks = tokens["attention_mask"].to(DEVICE)

        with torch.no_grad():
            query_embedding = self.text_model(ids, masks).cpu().numpy().astype("float32")[0]

        k = max(1, min(int(top_k), len(self.artifacts.captions)))
        query_batch = np.expand_dims(query_embedding, axis=0)
        image_scores, image_top_indices = self.artifacts.index_img.search(query_batch, k)
        text_scores, text_top_indices = self.artifacts.index_txt.search(query_batch, k)

        image_results = []
        for rank, image_idx in enumerate(image_top_indices[0], start=1):
            aligned_idx = int(image_idx)
            image_results.append(
                {
                    "rank": rank,
                    "image_index": aligned_idx,
                    "caption_index": aligned_idx,
                    "image_score": float(image_scores[0][rank - 1]),
                    "caption_score": float(np.dot(self.artifacts.text_embeddings[aligned_idx], query_embedding)),
                    "caption": self.artifacts.captions[aligned_idx],
                    "image_embedding": compress_embedding(self.artifacts.image_embeddings[image_idx]),
                    "caption_embedding": compress_embedding(self.artifacts.text_embeddings[aligned_idx]),
                    "image_base64": image_to_base64(self.artifacts.images[image_idx]),
                }
            )

        text_results = []
        for rank, text_idx in enumerate(text_top_indices[0], start=1):
            aligned_idx = int(text_idx)
            text_results.append(
                {
                    "rank": rank,
                    "caption_index": aligned_idx,
                    "caption_score": float(text_scores[0][rank - 1]),
                    "caption": self.artifacts.captions[aligned_idx],
                    "caption_embedding": compress_embedding(self.artifacts.text_embeddings[aligned_idx]),
                }
            )

        return {
            "query": query_text,
            "k": k,
            "device": str(DEVICE),
            "config": self.config,
            "query_embedding": compress_embedding(query_embedding),
            "image_results": image_results,
            "text_results": text_results,
        }


retrieval_service = RetrievalService()