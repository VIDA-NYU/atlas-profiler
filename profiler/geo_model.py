"""The spatial column classifier (PyTorch and transformers), loaded only when it is used."""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from . import spatial
from .spatial import GEO_MODEL_REQUIRED_FILES

logger = logging.getLogger(__name__)


def mean_pool(outputs, attention_mask):
    """Mean pooling over sequence length (matches train_cta_classifier.py)."""
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class CTAClassificationModel(nn.Module):
    """Classification model with mean pooling (matches train_cta_classifier.py)."""

    def __init__(self, num_labels, config=None, use_spatial_head=False):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.use_spatial_head = use_spatial_head
        if use_spatial_head:
            self.spatial_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None, spatial_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        logits = self.classifier(pooled)

        class Output:
            pass

        out = Output()
        out.logits = logits
        out.pooled = pooled
        if self.use_spatial_head and spatial_labels is not None:
            out.spatial_logits = self.spatial_head(pooled)
        return out


class CTAContrastiveModel(nn.Module):
    """Encoder with projection head for contrastive learning (matches train_cta_classifier.py)."""

    def __init__(self, embed_dim=128, num_labels=None, config=None):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )
        self.classifier = nn.Linear(hidden_size, num_labels) if num_labels else None

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        if return_embeddings:
            return F.normalize(self.projection(pooled), dim=1)
        return self.classifier(pooled) if self.classifier else pooled

    def get_embeddings(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask, return_embeddings=True)


class GeoClassifier:
    """Unified interface for CTA classification with automatic model download."""

    def __init__(self, model_dir: str | None = None, auto_download: bool = True):
        """
        Initialize the GeoClassifier.

        Args:
            model_dir: Path to the model directory (default: bundled model if present,
                otherwise a user cache directory)
            auto_download: If True, automatically download model if not found
        """
        required_files = list(GEO_MODEL_REQUIRED_FILES)
        if model_dir is None:
            # Default to profiler/model relative to this module's location
            profiler_dir = Path(__file__).parent
            package_model_dir = profiler_dir / "model"
            if all((package_model_dir / f).exists() for f in required_files):
                model_dir = str(package_model_dir)
            else:
                cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
                model_dir = str(cache_root / "atlas_profiler" / "model")

        self.model_dir = Path(model_dir)

        missing_files = [f for f in required_files if not (self.model_dir / f).exists()]
        if missing_files:
            if auto_download:
                logger.info(
                    "Model artifact incomplete (%s missing). Downloading complete artifact...",
                    missing_files,
                )
                spatial.download_geo_model(str(self.model_dir))
            else:
                raise FileNotFoundError(
                    f"Model files not found in {self.model_dir}: {missing_files}. "
                    "Set auto_download=True or download from NYU Box."
                )

        missing_files = [f for f in required_files if not (self.model_dir / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Model artifact in {self.model_dir} is incomplete: {missing_files}"
            )

        # Load label encoder config
        with open(self.model_dir / "label_encoder.json") as f:
            config = json.load(f)

        self.classes = config["classes"]
        self.mode = config.get("mode", "classification")
        self.model_name = config.get("model_name", "BAAI/bge-small-en-v1.5")
        self.embed_dim = config.get("embed_dim", 128)
        self.name_repeat = config.get("name_repeat", 3)  # Name emphasis
        self.special_tokens = config.get("special_tokens", {})  # [COL], [VAL] tokens

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Load encoder config from saved directory
        encoder_config = AutoConfig.from_pretrained(str(self.model_dir))

        # Load checkpoint to inspect keys and determine model architecture
        checkpoint_path = self.model_dir / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_keys = set(checkpoint.keys())

        # Detect model type from checkpoint keys (prioritize checkpoint structure over mode)
        has_projection = any(k.startswith("projection.") for k in checkpoint_keys)
        has_spatial_head = any("spatial_head" in k for k in checkpoint_keys)

        # Determine which model to use based on checkpoint structure
        # If checkpoint has projection head, use contrastive model
        # Otherwise, use classification model (works for classification, fine_tune, combined modes)
        if has_projection:
            # Contrastive model (has projection head)
            self.model = CTAContrastiveModel(
                embed_dim=self.embed_dim,
                num_labels=len(self.classes),
                config=encoder_config,
            )
            logger.info(
                "Using CTAContrastiveModel (detected projection head in checkpoint, mode=%s)",
                self.mode,
            )
        else:
            # Classification model (no projection head)
            # This works for classification, fine_tune, and combined modes
            # since combined/fine_tune models are saved as classification models
            self.model = CTAClassificationModel(
                num_labels=len(self.classes),
                config=encoder_config,
                use_spatial_head=has_spatial_head,
            )
            logger.info(
                "Using CTAClassificationModel (no projection head, mode=%s, has_spatial_head=%s)",
                self.mode,
                has_spatial_head,
            )

        # Load saved weights (use strict=False to handle missing/extra keys gracefully)
        checkpoint = {
            k: v for k, v in checkpoint.items() if not k.startswith("metric_projection.")
        }
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint, strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys when loading model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()

    def _format_input(self, text: str) -> str:
        """Format input with special tokens: [COL] name [VAL] val1 [VAL] val2..."""
        # Parse "name: val1, val2, val3" format
        if ": " not in text:
            return text

        name, values = text.split(": ", 1)
        col_tok = self.special_tokens.get("col_token", "")
        val_tok = self.special_tokens.get("val_token", "")

        # Format column name with repetition
        if col_tok and name:
            if self.name_repeat > 1:
                col_parts = " ".join([f"{col_tok} {name}"] * self.name_repeat)
            else:
                col_parts = f"{col_tok} {name}"
        else:
            col_parts = name

        # Format values with [VAL] token
        if val_tok:
            val_list = [v.strip() for v in str(values).split(",")]
            val_parts = " ".join([f"{val_tok} {v}" for v in val_list[:10]])
        else:
            val_parts = values

        return f"{col_parts} {val_parts}".strip()

    def predict_batch(
        self,
        texts: list[str],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Batch prediction for multiple columns in ONE forward pass.

        Args:
            texts: List of texts in format "column_name: val1, val2, val3"
            threshold: Confidence threshold below which to return non_spatial

        Returns:
            List of dicts with 'label' and 'confidence' for each input
        """
        if not texts:
            return []

        # Format all inputs
        formatted_texts = [self._format_input(t) for t in texts]

        # Batch tokenize (much faster than tokenizing one by one)
        encodings = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,  # Pad to longest in batch (more efficient)
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # SINGLE forward pass for ALL columns!
            # Handle both model types
            if isinstance(self.model, CTAClassificationModel):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:  # CTAContrastiveModel
                logits = self.model(input_ids, attention_mask)
                # If contrastive model doesn't have classifier, this won't work
                # But based on training code, it should have classifier for inference

            probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]

        # Process results for each column
        results = []
        for i in range(len(texts)):
            top_prob, top_idx = probs[i].max(dim=0)
            label = self.classes[top_idx.item()]
            confidence = top_prob.item()

            if threshold is not None and confidence < threshold:
                results.append(
                    {
                        "label": "non_spatial",
                        "confidence": 1.0 - confidence,
                    }
                )
            else:
                results.append(
                    {
                        "label": label,
                        "confidence": confidence,
                    }
                )

        return results
