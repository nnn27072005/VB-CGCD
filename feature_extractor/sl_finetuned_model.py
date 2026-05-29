import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
import torchvision
from torch.utils.data import DataLoader

from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms

from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from peft.peft_model import PeftModel
from peft.config import PeftConfig

from tqdm import tqdm
import copy

from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import vision_transformer as vits
from vision_transformer import DINOHead

import utils

import argparse
import os
from collections import Counter

torch.manual_seed(42)  # Set random seed for reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = backbone

        self.head = head

    def forward(self, x):
        # Run the head forward on the concatenated features.
        pooler_output = self.backbone(x).pooler_output

        cls_output = self.head(pooler_output)

        return cls_output, pooler_output

def finetune_dino(train_set, num_classes, model_name="facebook/dino-vitb16"):
        
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 224
    transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
        ])

    batch_size = 128

    epochs = 10

    lora_model = load_model(num_classes, model_name)

    # Optimizer
    optimizer = AdamW(lora_model.parameters(), lr=1e-3)

    lora_model.to(device)

    labels = []
    if hasattr(train_set, "samples"):
        labels = [int(sample[1]) for sample in train_set.samples]
    if labels:
        class_counts = dict(sorted(Counter(labels).items()))
        print(
            f"[FeatureExtractor] samples={len(train_set)} "
            f"num_classes={num_classes} observed_classes={sorted(class_counts)} "
            f"class_counts={class_counts}"
        )
    else:
        print(f"[FeatureExtractor] samples={len(train_set)} num_classes={num_classes}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Training loop
    for epoch in range(epochs):  # Adjust epochs as needed
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        pooler_norm_sum = 0.0
        logit_abs_sum = 0.0
        grad_norm_sum = 0.0
        n_batches = 0

        for bidx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            output, pooler_output = lora_model(images)

            cls_loss = F.cross_entropy(output, labels)

            loss = cls_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite feature extractor loss at epoch={epoch}, batch={bidx}: {loss.item()}"
                )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=10.0)
            optimizer.step()

            with torch.no_grad():
                pred = output.argmax(dim=1)
                epoch_loss += loss.item() * labels.size(0)
                epoch_correct += (pred == labels).sum().item()
                epoch_total += labels.size(0)
                pooler_norm_sum += pooler_output.norm(dim=1).mean().item()
                logit_abs_sum += output.detach().abs().mean().item()
                grad_norm_sum += float(grad_norm)
                n_batches += 1

        avg_loss = epoch_loss / max(epoch_total, 1)
        train_acc = 100.0 * epoch_correct / max(epoch_total, 1)
        avg_pooler_norm = pooler_norm_sum / max(n_batches, 1)
        avg_logit_abs = logit_abs_sum / max(n_batches, 1)
        avg_grad_norm = grad_norm_sum / max(n_batches, 1)
        print(
            f"[FeatureExtractor] Epoch {epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f} train_acc={train_acc:.2f}% "
            f"pooler_norm={avg_pooler_norm:.4f} "
            f"logit_abs={avg_logit_abs:.4f} grad_norm={avg_grad_norm:.4f}"
        )

    return lora_model.backbone


def load_model(num_classes, model_name):

    if model_name == "dinov2_vitb14":
        model_name = "facebook/dinov2-base"

    if model_name == "dino_vitb16":
        model_name = "facebook/dino-vitb16"

    model = AutoModel.from_pretrained(model_name, use_safetensors=True)


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )

    backbone = get_peft_model(model, peft_config)

    embed_dim = backbone.config.hidden_size

    lora_model = MultiCropWrapper(backbone, torch.nn.Linear(embed_dim, num_classes))

    lora_model.train()

    backbone.print_trainable_parameters()

    return lora_model
