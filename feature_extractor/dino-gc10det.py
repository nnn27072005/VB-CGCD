#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from sl_finetuned_model import finetune_dino


GC10DET_CLASSES = [
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "rolled_pit",
    "crease",
    "waist_folding",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class GC10DETDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return {"images": self.transform(image), "labels": torch.tensor(label, dtype=torch.long)}


def normalize_class_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_image_files(root):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def extract_class_from_json(value):
    if isinstance(value, dict):
        for key in ("classTitle", "class", "label", "category", "category_name", "name"):
            if key in value and isinstance(value[key], str):
                return value[key]
        for key in ("objects", "annotations", "shapes"):
            objects = value.get(key)
            if isinstance(objects, list) and objects:
                return extract_class_from_json(objects[0])
        for nested in value.values():
            result = extract_class_from_json(nested)
            if result is not None:
                return result
    elif isinstance(value, list) and value:
        return extract_class_from_json(value[0])
    return None


def find_matching_annotation(image_path, ann_dir):
    candidates = [
        ann_dir / f"{image_path.name}.json",
        ann_dir / f"{image_path.stem}.json",
        ann_dir / image_path.with_suffix(".json").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(ann_dir.rglob(f"{image_path.name}.json")) + list(ann_dir.rglob(f"{image_path.stem}.json"))
    return matches[0] if matches else None


def samples_from_annotations(root):
    ann_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in {"ann", "anns", "annotations"}]
    if not ann_dirs:
        return []

    ann_dir = ann_dirs[0]
    samples = []
    for image_path in find_image_files(root):
        if ann_dir in image_path.parents:
            continue
        ann_path = find_matching_annotation(image_path, ann_dir)
        if ann_path is None:
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        class_name = extract_class_from_json(annotation)
        if class_name is not None:
            samples.append((str(image_path.resolve()), normalize_class_name(class_name)))
    return samples


def samples_from_class_folders(root):
    samples = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        image_files = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        for image_path in image_files:
            samples.append((str(image_path.resolve()), normalize_class_name(class_dir.name)))
    return samples


def load_gc10det_samples(raw_data_dir):
    root = Path(raw_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"GC10-DET raw data directory does not exist: {root}")

    samples = samples_from_annotations(root)
    if not samples:
        samples = samples_from_class_folders(root)
    if not samples:
        raise ValueError(
            "Could not discover GC10-DET samples. Expected an ann/annotations folder "
            "with JSON files, or class-named folders containing images."
        )

    discovered = sorted({label for _, label in samples})
    known_order = [c for c in GC10DET_CLASSES if c in discovered]
    class_names = known_order + [c for c in discovered if c not in known_order]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    indexed_samples = [(path, class_to_idx[label]) for path, label in samples if label in class_to_idx]
    return indexed_samples, class_names


def stratified_split(samples, test_ratio, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train_samples = []
    test_samples = []
    for label in sorted(by_class):
        class_samples = by_class[label]
        rng.shuffle(class_samples)
        n_test = max(1, int(round(len(class_samples) * test_ratio)))
        n_test = min(n_test, len(class_samples) - 1)
        test_samples.extend(class_samples[:n_test])
        train_samples.extend(class_samples[n_test:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def build_transform():
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 224
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(image_size / crop_pct), interpolation),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ])


def load_dino_model(args, train_dataset, num_classes):
    if not args.finetuned:
        if args.model == "dinov2_vits14":
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif args.model == "dinov2_vitb14":
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif args.model == "dino_vitb16":
            dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif args.model == "dino_vitb8":
            dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        elif args.model == "dino_vits16":
            dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        else:
            raise ValueError("Model not supported")
        return dino, args.model.replace("_", "-")

    labeled_samples = [sample for sample in train_dataset.samples if sample[1] < args.labeled_classes]
    finetune_dataset = GC10DETDataset(labeled_samples, train_dataset.transform)
    dino = finetune_dino(finetune_dataset, num_classes, model_name=args.model)
    return dino, args.model.replace("_", "-") + "-sl"


def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, args):
    dino.to(device)
    dino.eval()

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for bidx, batch in enumerate(data_loader):
        images = batch["images"].to(device)
        if args.finetuned:
            features = dino(images).pooler_output
        else:
            features = dino(images)
        np.save(f"{features_dir}/features_{bidx}", features.cpu().data)
        np.save(f"{labels_dir}/labels_{bidx}", batch["labels"].cpu().data)


def merge_npy(features_dir, labels_dir, prefix, model_name, output_dir):
    feature_files = sorted([os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.npy')])

    assert len(feature_files) == len(label_files), "Mismatch in number of feature and label files"

    def merged_array(files):
        arrays = [np.load(f) for f in files]
        return np.concatenate(arrays, axis=0)

    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    np.save(f"{output_dir}/{model_name}/{prefix['feature']}-{model_name}.npy", merged_array(feature_files))
    np.save(f"{output_dir}/{model_name}/{prefix['label']}-{model_name}.npy", merged_array(label_files))


def save_image_paths(samples, output_dir, model_name, prefix):
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    paths = np.array([sample[0] for sample in samples])
    np.save(f"{output_dir}/{model_name}/{prefix}_images-{model_name}.npy", paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINO Inference on GC10-DET')
    parser.add_argument('--raw_data_dir', required=True, type=str, help='Directory of the extracted Kaggle GC10-DET dataset')
    parser.add_argument('--device', default='cuda', type=str, help='Device on which to run')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of dataloader workers')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument("--model", default="dino_vitb16", type=str, help="Model name")
    parser.add_argument("--output_dir", default="datasets/gc10det", type=str, help="Output directory")
    parser.add_argument("--finetuned", action='store_true', help='Finetuned model')
    parser.add_argument("--labeled_classes", default=5, type=int, help="Number of labeled classes for supervised fine-tuning")
    parser.add_argument("--test_ratio", default=0.2, type=float, help="Per-class test split ratio")
    args = parser.parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)

    samples, class_names = load_gc10det_samples(args.raw_data_dir)
    if len(class_names) != 10:
        print(f"Warning: expected 10 GC10-DET classes, discovered {len(class_names)}: {class_names}")

    train_samples, test_samples = stratified_split(samples, args.test_ratio, args.seed)
    transform = build_transform()
    train_dataset = GC10DETDataset(train_samples, transform)
    test_dataset = GC10DETDataset(test_samples, transform)

    dino, model_name = load_dino_model(args, train_dataset, num_classes=len(class_names))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump({name: idx for idx, name in enumerate(class_names)}, f, indent=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    infer_features_labels(
        dino,
        train_loader,
        f"{args.output_dir}/{args.model}_features",
        f"{args.output_dir}/{args.model}_labels",
        args.device,
        args,
    )
    merge_npy(
        f"{args.output_dir}/{args.model}_features",
        f"{args.output_dir}/{args.model}_labels",
        {"feature": "features", "label": "labels"},
        model_name,
        args.output_dir,
    )
    save_image_paths(train_samples, args.output_dir, model_name, "train")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    infer_features_labels(
        dino,
        test_loader,
        f"{args.output_dir}/{args.model}_test_features",
        f"{args.output_dir}/{args.model}_test_labels",
        args.device,
        args,
    )
    merge_npy(
        f"{args.output_dir}/{args.model}_test_features",
        f"{args.output_dir}/{args.model}_test_labels",
        {"feature": "test_features", "label": "test_labels"},
        model_name,
        args.output_dir,
    )
    save_image_paths(test_samples, args.output_dir, model_name, "test")
