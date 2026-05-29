#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Hao Dai

import os
import sys
import time
import argparse

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

import torch

import jax
import jax.numpy as jnp

from dataloaders.cifar100 import CIFAR100Loader
from dataloaders.tinyimagenet import TinyImageNetLoader
from dataloaders.imagenet100 import ImageNet100Loader
from dataloaders.cub200 import CUB200Loader
from dataloaders.gc10det import GC10DETLoader

from classifier.mngmm import MNGMMClassifier

from clustering.gmm import GMMCluster

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.losses import Debiased_Representation_Loss

class ImageStageDataset(Dataset):
    def __init__(self, data_point, transform=None):
        self.features = data_point._x 
        self.labels = data_point._y
        self.images = data_point._img  
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img_ref = self.images[idx]
        if isinstance(img_ref, (str, bytes, os.PathLike)):
            img = Image.open(img_ref).convert("RGB")
        else:
            img = Image.fromarray(img_ref).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.features[idx], self.labels[idx]

class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536, hidden_dim=2048, bottleneck_dim=384): # bottleneck mapped to 384
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x, logits

def build_models(device, out_dim):
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained('facebook/dino-vitb16')
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()  
    
    projector = DINOProjectionHead(in_dim=768, out_dim=out_dim)
    return backbone.to(device), projector.to(device)

dino_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def debias_dataset(backbone, projector, data_obj, batch_size, device):
    """Extract debiased 384-dim features using MLP output WITHOUT L2 normalization.
    
    The L2 norm projects features onto a unit sphere where Gaussian modeling
    fails (det(cov) underflows to 0 in 384-dim). The MLP output is in
    unconstrained Euclidean space — appropriate for MultivariateNormal.
    """
    ds = ImageStageDataset(data_obj, transform=dino_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            base_f = backbone(images).pooler_output
            z = projector.mlp(base_f)  # Pre-normalization: Euclidean, not spherical
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)

class SimpleData:
    """Lightweight data wrapper with _x and _y attributes for the classifier."""
    def __init__(self, x, y):
        self._x = x
        self._y = y

# get the current time with a format yyyyMMdd-HHmm
def get_current_time():
    return time.strftime("%Y%m%d-%H%M", time.localtime())

def Clustering_alg(alg):
    if alg == 'gmm':
        return GMMCluster
    else:
        raise ValueError('Clustering algorithm not supported')

def Classifier_alg(alg):
    if alg == 'mngmm':
        return MNGMMClassifier
    else:
        raise ValueError('Classifier algorithm not supported')

def load_mode(args, loader):
    if args.load_mode == 't5':
        return loader.makeT5Loader()
    elif args.load_mode == 'vin':
        return loader.makeVinLoader()
    elif args.load_mode == 't10':
        return loader.makeT10Loader()
    else:
        raise ValueError('Load mode not supported')

def Dataloader(args):
    # make data loader
    if args.dataset == 'cifar100':
        cifar100loader = CIFAR100Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args,cifar100loader)

    elif args.dataset == 'tinyimagenet':
        tinyimagenetloader = TinyImageNetLoader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, tinyimagenetloader)

    elif args.dataset == 'imagenet100':
        imageNet100Loader = ImageNet100Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, imageNet100Loader)

    elif args.dataset == 'cub200':
        cub200Loader = CUB200Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, cub200Loader)

    elif args.dataset == 'gc10det':
        gc10detLoader = GC10DETLoader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, gc10detLoader)

    else:
        raise ValueError('Dataset not supported')

    return train_loader, test_loader, test_old_loader, test_all_loader


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Generalized Class Incremental Learning')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to learn')
    parser.add_argument('--data_dir', type=str, default='datasets/cifar100', help='Directory to the data')
    parser.add_argument('--load_mode', type=str, default='t5', help='Dataset Loader Mode (t5 / t10 / vin)')
    parser.add_argument('--pretrained_model_name', type=str, default='dino-vitb16', help='Name of the model')
    parser.add_argument('--base', type=int, default=50, help='Number of base classes')
    parser.add_argument('--increment', type=int, default=10, help='Number of incremental classes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--trail_name', type=str, default=f'', help='Name of the trail')
    parser.add_argument('--clustering_alg', type=str, default='gmm', help='Clustering algorithm')
    parser.add_argument('--classifier_alg', type=str, default='mngmm', help='Classifier algorithm')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes for the classifier')
    parser.add_argument('--num_dim', type=int, default=384, help='Number of features\' dim for the classifier')
    parser.add_argument('--with_early_stop', default=True, action=argparse.BooleanOptionalAction, help='Whether to use early stop')
    parser.add_argument('--use_correct_scaling_factor', default=True, action=argparse.BooleanOptionalAction, help='Whether to use correct scaling factor')
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=4e-6, help="Learning rate")
    parser.add_argument("--scaling-factor", type=float, default=1.2, help="Scaling factor for learning from arbitary")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--early_stop_ratio", type=float, default=0, help="R in early stop")
    args = parser.parse_args()

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)

    Clustering = Clustering_alg(args.clustering_alg)

    Classifier = Classifier_alg(args.classifier_alg)

    train_loader, test_loader, test_old_loader, test_all_loader = Dataloader(args)

    # if saved_models dir does not exist, create it
    log_saved_dir = f"{args.trail_name}_{get_current_time()}"
    if not os.path.exists(f"logs/{log_saved_dir}/saved_models"):
        os.makedirs(f"logs/{log_saved_dir}/saved_models")

    # same classifier for all stages, static expension
    s_classifier = Classifier(num_classes=args.num_classes, num_dim=args.num_dim, with_early_stop=args.with_early_stop)

    print(f"scaling factor: {args.scaling_factor}")
    s_classifier.init_parameters(n_epochs=args.n_epochs, lr=args.lr, log_dir=f"logs/{log_saved_dir}/log/stage0", save_dir=f"logs/{log_saved_dir}/saved_models/stage0", batch_size=args.batch_size, increment=args.increment, base=args.base, scaling_factor=args.scaling_factor, use_correct_scaling_factor=args.use_correct_scaling_factor, early_stop_ratio=args.early_stop_ratio)

    for i, (train_data, test_data, test_old_data, test_all_data) in enumerate(zip(train_loader, test_loader, test_old_loader, test_all_loader)): 
        if i == 0:
            testing_set = {'test_old': test_data, 'test_all': test_data, 'known_test': test_data}

            s_classifier.run(train_data._x, train_data._y, test_data._x, test_data._y, current_stage=i, testing_set=testing_set)

            known_test_data = test_data

        else:

            label_offset = args.base + (i-1)*args.increment

            # ==============================================================================
            # START: ACTIVE REPRESENTATION LEARNING LOOP
            # ==============================================================================
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Dynamic Index Tracking
            old_class_indices = list(range(args.base + (i-1)*args.increment))
            new_class_indices = list(range(args.base + (i-1)*args.increment, args.base + i*args.increment))
            
            # Setup Dataloader
            stage_dataset = ImageStageDataset(train_data, transform=dino_transform)
            stage_loader = DataLoader(stage_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

            # Initialize Models & Loss
            out_dim = args.base + i*args.increment 
            if 'projector' not in locals():
                backbone, projector = build_models(device, out_dim=args.num_classes)
                optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=1e-4)
                # Ensure feature_dim matches our 384 bottleneck!
                debiased_loss_fn = Debiased_Representation_Loss(feature_dim=384, hidden_dim=128).to(device)
            
            print(f"--- Starting Debiased Representation Learning (Stage {i}) ---")
            projector.train()
            representation_epochs = 10 # Adjustable

            for epoch in range(representation_epochs):
                epoch_loss = 0.0
                epoch_dict = {'loss_entropy_inter': 0, 'loss_entropy_old_in': 0, 'loss_entropy_new_in': 0, 'loss_contrastive': 0}
                
                for images, static_feats, labels in stage_loader:
                    images = images.to(device)
                    
                    with torch.no_grad():
                        base_features = backbone(images).pooler_output
                        
                    z_u, logits = projector(base_features)
                    
                    loss, loss_dict = debiased_loss_fn(
                        z_u=z_u, 
                        logits=logits, 
                        old_class_indices=old_class_indices, 
                        new_class_indices=new_class_indices,
                        base_features=base_features
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    for k in epoch_dict.keys():
                        epoch_dict[k] += loss_dict.get(k, 0.0)
                    
                n_batches = len(stage_loader)
                print(f"Epoch {epoch+1}/{representation_epochs} | Loss: {epoch_loss/n_batches:.4f} "
                      f"| Ent_Inter: {epoch_dict['loss_entropy_inter']/n_batches:.4f} "
                      f"| Ent_Old: {epoch_dict['loss_entropy_old_in']/n_batches:.4f} "
                      f"| Ent_New: {epoch_dict['loss_entropy_new_in']/n_batches:.4f} "
                      f"| Contra: {epoch_dict['loss_contrastive']/n_batches:.4f}")

            # Re-extract Debiased Features for ALL datasets
            print(f"--- Re-extracting Features for VB Pipeline ---")
            projector.eval()

            # Training data
            debiased_features = debias_dataset(backbone, projector, train_data, args.batch_size, device)

            # All test sets — must be in the same debiased space
            debiased_test = debias_dataset(backbone, projector, test_data, args.batch_size, device)
            debiased_test_old = debias_dataset(backbone, projector, test_old_data, args.batch_size, device)
            debiased_test_all = debias_dataset(backbone, projector, test_all_data, args.batch_size, device)
            debiased_known_test = debias_dataset(backbone, projector, known_test_data, args.batch_size, device)

            # Standardize: zero mean + unit variance per dimension.
            # This mirrors what PCA does for Stage 0, making identity covariance
            # a valid initialization and preventing float32 det underflow.
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            debiased_features = scaler.fit_transform(debiased_features)
            debiased_test = scaler.transform(debiased_test)
            debiased_test_old = scaler.transform(debiased_test_old)
            debiased_test_all = scaler.transform(debiased_test_all)
            debiased_known_test = scaler.transform(debiased_known_test)
            # ==============================================================================
            # END: ACTIVE REPRESENTATION LEARNING LOOP
            # ==============================================================================

            clustering = Clustering(num_classes=args.increment, label_offset=label_offset)

            print("Clustering novel classes:", args.increment, "Offset:", label_offset)
            
            # Separate old (replay buffer) and novel data
            novel_mask = train_data._y >= label_offset
            old_mask = ~novel_mask
            
            # 1. Fit clustering ONLY on the highly-separated Debiased Latent Space of NOVEL samples
            novel_debiased_features = debiased_features[novel_mask]
            clustering.fit(novel_debiased_features)

            # 2. Predict pseudo-labels ONLY for novel features
            novel_pred = clustering.predict(novel_debiased_features, train_data._y[novel_mask], with_known=False)

            # 3. Construct the perfect pred array! 
            # Replay buffer gets true old labels. Novel data gets clustering pseudo-labels!
            pred = np.copy(train_data._y)
            pred[novel_mask] = novel_pred

            print("Combined Pred Unique Counts:", np.unique(pred, return_counts=True))

            # ==============================================================================
            # Re-initialize global_params in the standardized debiased feature space.
            # Identity covariance is appropriate because features are standardized
            # (zero mean, unit variance) — same property that PCA gives Stage 0.
            # det(I) = 1, trivially PD, no float32 underflow.
            # ==============================================================================
            if s_classifier.global_params is not None:
                old_debiased = debiased_features[old_mask]
                old_labels = train_data._y[old_mask]
                num_dim = debiased_features.shape[1]

                new_means = np.zeros((s_classifier.num_classes, num_dim))
                new_covs = np.stack([np.eye(num_dim)] * s_classifier.num_classes)

                for c in range(label_offset):
                    c_mask = old_labels == c
                    if c_mask.sum() > 0:
                        new_means[c] = old_debiased[c_mask].mean(axis=0)

                s_classifier.global_params = {
                    'class_means': jnp.array(new_means, dtype=jnp.float32),
                    'class_covs': jnp.array(new_covs, dtype=jnp.float32)
                }
                s_classifier.pca = None  # Debiased features are already 384-dim
                print(f"Re-initialized global_params for {label_offset} old classes in debiased space")

            # for ngmm merge
            s_classifier._set_label_offset(label_offset)

            s_classifier.update_dir_infos(log_dir=f"logs/{log_saved_dir}/log/stage{i}", save_dir=f"logs/{log_saved_dir}/saved_models/stage{i}")

            # Build testing_set with debiased features
            testing_set = {
                'test_old': SimpleData(debiased_test_old, test_old_data._y),
                'test_all': SimpleData(debiased_test_all, test_all_data._y),
                'known_test': SimpleData(debiased_known_test, known_test_data._y)
            }

            # Train and evaluate in the unified debiased feature space
            s_classifier.run(debiased_features, pred, debiased_test, test_data._y, current_stage=i, testing_set=testing_set)
