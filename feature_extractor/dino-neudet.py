%%writefile /kaggle/working/VB-CGCD-main/feature_extractor/dino-neu.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/05/29
# @Author  : Truong An Khang

import argparse
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, args):
    dino.to(device)
    dino.eval()

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"[+] Tiến hành trích xuất đặc trưng vector...")
    for bidx, (images, labels) in enumerate(tqdm(data_loader)):
        images = images.to(device)

        with torch.no_grad():
            if args.finetuned:
                features = dino(images).pooler_output
            else:
                features = dino(images)

        # Lưu thành từng file batch nhỏ trong thư mục con y hệt cấu trúc tác giả
        np.save(f"{features_dir}/features_{bidx}.npy", features.cpu().data.numpy())
        np.save(f"{labels_dir}/labels_{bidx}.npy", labels.cpu().data.numpy())


def merge_npy(features_dir, labels_dir, prefix, model_name, output_dir):
    feature_files = sorted([os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.npy')])

    assert len(feature_files) == len(label_files), "Mismatch in number of feature and label files"

    def merged_array(files):
        arrays = [np.load(f) for f in files]
        return np.concatenate(arrays, axis=0)

    # Thư mục đích cuối cùng chứa file tổng hợp
    final_output_dir = f"{output_dir}/{model_name}"
    os.makedirs(final_output_dir, exist_ok=True)
    
    np.save(f"{final_output_dir}/{prefix['feature']}-{model_name}.npy", merged_array(feature_files))
    np.save(f"{final_output_dir}/{prefix['label']}-{model_name}.npy", merged_array(label_files))
    print(f"[=>] Đã gom cụm dữ liệu tổng hợp tại: {final_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINO Feature Extraction on NEU-CLS')

    parser.add_argument(
        '--device',
        default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),
        type=str,
        help='Device on which to run'
    )
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--model", default="dinov2_vitb14", type=str, help="Model name")
    
    # Đưa thư mục đầu ra vào đúng nhóm 'output-features/neu-det' để cô lập dữ liệu
    parser.add_argument("--output_dir", default="output-features/neu-det", type=str, help="Output directory")
    parser.add_argument("--finetuned", action='store_true', help='Finetuned model')

    args = parser.parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    # Đọc trực tiếp từ thư mục ảnh thô NEU_GCD_Dataset
    dataset_root = "NEU_GCD_Dataset"

    print(f"[*] Đang quét toàn bộ ảnh lỗi từ danh mục: {dataset_root}")
    full_dataset = ImageFolder(root=dataset_root, transform=train_transforms)
    
    # Chia dataset gốc thành 2 luồng Train/Test ảo ngay trong bộ nhớ theo tỷ lệ 80/20
    generator = torch.Generator().manual_seed(args.seed)
    train_sub, valid_sub = torch.utils.data.random_split(full_dataset, [0.8, 0.2], generator=generator)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(valid_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model_name = args.model.replace("_", "-")
    # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14' if args.model == "dinov2_vitb14" else args.model)
    model_name = args.model.replace("_", "-")
    
    print(f"[+] Đang khởi tạo kiến trúc mạng từ Torch Hub cho mô hình: {args.model}")
    # dùng các dòng DINOv2 thế hệ mới
    if "dinov2" in args.model:
        dino = torch.hub.load('facebookresearch/dinov2', args.model)
    
    # dùng các dòng DINOv1 đời đầu (như dino_vitb16)
    elif "dino" in args.model:
        dino = torch.hub.load('facebookresearch/dino:main', args.model)
        
    else:
        raise ValueError(f"Mô hình {args.model} không được hỗ trợ trong luồng cấu hình này.")
        

    # --- 1. XỬ LÝ TRÍCH XUẤT TẬP TRAIN (Sinh ra folder chứa batch nhỏ) ---
    print(f"\n[>>>] Đang xử lý luồng TRAIN...")
    features_dir = f"{args.output_dir}/{args.model}_features"
    labels_dir = f"{args.output_dir}/{args.model}_labels"

    infer_features_labels(dino, train_loader, features_dir, labels_dir, args.device, args)
    merge_npy(features_dir, labels_dir, {"feature": "features", "label": "labels"}, model_name, args.output_dir)

    # --- 2. XỬ LÝ TRÍCH XUẤT TẬP TEST (Sinh ra folder chứa batch nhỏ cho kiểm thử) ---
    print(f"\n[>>>] Đang xử lý luồng TEST...")
    features_dir = f"{args.output_dir}/{args.model}_test_features"
    labels_dir = f"{args.output_dir}/{args.model}_test_labels"

    infer_features_labels(dino, test_loader, features_dir, labels_dir, args.device, args)
    merge_npy(features_dir, labels_dir, {"feature": "test_features", "label": "test_labels"}, model_name, args.output_dir)

    print("\n[==>] Hoàn tất")