#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Truong An Khang

import argparse
import os

import numpy as np
import torch
import torchvision


class SafeImageFolder(torchvision.datasets.ImageFolder):
        def find_classes(self, dir: str):
            classes, class_to_idx = super().find_classes(dir)
            # Nếu thấy folder 'lable' thì chủ động xóa tên khỏi danh sách lớp
            if 'lable' in classes:
                classes.remove('lable')
                if 'lable' in class_to_idx:
                    del class_to_idx['lable']
            return classes, class_to_idx

def infer_features_labels(dino, data_loader, features_dir, labels_dir, device):
    dino.to(device)
    dino.eval()

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Đọc batch dạng tuple (images, labels) chuẩn PyTorch DataLoader
    for bidx, (images, labels) in enumerate(data_loader):
        images = images.to(device)

        with torch.no_grad():
            features = dino(images)

        np.save(f"{features_dir}/features_{bidx}", features.cpu().data)
        np.save(f"{labels_dir}/labels_{bidx}", labels.cpu().data)


def merge_npy(features_dir, labels_dir, prefix, model_name, output_dir):
    # Lấy danh sách các file batch đã lưu và sắp xếp theo thứ tự
    feature_files = sorted([os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.npy')])

    assert len(feature_files) == len(label_files), "Mismatch in number of feature and label files"

    def merged_array(files):
        arrays = [np.load(f) for f in files]
        return np.concatenate(arrays, axis=0)

    target_dir = f"{output_dir}/gc10-det"
    os.makedirs(target_dir, exist_ok=True)
    
    # Lưu file gộp cuối cùng theo đúng định dạng tên mô hình
    np.save(f"{target_dir}/{prefix['feature']}-{model_name}.npy", merged_array(feature_files))
    np.save(f"{target_dir}/{prefix['label']}-{model_name}.npy", merged_array(label_files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINO Inference on GC10-DET')

    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str,
        help='Device on which to run (cuda, mps, cpu)'
    )
    parser.add_argument('--data_dir', default='GC10-DET', type=str, help='Path to GC10-DET root folder')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument("--model", default="dino_vitb16", type=str, help="Model name")
    parser.add_argument("--output_dir", default="output-features", type=str, help="Output directory")
    parser.add_argument("--finetuned", action='store_true', help='Finetuned model')
    parser.add_argument("--labeled_classes", default=50, type=int, help="Number of labeled classes")

    args = parser.parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Cấu hình bộ tiền xử lý ảnh chuẩn kiến trúc mạng ViT (Bicubic Resize & Normalization)
    interpolation = 3  # Bicubic
    crop_pct = 0.875
    image_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(image_size / crop_pct), interpolation),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    # 1. Đọc toàn bộ dữ liệu ảnh từ cấu hình thư mục (1 -> 10)
    # print(f"[+] Đang nạp dữ liệu từ thư mục: {args.data_dir}")
    # full_dataset = torchvision.datasets.ImageFolder(root=args.data_dir, transform=train_transforms)
    # Định nghĩa một lớp bọc nhỏ để chủ động lọc bỏ folder 'lable'
    

    print(f"[+] Đang nạp dữ liệu từ thư mục: {args.data_dir}")
    # Đổi hàm ImageFolder gốc thành SafeImageFolder vừa tạo
    full_dataset = SafeImageFolder(root=args.data_dir, transform=train_transforms)
    
    # 2. Thực hiện phân chia tỷ lệ ngẫu nhiên cố định bằng Seed (80% Train, 20% Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    # 3. Tạo DataLoaders cho hai tập
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Tải kiến trúc mạng tự giám sát DINO tương ứng
    print(f"[+] Đang tải mô hình backbone: {args.model}")
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

    model_name = args.model.replace("_", "-")

    # --- TIẾN HÀNH TRÍCH XUẤT TẬP TRAIN ---
    # --- TIẾN HÀNH TRÍCH XUẤT TẬP TRAIN ---
    print("[->] Đang trích xuất đặc trưng tập TRAIN...")
    # SỬA Ở ĐÂY: Ép các thư mục batch tạm chui hết vào trong folder gc10-det
    features_dir = f"{args.output_dir}/gc10-det/{args.model}_features"
    labels_dir = f"{args.output_dir}/gc10-det/{args.model}_labels"
    
    infer_features_labels(dino, train_loader, features_dir, labels_dir, args.device)
    merge_npy(features_dir, labels_dir, {"feature": "features", "label": "labels"}, model_name, f"{args.output_dir}/gc10-det")

    # --- TIẾN HÀNH TRÍCH XUẤT TẬP TEST ---
    print("[->] Đang trích xuất đặc trưng tập TEST...")
    # SỬA Ở ĐÂY: Ép các thư mục batch tạm chui hết vào trong folder gc10-det
    features_dir = f"{args.output_dir}/gc10-det/{args.model}_test_features"
    labels_dir = f"{args.output_dir}/gc10-det/{args.model}_test_labels"
    
    infer_features_labels(dino, test_loader, features_dir, labels_dir, args.device)
    # merge_npy(features_dir, labels_dir, {"feature": "test_features", "label": "test_labels"}, model_name, f"{args.output_dir}/gc10-det")
    merge_npy(features_dir, labels_dir, {"feature": "test_features", "label": "test_labels"}, model_name, args.output_dir)


    print(f"[===] HOÀN THÀNH BIÊN DỊCH! Các file .npy đã sẵn sàng tại: {args.output_dir}/gc10-det/")