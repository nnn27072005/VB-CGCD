#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Truong An Khang

from utils import dataloader
from utils.dataloader import ClassIncrementalLoader


class GC10_DETLoader():
    def __init__(self, args):
        self.args = args

    def makeT5Loader(self):
        # Bộ GC10-DET có 10 lớp:
        # Giai đoạn gốc Offline (S0) chiếm 50% số lớp = 5 lớp
        base = 5 
        # 5 lớp mới còn lại chia đều cho 5 online stages tiếp theo (mỗi stage nạp 1 lớp)
        increment = 1 
        
        # Cấu hình số lượng mẫu thực tế dựa trên quy mô ~350 ảnh/lớp của GC10-DET:
        num_labeled = 1000          # Số mẫu có nhãn ở pha Offline S0
        num_novel_inc = 250        # Số mẫu lỗi mới xuất hiện ở mỗi online stage
        num_known_inc = 40         # Số mẫu đệm lỗi cũ đưa vào để test chống quên

        loader = dataloader.StrictPerClassIncrementalLoader(
            data_dir=self.args.data_dir, 
            pretrained_model_name=self.args.pretrained_model_name, 
            base=base, 
            increment=increment, 
            num_labeled=num_labeled, 
            num_novel_inc=num_novel_inc, 
            num_known_inc=num_known_inc
        )

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader

    def makeT10Loader(self):
        # Vì bộ dữ liệu có đúng 10 lớp, ta có thể chạy kịch bản T10 cực hạn:
        # Học trước 2 lớp làm Base, 8 lớp còn lại tăng trưởng qua 8 giai đoạn online liên tục
        base = 2
        increment = 1
        num_labeled = 400
        num_novel_inc = 250
        num_known_inc = 20

        loader = dataloader.StrictPerClassIncrementalLoader(
            data_dir=self.args.data_dir, 
            pretrained_model_name=self.args.pretrained_model_name, 
            base=base, 
            increment=increment, 
            num_labeled=num_labeled, 
            num_novel_inc=num_novel_inc, 
            num_known_inc=num_known_inc
        )

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader