# %%writefile /kaggle/working/VB-CGCD-main/dataloaders/neudet.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026/05/29
# @Author  : Truong An Khang

from utils import dataloader
from utils.dataloader import ClassIncrementalLoader


class NEU_DETLoader():
    def __init__(self, args):
        self.args = args

    def makeT5Loader(self):
        # Bộ NEU có 6 lớp: 
        # Base Stage (S0) chiếm 50% số lớp = 3 lớp
        base = 3 
        # 3 lớp mới còn lại chia đều cho 3 online stages tiếp theo (mỗi stage tăng 1 lớp)
        increment = 1 

        # Cấu hình số lượng mẫu thực tế dựa trên phân bổ dữ liệu lỗi công nghiệp:
        # Lớp base có 3 lớp * ~240 ảnh/lớp = 720 ảnh tối đa -> chọn num_labeled phù hợp
        num_labeled = 700          # Số mẫu có nhãn ở pha Offline S0
        num_novel_inc = 180        # Số mẫu lỗi mới xuất hiện ở mỗi online stage
        num_known_inc = 40         # Số mẫu đệm lỗi cũ đưa vào từng stage để test chống quên

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
        """
        Vì bộ NEU chỉ có tối đa 6 lớp, thiết lập 10 sessions (T10) không khả thi 
        do không đủ số lớp để phân rã. Ta giữ nguyên cấu trúc mỏ neo T5 để tránh lỗi hệ thống.
        """
        return self.makeT5Loader()

    def makeVinLoader(self):
        """
        Kịch bản label-limited cực hạn (Vin): Chỉ có 1 lớp lỗi làm Base Stage (S0).
        5 lớp lỗi mới còn lại sẽ tự động tăng trưởng trực tuyến qua các giai đoạn sau.
        """
        base = 1
        increment = 1
        num_labeled = 150          # Chỉ gán nhãn 1 lớp lỗi ban đầu
        num_novel_inc = 180
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