import re
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn tới log của model thứ 2
# Bạn sửa lại đúng tên file log của bạn
log_files = {
    "CIFAR-100": "logs/c100.txt",
    "Tiny ImageNet": "logs/tiny.txt",
    "ImageNet100": "logs/in100.txt",
    "CUB-200": "logs/cub.txt",
}

def extract_novel_all_counts(log_path):
    """
    Extract lines like:
    Number of Novel Samples: 3976 / 5250
    """
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pattern = r"Number of Novel Samples:\s*(\d+)\s*/\s*(\d+)"
    matches = re.findall(pattern, text)

    novel_counts = [int(x[0]) for x in matches]
    all_counts = [int(x[1]) for x in matches]

    return novel_counts, all_counts


fig, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.flatten()

for ax, (dataset_name, log_path) in zip(axes, log_files.items()):
    novel_counts, all_counts = extract_novel_all_counts(log_path)

    sessions = np.arange(1, len(novel_counts) + 1)
    width = 0.38

    ax.bar(sessions - width / 2, novel_counts, width, label="Novel")
    ax.bar(sessions + width / 2, all_counts, width, label="All")

    # Đường đỏ: số mẫu novel trung bình hoặc kỳ vọng
    # Nếu muốn dùng trung bình từ log:
    expected_novel = np.mean(novel_counts)
    ax.axhline(expected_novel, linestyle="--", linewidth=1.5, color = "red")

    ax.set_title(dataset_name)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(sessions)
    ax.tick_params(axis="both", direction="in", top=True, right=True)

# Chỉ hiện legend ở subplot cuối giống hình mẫu
axes[-1].legend(title="Type", loc="upper right")

plt.tight_layout()
plt.savefig("novel_all_samples_model2.png", dpi=300, bbox_inches="tight")
plt.show()