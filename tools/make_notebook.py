"""Sinh notebooks/badtime_poc_colab_kaggle.ipynb từ chính các file của dự án.

Chạy từ gốc dự án: python tools/make_notebook.py . notebooks/badtime_poc_colab_kaggle.ipynb
"""

import json
import os
import sys

ROOT = sys.argv[1]
OUT = sys.argv[2]

INCLUDE_DIRS = ("badtime_repro", "tests")
INCLUDE_FILES = ("configs/etth1_timesnet_96.json", "dataset/ETTh1/SOURCE.md", "pytest.ini")


def package_files():
    """Mọi file mã nguồn của dự án (thứ tự ổn định), trừ cache."""
    out = []
    for top in INCLUDE_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, top)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith((".py", ".json")) or name.startswith("LICENSE"):
                    out.append(os.path.relpath(os.path.join(dirpath, name), ROOT).replace(os.sep, "/"))

    return out + list(INCLUDE_FILES)


PACKAGE_FILES = package_files()
PACKAGE_DIRS = sorted({os.path.dirname(f) for f in PACKAGE_FILES if os.path.dirname(f)} | {"dataset/ETTh1"})

cells = []


def lines(text):
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])

    return out


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(text.strip("\n"))})


def code(text, collapsed=False):
    meta = {"jupyter": {"source_hidden": True}} if collapsed else {}
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": meta,
        "outputs": [], "source": lines(text.strip("\n")),
    })


md("""
# BadTime PoC: ETTh1 + TimesNet, 96 → 96 (Colab / Kaggle)

Notebook này chứa toàn bộ mã của `badtime_repro`. Nó tự ghi mã ra đĩa, tải ETTh1 (kiểm tra sha256), chạy kiểm tra tính đúng, huấn luyện **Benign (clean)** và **BadTime** với nhiều seed, chạy đối chứng 4 điều kiện, rồi lập báo cáo so với Bảng V của bài báo v2.

**Trước khi chạy**
- Colab: `Runtime → Change runtime type → T4 GPU`.
- Kaggle: `Settings → Accelerator → GPU T4/P100` và **bật Internet** (cần để tải ETTh1). Nếu không có Internet, thêm ETTh1.csv làm Kaggle Dataset; ô dữ liệu sẽ tự tìm trong `/kaggle/input`.
- Chạy lần lượt từ trên xuống (`Run all` cũng được). Ước lượng trên T4: kiểm tra chậm ~5–10 phút (cố ý chạy trên CPU), mỗi lần train ~3–8 phút.

**Lưu ý khi đọc kết quả.** Đây là bản tái lập độc lập, còn nhiều giả định (A01–A21, in ở cuối notebook). Một con số khác bài báo chưa đủ để kết luận bài báo sai; xem `docs/06-validation-plan.md`.
""")

md("## 1. Tham số của notebook")
code("""
# Seed cho mỗi nhánh. Đề xuất 3 lần lặp độc lập (docs/04).
SEEDS = [2021, 2022, 2023]

# Kiểm tra huấn luyện thật (CPU, mô hình nhỏ): đường xử lý sạch == BadTime đã tắt, tính lại chỉ số, đối chứng.
# Nên để True ở lần chạy đầu tiên; đây là cổng tính đúng trước khi xem số.
RUN_SLOW_TESTS = True

# Ghi đè cấu hình cho MỌI lệnh train/report, ví dụ ["attack.pattern.amplitude=2.0"].
# Đổi giả định thì nên đổi luôn runs_root để không trộn với lần chạy mặc định, ví dụ "runs_root=runs_amp2".
EXTRA_SET = []

# Mục 9 (tùy chọn): khảo sát độ nhạy với vài giả định chưa xác định.
RUN_SENSITIVITY = False
""")

md("## 2. Môi trường")
code("""
import os, sys, platform, subprocess

PACKAGE_DIRS = __PACKAGE_DIRS__

if os.path.isdir("/kaggle/working"):
    PLATFORM = "kaggle"
    PROJECT = "/kaggle/working/Badtime"
elif "google.colab" in sys.modules or os.path.isdir("/content"):
    PLATFORM = "colab"
    PROJECT = "/content/Badtime"
else:
    PLATFORM = "local"
    PROJECT = os.path.abspath("Badtime")

for d in PACKAGE_DIRS:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)

os.chdir(PROJECT)
print("platform:", PLATFORM, "| project:", PROJECT)

import torch, numpy, pandas
print("python", platform.python_version(), "| torch", torch.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    print("CẢNH BÁO: không có GPU, huấn luyện sẽ rất chậm.")

try:
    import pytest  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

SET_ARGS = " ".join(f"--set {s}" for s in EXTRA_SET)
SEEDS_ARG = " ".join(str(s) for s in SEEDS)
""".replace("__PACKAGE_DIRS__", repr(PACKAGE_DIRS)))

md("""
## 3. Ghi mã nguồn ra đĩa

Mỗi ô dưới đây là một file của dự án, giống hệt bản trong folder `Badtime` của bạn. TimesNet được vendored từ Time-Series-Library@`4e938a17` (MIT). Các ô đã thu gọn; mở ra để xem.
""")
for rel in PACKAGE_FILES:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        content = f.read()

    if not content.endswith("\n"):
        content += "\n"

    cells.append({
        "cell_type": "code", "execution_count": None,
        "metadata": {"jupyter": {"source_hidden": True}},
        "outputs": [], "source": lines(f"%%writefile {rel}\n" + content.rstrip("\n")),
    })

md("""
## 4. Dữ liệu ETTh1

Tải đúng commit `1d16c8f4` của zhouhaoyi/ETDataset và kiểm tra sha256. Không tự sửa dữ liệu. Nếu tải không được, ô này tìm `ETTh1.csv` trong `/kaggle/input`; nếu vẫn không có, hãy upload file vào `dataset/ETTh1/`.
""")
code("""
import glob, hashlib, shutil, urllib.request

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv"
ETTH1_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
dst = os.path.join(PROJECT, "dataset", "ETTh1", "ETTh1.csv")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

if not (os.path.isfile(dst) and sha256(dst) == ETTH1_SHA256):
    try:
        urllib.request.urlretrieve(ETTH1_URL, dst)
        print("Đã tải từ GitHub.")
    except Exception as exc:
        print("Không tải được:", exc)
        for cand in glob.glob("/kaggle/input/**/ETTh1.csv", recursive=True):
            if sha256(cand) == ETTH1_SHA256:
                shutil.copy(cand, dst)
                print("Đã chép từ", cand)
                break

assert os.path.isfile(dst), "Thiếu dataset/ETTh1/ETTh1.csv"
got = sha256(dst)
assert got == ETTH1_SHA256, f"sha256 không khớp: {got}"
print("ETTh1.csv OK", got)
""")
code("""
!python -m badtime_repro check
""")

md("Kiểm tra nhanh thông báo khi chọn tổ hợp ngoài PoC (mã thoát 3 = chưa hỗ trợ, 2 = không hợp lệ). Không có tổ hợp nào được tự chuyển sang ETTh1 + TimesNet.")
code("""
!python -m badtime_repro check --set dataset=ETTH2; echo "exit=$?"
!python -m badtime_repro check --set model=Autoformer; echo "exit=$?"
!python -m badtime_repro check --set pred_len=720; echo "exit=$?"
!python -m badtime_repro check --method backtime; echo "exit=$?"
!python -m badtime_repro check --set dataset=Foo; echo "exit=$?"
""")

md("""
## 5. Kiểm tra tính đúng

Nếu ô nào ở đây **fail**, dừng lại và gửi log; đừng diễn giải số ở các mục sau.
- Nhanh: các test numpy và torch không huấn luyện.
- Chậm (`RUN_SLOW_TESTS`): huấn luyện thật 2 epoch trên CPU với mô hình nhỏ. Kiểm tra (a) nhánh sạch và BadTime đã tắt (0% mẫu) cho cùng lịch sử huấn luyện; (b) tính lại chỉ số từ dự báo đã lưu khớp `metrics.json`; (c) trigger được cập nhật đúng lịch; (d) đối chứng 4 điều kiện chạy được.
""")
code("""
!python -m pytest -q tests -p no:cacheprovider
""")
code("""
if RUN_SLOW_TESTS:
    !BADTIME_SLOW=1 python -m pytest -q tests/test_torch_units.py tests/test_training.py -p no:cacheprovider -x
else:
    print("Bỏ qua kiểm tra chậm.")
""")

md("""
## 6. Huấn luyện

Mỗi lần chạy có thư mục riêng `runs/ETTh1_TimesNet_96_96/<method>/seed<seed>_<thời điểm>/` và không ghi đè lần khác. Log in MAE val theo epoch; BadTime in thêm MAE_pa / MAE_pn.
""")
code("""
!python -u -m badtime_repro train --method clean --seeds {SEEDS_ARG} {SET_ARGS}
""")
code("""
!python -u -m badtime_repro train --method badtime --seeds {SEEDS_ARG} {SET_ARGS}
""")

md("""
## 7. Đối chứng 4 điều kiện và báo cáo

Ghép từng seed: mô hình sạch seed *s* với trigger đã khóa của BadTime seed *s*. Benign MAE_pa / MAE_pn trong báo cáo lấy từ ô `clean_model|trigger_input`.
""")
code("""
import sys
sys.path.insert(0, PROJECT)

from badtime_repro.config import load_config
from badtime_repro.runs.store import RunIndex

CFG = load_config(os.path.join(PROJECT, "configs", "etth1_timesnet_96.json"), EXTRA_SET)
SLOT_DIR = os.path.join(PROJECT, CFG.runs_root, CFG.slot)
INDEX = RunIndex(SLOT_DIR)

for seed in SEEDS:
    clean_run, attack_run = INDEX.latest("clean", seed), INDEX.latest("badtime", seed)
    if clean_run is None or attack_run is None:
        print(f"seed {seed}: thiếu lần chạy hoàn thành (clean={clean_run is not None}, badtime={attack_run is not None})")
        continue

    clean_path, attack_path = clean_run.path, attack_run.path
    print(f"=== seed {seed}")
    !python -u -m badtime_repro controls --clean-run "{clean_path}" --attack-run "{attack_path}"
""")
code("""
from IPython.display import Markdown, display

!python -m badtime_repro report {SET_ARGS} > /dev/null
display(Markdown(open(os.path.join(SLOT_DIR, "report.md"), encoding="utf-8").read()))
""")

md("""
## 8. Xem nhanh một lần chạy BadTime

Kênh bị sửa và độ trễ mà GAT/Thuật toán 3 chọn, mức sửa đầu vào, và một cửa sổ test trên kênh đích: nhãn gốc, dự báo từ đầu vào sạch, dự báo từ đầu vào có trigger, cùng mục tiêu.
""")
code("""
import numpy as np
import matplotlib.pyplot as plt

run = INDEX.latest("badtime", SEEDS[0])
plan, met, preds = run.plan(), run.metrics(), run.predictions("test")
print("lần chạy:", run.path)
print("kênh đích:", [plan.channels[i] for i in plan.target_idx], "| kênh bị sửa:", [plan.channels[i] for i in plan.poison_idx])
print("đoạn:", plan.seg_lens, "| độ trễ:", plan.lags, "| vị trí trong đầu vào [s, e):",
      [(o, o + l) for o, l in zip(plan.offsets, plan.seg_lens)])
print("ảnh hưởng GAT:", dict(zip(plan.channels, np.round(plan.trace["influence"], 4))))
print("số cửa sổ bị chọn:", len(plan.poison_starts), "| ô sửa mỗi cửa sổ:", plan.trigger_len)
print("chọn epoch:", met["selection"])
print("mức sửa đầu vào (test):", met["input_change_test"])

t = plan.target_idx[0]
for k in (0, len(preds["y"]) // 2):
    plt.figure(figsize=(9, 3))
    plt.plot(preds["y"][k, :, t], label="nhãn gốc")
    plt.plot(preds["pred_clean"][k, :, t], label="dự báo, đầu vào sạch")
    plt.plot(preds["pred_trig"][k, :, t], label="dự báo, có trigger")
    plt.plot(range(plan.t_pin), plan.pattern[:, 0], "k--", label="mục tiêu")
    plt.title(f"test window {k} (s={int(preds['starts'][k])}), kênh {plan.channels[t]}, thang chuẩn hóa")
    plt.legend(fontsize=8)
    plt.show()
""")

md("""
## 9. (Tùy chọn) Độ nhạy với giả định chưa xác định

Mỗi biến thể dùng `runs_root` riêng nên không trộn với lần chạy mặc định, và mỗi lần chỉ đổi một giả định: A06 (dạng và biên độ mục tiêu), A08 (cách chia ngân sách), A13 (phép tổng hợp loss). Chỉ train lại BadTime, vì các giả định này không ảnh hưởng nhánh sạch (clean dùng loss MSE). Chi phí: số seed × số biến thể lần train BadTime. Báo cáo của mỗi biến thể vì vậy không có dòng Benign.
""")
code("""
VARIANTS = {
    "amp2":        ["attack.pattern.amplitude=2.0"],
    "hump_sine":   ["attack.pattern.shape=hump_sine"],
    "equal_seg":   ["attack.budget_protocol=equal_segments"],
    "loss_mean":   ["attack.loss_reduction=mean"],
}

if RUN_SENSITIVITY:
    for name, sets in VARIANTS.items():
        args = " ".join(f"--set {s}" for s in sets + [f"runs_root=runs_sens/{name}"])
        print(f"##### {name}")
        !python -u -m badtime_repro train --method badtime --seeds {SEEDS_ARG} {args}
        !python -m badtime_repro report {args}
else:
    print("RUN_SENSITIVITY = False")
""")

md("## 10. Lưu kết quả")
code("""
import shutil

archive = shutil.make_archive(os.path.join(os.path.dirname(PROJECT), "badtime_runs"), "zip", PROJECT,
                              base_dir=".")
print("Đã nén:", archive, f"({os.path.getsize(archive) / 2**20:.1f} MB)")

if PLATFORM == "colab":
    from google.colab import files
    files.download(archive)
    # Hoặc lưu vào Drive:
    # from google.colab import drive; drive.mount("/content/drive"); shutil.copy(archive, "/content/drive/MyDrive/")
elif PLATFORM == "kaggle":
    print("Kaggle: file trong /kaggle/working được giữ ở tab Output sau khi Save Version.")
""")

md("## Phụ lục: sổ giả định đang áp dụng")
code("""
from badtime_repro.assumptions import ASSUMPTIONS
for a in ASSUMPTIONS:
    print(f"{a['id']} [{a['status']}] {a['topic']}: {a['choice']}")
    if a.get("note"):
        print("      ghi chú:", a["note"])
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("cells:", len(cells), "->", OUT)
