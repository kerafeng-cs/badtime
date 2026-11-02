"""Ràng buộc kiến trúc, kiểm bằng phân tích tĩnh các câu import (không cần chạy mã).

1. Phân tầng: một module chỉ được import module cùng tầng hoặc tầng thấp hơn, kể cả import bên trong hàm.
2. Không torch: các module trong TORCH_FREE không được import torch ở cấp module (import trong hàm được phép,
   vì chỉ chạy khi gọi hàm đó), để lệnh `check` và phần lớn test chạy khi chưa cài torch.
"""

import ast
import os

PACKAGE = "badtime_repro"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), PACKAGE)

# Tầng thấp -> cao. Khóa là tiền tố tên module (sau "badtime_repro.").
LAYERS = [
    ("common", "assumptions"),                      # 0  tiện ích, dữ liệu tĩnh
    ("catalog", "config"),                          # 1  danh mục, schema cấu hình
    ("data",),                                      # 2  nguồn, phân chia, chuẩn hóa, cửa sổ
    ("attack",),                                    # 3  khối thiết kế BadTime
    ("metrics", "runs", "support"),                 # 4  hợp đồng phép đo, hiện vật lần chạy, kiểm tra hỗ trợ
    ("models",),                                    # 5  mô hình dự báo
    ("training",),                                  # 6  Trainer + TrainingMethod
    ("experiment",),                                # 7  mặt tiền ứng dụng
    ("evaluation",),                                # 8  đối chứng, tính lại, báo cáo
    ("cli", "__main__"),                            # 9  giao diện dòng lệnh
]

TORCH_FREE = (
    "common", "assumptions", "catalog", "config", "data", "metrics", "runs", "support", "experiment",
    "evaluation.report", "evaluation.recompute", "cli",
    "attack.budget", "attack.channels", "attack.injection", "attack.lag", "attack.patterns", "attack.plan",
    "attack.sample_selection", "models.__init__", "training.methods.__init__", "training.schedule",
)


def _layer_of(name):
    for i, prefixes in enumerate(LAYERS):
        for p in prefixes:
            if name == p or name.startswith(p + "."):
                return i

    raise AssertionError(f"Module '{name}' chưa được xếp tầng trong LAYERS.")


def _modules():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "tslib")]
        for f in filenames:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
                yield rel, path


def _imports(rel, path):
    """(tên module nội bộ được import, có phải import torch không) cho mọi câu import trong file."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    is_pkg = rel.endswith("__init__")
    here = rel.split(".")[:-1]                        # gói chứa module
    in_function = {id(n) for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in ast.walk(f)}
    internal, torch_used = [], False                  # torch_used: chỉ tính import ở cấp module (khi nạp module)
    for node in ast.walk(tree):
        top = id(node) not in in_function
        if isinstance(node, ast.Import):
            for alias in node.names:
                torch_used |= top and alias.name.split(".")[0] == "torch"
                if alias.name.startswith(PACKAGE + "."):
                    internal.append(alias.name[len(PACKAGE) + 1:])

        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                torch_used |= top and (node.module or "").split(".")[0] == "torch"
                if (node.module or "").startswith(PACKAGE + "."):
                    internal.append(node.module[len(PACKAGE) + 1:])
            else:
                base = here[: len(here) - (node.level - 1)] if not is_pkg or node.level > 1 else here
                target = ".".join(base + ([node.module] if node.module else []))
                if target:
                    internal.append(target)

    return internal, torch_used


def test_every_module_has_a_layer():
    for rel, _ in _modules():
        _layer_of(rel.replace(".__init__", "") if rel != "__init__" else "common")


def test_imports_respect_layers():
    violations = []
    for rel, path in _modules():
        if rel == "__init__":
            continue

        src_layer = _layer_of(rel.replace(".__init__", ""))
        for target in _imports(rel, path)[0]:
            if _layer_of(target) > src_layer:
                violations.append(f"{rel} (tầng {src_layer}) -> {target} (tầng {_layer_of(target)})")

    assert not violations, "Import ngược tầng:\n" + "\n".join(violations)


def test_torch_free_modules_do_not_import_torch():
    offenders = []
    for rel, path in _modules():
        name = rel.replace(".__init__", "") if rel.endswith(".__init__") and rel != "__init__" else rel
        declared = any(name == p or name.startswith(p + ".") or rel == p for p in TORCH_FREE)
        if declared and _imports(rel, path)[1]:
            offenders.append(rel)

    assert not offenders, f"Module khai báo không dùng torch nhưng import torch: {offenders}"
