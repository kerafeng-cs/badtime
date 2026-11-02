"""Giao diện dòng lệnh. Chạy từ gốc dự án:

    python -m badtime_repro check   [--config FILE] [--set KHOA=GIATRI ...] [--method M]
    python -m badtime_repro train   --method {clean,badtime} [--seeds S ...] [--config FILE] [--set ...]
    python -m badtime_repro controls --clean-run DIR --attack-run DIR
    python -m badtime_repro report  [--config FILE] [--set ...]
    python -m badtime_repro recompute --run DIR

Mã thoát: 0 thành công, 1 lỗi khi chạy, 2 cấu hình không hợp lệ, 3 chưa hỗ trợ, 4 dữ liệu thiếu/không hợp lệ.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from .common.errors import BadTimeError, ConfigInvalidError
from .config.loader import load_config
from .config.schema import ExperimentConfig
from .evaluation.recompute import MetricRecomputer
from .evaluation.report import MarkdownReportRenderer, ReportBuilder
from .experiment import Experiment
from .support import SupportChecker, SupportVerdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "etth1_timesnet_96.json")


class Command(ABC):
    name: str = ""
    help: str = ""
    uses_config: bool = True

    def __init__(self, project_root: str):
        self.project_root = project_root

    def configure(self, parser: argparse.ArgumentParser) -> None:
        if self.uses_config:
            parser.add_argument("--config", default=DEFAULT_CONFIG)
            parser.add_argument("--set", action="append", default=[], metavar="KHOA=GIATRI")

    def load(self, args: argparse.Namespace) -> ExperimentConfig:
        return load_config(args.config, args.set)

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        ...


class CheckCommand(Command):
    name = "check"
    help = "Kiểm tra tổ hợp + dữ liệu, không huấn luyện"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--method", default=None)

    def run(self, args: argparse.Namespace) -> int:
        cfg = self.load(args)
        if args.method:
            cfg = cfg.replace(method=args.method)

        cfg, dataset = Experiment(self.project_root).resolve(cfg)
        r = dataset.report
        print(f"Tổ hợp hợp lệ: {cfg.dataset} + {cfg.model} + {cfg.method}, {cfg.seq_len} -> {cfg.pred_len}")
        print(f"Dữ liệu: {r.path}")
        print(f"  sha256 {r.sha256}")
        print(f"  {r.n_rows} hàng, {r.start} -> {r.end}, kênh {r.channels}")
        print(f"  thiếu {r.n_missing}, trùng mốc {r.n_duplicate_timestamps}, bước không đều {r.n_irregular_steps}")
        for name, (s, e) in dataset.bounds.items():
            st = dataset.starts[name]
            print(f"  {name:5s} hàng [{s}, {e})  {len(st)} cửa sổ, s in [{st[0]}, {st[-1]}]")

        return 0


class TrainCommand(Command):
    name = "train"
    help = "Huấn luyện một nhánh với một hoặc nhiều seed"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--method", required=True)
        parser.add_argument("--seeds", type=int, nargs="+", default=[2021])

    def run(self, args: argparse.Namespace) -> int:
        cfg = self.load(args)
        experiment = Experiment(self.project_root)
        for seed in args.seeds:
            experiment.train(cfg, args.method, seed)

        return 0


class ControlsCommand(Command):
    name = "controls"
    help = "Đối chứng bốn điều kiện mô hình x đầu vào"
    uses_config = False

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--clean-run", required=True)
        parser.add_argument("--attack-run", required=True)

    def run(self, args: argparse.Namespace) -> int:
        from .evaluation.controls import ControlExperiment

        ControlExperiment(self.project_root).run(args.clean_run, args.attack_run)
        return 0


class ReportCommand(Command):
    name = "report"
    help = "Tổng hợp các lần chạy so với bài báo"

    def run(self, args: argparse.Namespace) -> int:
        check = SupportChecker().check(self.load(args))
        if check.verdict is SupportVerdict.INVALID:
            raise ConfigInvalidError(check.messages)

        cfg = check.config
        builder = ReportBuilder(os.path.join(self.project_root, cfg.runs_root, cfg.slot), cfg.slot)
        report = builder.build()
        builder.write(report)
        print(MarkdownReportRenderer().render(report))
        return 0


class RecomputeCommand(Command):
    name = "recompute"
    help = "Tính lại chỉ số từ dự báo đã lưu"
    uses_config = False

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--run", required=True)

    def run(self, args: argparse.Namespace) -> int:
        result = MetricRecomputer().run(args.run)
        print(result)
        return 0 if all(v["ok"] for v in result.values()) else 1


COMMANDS = (CheckCommand, TrainCommand, ControlsCommand, ReportCommand, RecomputeCommand)


class Application:
    def __init__(self, project_root: str = PROJECT_ROOT):
        self.commands: Dict[str, Command] = {cls.name: cls(project_root) for cls in COMMANDS}
        self.parser = argparse.ArgumentParser(prog="badtime_repro")
        sub = self.parser.add_subparsers(dest="cmd", required=True)
        for name, command in self.commands.items():
            command.configure(sub.add_parser(name, help=command.help))

    def main(self, argv: Optional[Sequence[str]] = None) -> int:
        _configure_io()
        args = self.parser.parse_args(argv)
        try:
            return self.commands[args.cmd].run(args)
        except BadTimeError as err:
            for message in err.messages:
                print(message, file=sys.stderr)

            return err.exit_code


def _configure_io() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    root = logging.getLogger("badtime_repro")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False


def main(argv: Optional[List[str]] = None) -> int:
    return Application().main(argv)
