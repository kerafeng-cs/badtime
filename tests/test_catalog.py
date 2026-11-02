"""Danh mục và các registry cài đặt phải nhất quán: mục nào chạy được thì phải có cài đặt."""

from badtime_repro.catalog import CATALOG
from badtime_repro.data.sources import DATA_SOURCES
from badtime_repro.models import MODEL_FACTORIES
from badtime_repro.training.methods import TRAINING_METHODS


def test_runnable_entries_have_implementations():
    for name, spec in CATALOG.datasets.items():
        if spec.status.runnable:
            assert spec.source in DATA_SOURCES, name

    for name, spec in CATALOG.models.items():
        if spec.status.runnable:
            assert name in MODEL_FACTORIES, name

    for name, spec in CATALOG.methods.items():
        if spec.status.runnable:
            assert name in TRAINING_METHODS, name


def test_described_entries_are_not_registered():
    for name, spec in CATALOG.methods.items():
        if not spec.status.runnable:
            assert name not in TRAINING_METHODS, name

    for name, spec in CATALOG.models.items():
        if not spec.status.runnable:
            assert name not in MODEL_FACTORIES, name


def test_supported_slots_are_runnable():
    for dataset, model in CATALOG.supported_slots:
        assert CATALOG.datasets[dataset].status.runnable
        assert CATALOG.models[model].status.runnable
