import importlib


def test_import_quiche_root():
    quiche = importlib.import_module("quiche")
    assert hasattr(quiche, "pp")
    assert hasattr(quiche, "tl")


def test_import_quiche_submodules():
    importlib.import_module("quiche.preprocessing")
    importlib.import_module("quiche.tools")
    importlib.import_module("quiche.plotting")
