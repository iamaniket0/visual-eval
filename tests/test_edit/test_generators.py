"""Tests for the editor registry."""
from src.edit.editors import all_registered


def test_all_editors_registered():
    registered = all_registered()
    expected = {"flux_kontext", "bria_edit", "firefly", "photoroom", "picsart", "canva_leonardo"}
    assert expected.issubset(set(registered)), f"Expected {expected}, got {set(registered)}"


def test_registry_count():
    assert len(all_registered()) >= 6
