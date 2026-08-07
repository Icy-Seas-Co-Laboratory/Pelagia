from __future__ import annotations

from Pelagia.services.taxonomy import DEFAULT_TAXONOMY_FILENAME, default_taxonomy_dictionary


def test_packaged_default_taxonomy_is_a_valid_label_dictionary():
    dictionary = default_taxonomy_dictionary()
    labels = dictionary["labels"]

    assert dictionary["filename"] == DEFAULT_TAXONOMY_FILENAME
    assert dictionary["key"] == "pelagia-core@0.1.1"
    assert dictionary["vocabulary"]["name"] == "Pelagia Core Vocabulary"
    assert dictionary["selectable_count"] == sum(
        label.get("selectable", True) is not False for label in labels
    )
    assert dictionary["selectable_count"] > 0
    assert len({label["id"] for label in labels}) == len(labels)


def test_default_taxonomy_returns_an_independent_copy():
    first = default_taxonomy_dictionary()
    first["labels"][0]["name"] = "changed"

    assert default_taxonomy_dictionary()["labels"][0]["name"] != "changed"
