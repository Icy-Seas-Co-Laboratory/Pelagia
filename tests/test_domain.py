from Pelagia.domain import DEFAULT_COLLECTION, RawAssetManifest, AssetKind, normalize_collections


def test_normalize_collections_defaults_to_none():
    assert normalize_collections(None) == [DEFAULT_COLLECTION]
    assert normalize_collections("") == [DEFAULT_COLLECTION]
    assert normalize_collections(" , ") == [DEFAULT_COLLECTION]


def test_normalize_collections_accepts_comma_separated_values():
    assert normalize_collections("skq202510S-T1, test, transect1, test") == [
        "skq202510S-T1",
        "test",
        "transect1",
    ]


def test_raw_asset_manifest_normalizes_collections():
    asset = RawAssetManifest(
        asset_id="asset-1",
        asset_key="sample.avi",
        path="/tmp/sample.avi",
        kind=AssetKind.VIDEO,
        size_bytes=10,
        checksum="sha256:test",
        collections=[" test ", "", "transect1"],
    )

    assert asset.collections == ["test", "transect1"]
