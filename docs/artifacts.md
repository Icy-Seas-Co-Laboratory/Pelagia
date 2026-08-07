# Artifact Organization

Pelagia distinguishes packaged assets from local runtime artifacts.

## Packaged Assets

Files under `Pelagia/assets/` ship with the Python package. Use this location
for small schemas and built-in plugin placeholders. ML model products are owned
and served by Oracle Builder and must not be bundled into Pelagia.

```text
Pelagia/assets/
  plugins/
    plugin_name/
      metadata.toml
  schemas/
```

Every plugin artifact directory should include `metadata.toml`.

## Local Artifact Library

Files imported or created after installation should live in the configured local
library. The defaults are:

```text
.pelagia/
  plugins/
```

These paths can be changed in `config.toml`:

```toml
[artifacts]
local_root = "./.pelagia"

[artifacts.plugins]
local_path = "./.pelagia/plugins"
```

## Plugin Metadata

Plugin support is manifest-only for now. Pelagia can discover plugin manifests,
but it does not import or execute plugin code yet.

```toml
name = "example_plugin"
kind = "plugin"
version = "0.1.0"
description = "Example plugin manifest."

[plugin]
entrypoint = "example_plugin:register"
capabilities = ["export"]
```

The future plugin system can build on this manifest layout without changing
where files live.

## Model references

`GET /roi-refinement/options` proxies Oracle Builder's model catalog and exposes
operational aliases plus immutable artifact identity. Pelagia stores the
resolved artifact/run IDs and fingerprint on refined detections; it does not
interpret model directories or framework files. See [Oracle Builder
inference](oracle-builder.md).
