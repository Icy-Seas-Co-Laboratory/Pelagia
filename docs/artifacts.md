# Artifact Organization

Pelagia distinguishes packaged assets from local runtime artifacts.

## Packaged Assets

Files under `Pelagia/assets/` ship with the Python package. Use this location
for built-in manifests, small schemas, built-in plugin placeholders, and bundled
model artifacts that should be available immediately after installation.

```text
Pelagia/assets/
  models/
    roi_refinement/
      model_name/
        metadata.toml
        model.keras
  plugins/
    plugin_name/
      metadata.toml
  schemas/
```

Every model or plugin artifact directory should include `metadata.toml`.

## Local Artifact Library

Files imported or created after installation should live in the configured local
library. The defaults are:

```text
.pelagia/
  models/
  plugins/
```

These paths can be changed in `config.toml`:

```toml
[artifacts]
local_root = "./.pelagia"

[artifacts.models]
local_path = "./.pelagia/models"

[artifacts.plugins]
local_path = "./.pelagia/plugins"
```

## Model Metadata

Recommended `metadata.toml`:

```toml
name = "oracle_unet_v1"
kind = "roi_refinement"
version = "0.1.0"
description = "Bundled U-Net ROI mask refinement model."

[artifact]
framework = "keras"
format = "keras"
path = "model.keras"

[io]
input_shape = [256, 256, 2]
output_shape = [256, 256, 1]
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

## Model References

Discovered artifacts receive stable references:

```text
builtin:model/roi_refinement/example_model
local:model/roi_refinement/my_model
```

ROI refinement can use these references directly:

```toml
[processing.roi_refinement]
enabled = true
model_kind = "keras_artifact"
model_ref = "builtin:model/roi_refinement/example_model"
```

The API exposes available ROI refinement model references through
`GET /roi-refinement/options` and `GET /system/capabilities`.
