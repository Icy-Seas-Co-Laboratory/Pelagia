# Built-In Model Artifacts

Each model artifact directory should contain a `metadata.toml` file. Optional
payload files, such as `.keras`, SavedModel directories, ONNX files, or future
model packs, live beside that manifest.

Recommended layout:

```text
Pelagia/assets/models/
  roi_refinement/
    example_model/
      metadata.toml
      model.keras
```
