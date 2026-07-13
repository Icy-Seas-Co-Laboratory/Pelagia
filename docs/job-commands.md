# Versioned Job Commands

Pelagia queues pipeline work with versioned, typed command payloads. The payload remains a flat JSON object so existing workers and tooling can read its operational fields directly.

## Contract

Commands for `extract_frames`, `preprocess_frames`, `segment`, `background_frames`, and `roi_refinement` include:

```json
{
  "command_type": "segment_frames",
  "command_version": 1
}
```

The remaining fields are unchanged from the previous flat payloads. For example, a segment job still includes `frame_ids`, `start_frame`, `end_frame`, `limit`, and its resolved segmentation options.

## Compatibility

- Workers accept legacy payloads without command metadata and upgrade them in memory.
- New API, CLI, and `JobService` submissions add the version fields automatically.
- A payload declaring a mismatched `command_type`, or an unsupported command version, is rejected before work begins.
- Unknown fields are retained for legacy extensions.

External producers should add the command metadata now. During the compatibility period, legacy flat payloads continue to work, but a future major release can require `command_type` and `command_version`.
