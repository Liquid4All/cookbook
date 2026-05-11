# Vendoring `preprocess_ohf_voice.py` and `train.py` from `liquid-audio-staging`

Voice-assistant depends on public `liquid-audio` v1.2.0+ for the runtime training stack (`Trainer`, `LFM2DataLoader`, data preprocessing pipeline, `LFM2AudioModel`, `LFM2AudioProcessor`), but vendors two example scripts from `liquid-audio-staging`'s `examples/audio-to-function-calling` branch directly into `voice-assistant/finetuning/`: `preprocess_ohf_voice.py` (OHF-Voice-specific iterator and dataset preprocessing) and `train.py` (Modal + W&B + val-split wrapper around the public trainer). Public `liquid-audio` ships `examples/preprocess_jenny_tts.py` and a bare-local `examples/train.py`, neither of which is sufficient: cookbook readers without private access to staging cannot otherwise run the full OHF-Voice pipeline. The eval script is **not** vendored: it is written fresh in `voice-assistant/evaluation/` to match the methodology in ADR-0001.

## Considered options

- **Wait for `liquid-audio` to publish the OHF-Voice example scripts.** Rejected: external release-schedule dependency, blocks the cookbook example with no committed ETA.
- **Fully vendor staging's `liquid_audio` package.** Rejected: the package's runtime classes are now public; only the example scripts are missing. Vendoring more than necessary doubles the maintenance surface.
- **Vendor `eval_ohf_voice.py` from staging.** Rejected: that script's methodology (single-seed sampling from `train`, only two metrics, no stratification) is exactly what we need to replace per ADR-0001.

## Consequences

- The vendored files carry a header comment recording source repo, branch, commit SHA, and copy date so future re-syncs are obvious.
- When `liquid-audio` publishes equivalents on `main`, voice-assistant deletes its copies and switches to importing from the public package.
