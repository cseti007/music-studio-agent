# Changelog

Notable changes to music-mix-agent. Versions are by-commit; the project is not
yet semver-tagged.

The format groups changes by intent rather than by file, so a reader can see
why each batch happened. Newer entries on top.

---

## Unreleased — Polish and docs round

These items live on master but have not been tagged yet.

### Tests, docs, and final tuning

- **pytest smoke suite** (`tests/test_smoke.py`, 17 tests) — covers the
  load-bearing DSP and relevance-check logic:
  - True peak vs sample peak (4× oversampled ISP detection)
  - M/S encode/decode round-trip identity
  - Sub-sample phase alignment with known fractional delay
  - Pumping detector handles silent-gap signals (active-frame fix)
  - Per-band crest factor reflects transient vs sustained content
  - Each make-it-hit `relevance_check` makes the expected skip/apply call
- **`docs/knowledge.md`** — new sections "Make-it-hit Philosophy" and
  "Reading the New Analysis Metrics" sync the domain knowledge with the
  code: required-evidence thresholds per tool, process budget, re-analyze
  loop, and the rock-band-tracking + pumping-as-musical-pulse field-test
  lessons.
- **README.md** — workflow diagram expanded to include the make-it-hit tools
  and `mix_health`. New `tests/` directory documented.
- **`apply_haas`** — new gate: detects mono filenames matching stereo-pair
  patterns (` L.`, `_R.`, `.L.`, etc.) and refuses on them. Applying Haas
  to one half of an OH/room mic pair breaks the pair's stereo relationship.
- **`apply_exciter`** — new gate: refuses on low-frequency-dominant stems
  (kick, bass, sub sources) where the low band is 6+ dB louder than the
  high band AND spectral centroid is below 800 Hz. The old check (centroid
  < 4 kHz, air < -40 dBFS) was DSP-correct but musically wrong on bass/kick.
- **`requirements.txt`** — `pytest>=8.0` added under `# dev / test`.

### Pumping reinterpretation (a54cadd)

The pumping detector cannot distinguish a real comp artifact from a musical
strumming/groove pulse from envelope statistics alone — both produce
similar 1–5 Hz amplitude modulation. Raising the threshold would miss real
artifacts; instead, the project reframes what the flag means:

- `pumping_detected: true` is now documented as a **suspicion**, not a verdict.
- New CLAUDE.md subsection **"Interpreting pumping_detected"** with a 4-step
  disambiguation checklist: before/after the step? rate matches song tempo?
  what stem? depth vs excess profile?
- The make-it-hit re-analyze rule is clarified — only revert on a flag that
  *flipped* false→true after the step. A pre-existing flag is not the step's
  fault.
- `mix_health` stems-pumping verdict is now **YELLOW (review)**, not RED
  (broken), so musical-pulse false positives don't drown out real problems.

### Field-test tuning (ee78016)

After real-session test on `output/terido_v2`:

- **`apply_subharm`** gains a third gate: target band (where the new
  harmonics land) must not be more than 3 dB louder than the fundamental
  band. With typical preset settings the harmonics sit ~15 dB below the
  fundamental; if the target is already >3 dB above fundamental, the new
  content is mathematically too small to lift the band audibly. Reports
  `target_band_rms_dbfs` and `target_over_fundamental_db`.
- **`analyze._detect_pumping`** depth (p95/p5) is now computed on **active
  frames only** (RMS > -40 dBFS). The old code measured across the full
  envelope, so silent gaps between hits/sections pushed p5 to ~0 and the
  depth_db short-circuit returned 0.0, hiding real pumping on intermittent
  material. Reports `active_frame_ratio`.
- `mix_health._detect_pumping_quick` gets the same fix for consistency.

## Make-it-hit toolkit (c0eae35) — data-driven loudness/weight/width

Eight new make-it-hit features with built-in `relevance_check` gating, plus
the analysis metrics needed to power them and verify their results. The
defining property: every tool refuses to write audio when the input data
doesn't justify the processing (set `recommend_skip: true` with a list of
specific `issues`). Only `--force` overrides.

### New standalone processors
- **`apply_subharm`** — sub-bass harmonic synthesizer (missing-fundamental
  psychoacoustic for small-speaker translation)
- **`apply_haas`** — 5-25 ms channel delay stereo widener
- **`apply_exciter`** — Aphex-style HF harmonic generator
- **`apply_multiband_comp`** — 3-band Linkwitz-Riley LR4 split + per-band
  pedalboard.Compressor

### New `render_mix` chain stages (all guarded)
- `master.clipper {threshold_db, knee_db, mode}` — soft cubic / hard clip
  between glue comp and EQ
- `master.ms {mid_eq, side_eq, mid_gain_db, side_gain_db}` — independent
  mid/side EQ + gain
- `buses.*.parallel_saturation {mode, drive, mix}` — tube/tape/clipper-
  saturated blend on the drum bus

### New analysis fields
- `analyze.frequency_bands_crest_db.*` — per-band peak-to-RMS (sub / low /
  mid / high / air)
- `analyze.pumping.*` — 1-5 Hz envelope modulation detector for over-comp
  artifacts (rate, depth, lf_excess, active_frame_ratio)

### New session-level tool
- **`mix_health.py`** — scorecard for the rendered mix: LUFS, true peak,
  LRA, M/S width, low-freq mono compatibility, tonal balance vs reference,
  masking pair counts, stem pumping. Green/yellow/red verdicts.

### compare_reference --apply
- Auto inverse-delta peak EQ chain bake. Top 6 deltas by magnitude,
  each capped at ±6 dB so the chain stays phase-coherent.

### CLAUDE.md
- "Make-it-hit tools — DATA-GATED, NOT DEFAULT" table
- "Make-it-hit decision rules" with required-evidence thresholds, 4-step
  process budget, mandatory re-analyze loop, "don't chain blindly" rule
- "Analysis tool decision tree — when to run what" — trigger-based
  obligations table consolidating four previously scattered sections
- Ground rule: every `render_mix --render` MUST be followed by `mix_health`

## DSP foundations (87c5b6a)

Five sharpenings of the core DSP after a methodical code review against
2026 mixing best-practice references:

- **True peak**: `analyze` reports both `true_peak_dbfs` (4×-oversampled,
  ITU-R BS.1770-4 style) and `sample_peak_dbfs`. `render_mix` does a
  second-pass ISP measurement after pedalboard's Limiter and scales the
  master down if the oversampled true peak still exceeds the ceiling —
  so the configured -2 dBTP holds against streaming codec encoding.
- **Sub-sample phase alignment**: parabolic interpolation around the
  correlation peak + windowed-sinc fractional-delay FIR. Also fixes an
  inverted sign convention — the docstring said `delay_samples > 0` =
  "tgt arrived AFTER ref" but the code applied the opposite correction.
- **Minimum-phase EQ**: `apply_eq --phase {minimum, zero}` (default
  minimum). Insert EQ on tracks uses causal `sosfilt` — no pre-ringing
  on transients. Zero-phase remains opt-in for mastering / linear-phase.
- **Convolution reverb**: `apply_reverb --ir <wav>` switches the engine
  to `pedalboard.Convolution`. Freeverb stays the default for small spaces
  and plates; convolution gives authentic large halls.
- **Time-gated masking**: PSD computed only on active stem portions
  (100 ms-frame RMS gate at -45 dBFS), and pairs whose stems don't co-
  occur in time (Jaccard < 0.15) are suppressed. Kills false positives
  like "rhythm guitar (verses) vs lead vocal (choruses)".

## Cleanup and config consistency (d1ef151)

- `analyze` default target LUFS pulled from `config.toml [analyze]
  default_target_lufs` (default -18) — matches the rest of the pipeline.
  Previously hardcoded to -23 (broadcast-style).
- `apply_gain --per-channel` writes `<stem>_gain_report.json` (unique per
  input) instead of overwriting `gain_report.json`.
- `parse_session.py` honours `PTFTOOL_PATH` env var so the ptformat binary
  can live somewhere durable instead of `/tmp`.
- `apply_reverb`, `apply_amp`, `apply_transient` output filenames no
  longer include the preset name (`_reverb[_send].wav`, `_amp.wav`,
  `_transient.wav`). Preset lives in the report JSON.
- Extracted `tools/_stages.py` so `render_mix` and `detect_masking` share
  one source of truth for stage → stem-file resolution (was duplicated
  and slightly divergent).
- CLAUDE.md tools table synced with the actual `tracks/` subdir layout.

## Mixing pipeline genesis (f5af0c1)

The first commit that made the project usable from a fresh clone:

- 12 new processor tools (`align_phase`, `apply_eq`, `apply_compression`,
  `apply_gate`, `apply_reverb`, `apply_saturation`, `apply_transient`,
  `apply_amp`, `apply_delay`, `compare_reference`, `detect_masking`,
  `render_mix`)
- 46 instrument and processor presets
- `config.toml`, `README.md`
- `.gitignore` whitelist for `tools/presets/*.json` (the rule `*.json`
  used to silently hide every preset from git)
- Fix `render_mix._load_as_stereo` — previously dropped the right channel
  of stereo files (loaded only L and duplicated it)
- Remove obsolete `assemble_channel.py` (subsumed by `apply_gain --per-clip`)

## Project history before this changelog (cd890f2 and earlier)

- `parse_session.py` — Pro Tools (.ptx) and Ableton (.als) session parser
- `assemble_channel.py` — timeline-accurate channel assembler (later
  replaced by `apply_gain --per-clip`)
- `apply_gain.py` — per-clip and per-channel gain staging
- `analyze.py` — initial LUFS, transient, hum-detection analysis
- `CLAUDE.md`, `docs/knowledge.md` — first cuts of the agent system prompt
  and domain knowledge base
