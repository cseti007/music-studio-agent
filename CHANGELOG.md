# Changelog

Notable changes to music-studio-agent. Versions are by-commit; the project is not
yet semver-tagged.

The format groups changes by intent rather than by file, so a reader can see
why each batch happened. Newer entries on top.

---

## Unreleased — Polish and docs round

These items live on master but have not been tagged yet.

### Style-aware mix evaluation (2ab42d9)

Reference-free, genre-aware mix grading.

- **5 built-in style profiles** in `tools/style_profiles/`: `modern_rock`,
  `classic_rock`, `pop`, `hip_hop`, `jazz_acoustic`. Each profile fixes
  LUFS / LRA / crest-factor / 5-band tonal-balance targets with per-target
  tolerance. Numbers are wideband band-RMS measured at the profile's LUFS
  target (NOT iZotope-TBC PSD-curve values) — calibrated against real
  terido session masters and adjusted for documented genre-specific
  spectral shifts.
- **`tools/style_check.py mix.wav --style NAME`** measures the mix
  (LUFS-normalised to the profile's target before spectral analysis) and
  emits a 0–100 score, GREEN/YELLOW/RED verdict, per-check delta with
  ASCII bar chart, and EQ recommendations for off-target bands. Hard-fail
  rule: a RED on LUFS or LRA forces overall RED.
- 5 new pytest cases in `TestStyleCheck` (profile schema, grade thresholds,
  hard-fail rule, all-green score).

### mix_chain.json recall sheets (7abeb77)

End-to-end session reproducibility.

- **`tools/build_chain.py output/<session>`** aggregates every per-stem
  `*_report.json` into a single `mix_chain.json` listing the ordered
  processing chain with exact parameters per stem. Non-invasive — only
  reads existing reports, no apply_*.py changes. Topo sort is
  filename-based (not full-path), so historical path bugs (e.g. an
  accidentally-nested "stem/stem/file.wav") don't break ordering.
- **`tools/replay_chain.py <chain>`** re-runs every step via subprocess
  against the canonical CLI tools, then `render_mix --render --stems`.
  Default behaviour is overwrite-in-place; `--dry-run` prints commands
  without executing; `--stem NAME` replays a single stem for debugging.
- Validated on real terido session: 64 stems → 198 chain steps,
  dry-run argv generates correctly for every step type.
- 2 new pytest cases in `TestChainRecall`.

### Analysis content expansion + parallel batch (bac5b84..598e8dc)

The serial pattern `for stem in *; do analyze.py "$stem"; done` was
slow (library imports re-paid per stem, single-core CPU use) and the
per-stem `analysis.json` lacked time-series and tonal-context data.

- **`tools/batch_analyze.py output/<session> [--workers N]`** —
  multiprocessing pool wrapper around `analyze`. Default workers
  `min(cpu_count, 8)`. **5–6× speed-up** on a 64-stem batch
  (~12.8 min serial → ~2 min 22 s with 8 workers on the terido
  session). `--skip-existing` to resume; `--files` for an explicit
  WAV list.
- **Four new analysis field groups in `analysis.json`** (also surfaced
  in `spectrogram.txt` STATS SUMMARY for tempo + key):
  - `onsets_sec[]` — onset times in seconds
  - `tempo_bpm` — librosa beat-track estimate (None for short / unstable)
  - `estimated_key` — `{key, mode, confidence}` via Krumhansl-Schmuckler
    on `chroma_stft` (5–10× faster than chroma_cqt, accurate enough for
    rock)
  - `envelopes.rms_db_per_second[]`, `envelopes.lufs_short_term[]`
    (BS.1770 short-term, 3 s window, 1 s step), `envelopes.spectral_flux_per_second[]`
- Optimisations to stay within budget: 1 s LUFS-short-term step
  (was 0.1 s — 10× fewer meter calls), `chroma_stft` instead of
  `chroma_cqt`, shared `onset_env` between `_transient_density` and
  `_spectral_flux_per_sec`. Final overhead vs. the no-extra-fields
  baseline: +37 % wall-clock (acceptable, still ~3.9× faster than
  pre-batch serial).
- **`spectrogram_text` field dropped from `analysis.json`** (it
  duplicated `spectrogram.txt`). The .txt file is the canonical
  human-readable view; the JSON now holds structured data only.

### parse_session: NFC/NFD Unicode normalisation in audio-file resolution

`parse_session.py _resolve_audio()` now matches a clip's `source_file`
to disk under both NFC and NFD Unicode forms. Pro Tools ptftool emits
filenames NFC-normalised on Linux output, while audio-folder filenames
imported from macOS-originated drives can be NFD-normalised — same
visual character ("ő", "ű", "á"), different code-point sequence. The
previous code-point-equality match silently failed on these, leaving
the clip's `source_file` unresolved. Now both forms resolve correctly.

### Documentation polish (ebfdfc7)

- Documented why analysis files live in three places (per-stem in
  `tracks/<name>/`, session-wide in `analysis/`, master-level in
  `masters/`) — intentional co-location with the asset each report
  describes, not clutter.
- BACKLOG cleanup: shipped `style profiles` and `mix_chain.json`
  entries moved to "Completed since this file was created".

### Reverb extensions: BPM-synced pre-delay, sidechain ducking, built-in IR pack

Three non-vocal-specific reverb features that don't require the
deferred vocal-toolkit work:

- **BPM-synced pre-delay.** `apply_reverb --bpm 184
  --pre-delay-division sixteenth` computes the pre-delay from
  tempo + note division (12 named divisions including dotted and
  triplet variants). 184 BPM sixteenth = 81.5 ms; 120 BPM eighth =
  250 ms. Replaces explicit `--pre-delay MS` when given.
- **Sidechain ducking on the reverb tail** ("pumping reverb").
  `apply_reverb --sidechain kick.wav --sc-depth -12 --sc-hp 60
  --sc-lp 200` ducks the wet signal each time the sidechain triggers,
  so the reverb breathes with the kick instead of washing over
  transients. Reports mean / peak gain reduction in the report JSON.
- **Built-in IR pack.** New `tools/generate_irs.py` synthesises six
  impulse responses into `tools/irs/` (plate_short, plate_long,
  room_tight, room_live, hall_concert, spring_guitar). Synthetic so
  no licensing — generated from noise + exponential decay + spectral
  shaping. `apply_reverb --ir-preset hall_concert` loads
  `tools/irs/hall_concert.wav` via the existing Convolution engine.
  `--list-ir-presets` shows what's available.

CLAUDE.md gains an updated `apply_reverb` row covering the new flags
plus a new `generate_irs.py` row. docs/knowledge.md gains three new
subsections under "Reverb in a Rock Mix": "Choosing between algorithmic
and convolution", "BPM-synced pre-delay" (with a tempo lookup table),
and "Sidechain reverb (pumping pattern)".

Tests: 4 new cases in test_smoke.py cover BPM-to-pre-delay math (known
values + unknown-division rejection), sidechain envelope ducking
(loud input ducks, quiet input passes through, depth floor honoured).
Total suite 45 tests, all green.

### Session-audit tool for phase-coherent duplicates

A user-by-ear test on the terido_v3 mix surfaced that two drum tracks
were being summed phase-coherently — KICK OUT and KICK OUT.dup1, both
pointing at the same WAV in the session.json, both active in the
mix_config — adding +6 dB to the kick-out body. The user heard it as
"the drummer played that part twice" but it was the session editor's
duplication, not a re-take.

To make this catchable up-front (not by re-listen after delivery):

- **New tool `tools/audit_session.py`** groups session.json tracks by
  the set of source files their clips reference. Any group with more
  than one track is a duplicate candidate; the report recommends a
  primary to keep (shortest track name) and the rest to deactivate.
  Output: `audit_report.json` + `audit_report.txt`. On the terido
  session it surfaces 8 groups (KICK OUT × 2, SN TOP × 2, BASS DI × 3,
  four vocal groups × 3 each, Urgen Serenity Master × 2).
- **CLAUDE.md Session start checklist** gains a required step
  immediately after reading knowledge.md: run audit_session.py, show
  the grouped report to the user, and ASK which copy to keep before
  generating mix_config.
- **CLAUDE.md Analysis decision tree** gains a "session opened, mix_config
  not yet generated" trigger with `audit_session.py` as the required tool.
- **CLAUDE.md tools table** gains the new tool row.
- **Pytest**: 2 new test cases in test_smoke.py (duplicate detection +
  no-duplicate sanity). Total suite now 41 tests, all green.

The terido_v3 dedup iteration documented in
`output/terido_v3/session_summary.md` is what unblocked the drum-bus
parallel-sat relevance check (LRA stayed above 4 LU once the duplicates
weren't doubling the bus peak) and tightened the mix LRA from 3.02 to
3.96 LU on the rendered mix.

### Iteration findings: gentle drum-bus preset, master-preset choice, ref-deck severity (cf3f1ea)

Three operational rules surfaced while pushing the terido_v3 session
from raw to delivery-ready. Captured so the agent doesn't have to
re-derive them next session.

- **New preset `comp_drum_bus_gentle`** (2:1, -8 dB, 15 ms attack,
  150 ms release): a gentler SSL-glue alternative to the default
  `comp_drum_bus` (4:1, -10 dB). The default preset compresses hard
  enough to drop the mix LRA below 4 LU, which then blocks the master
  clipper and drum-bus parallel-sat relevance checks downstream. The
  gentle preset preserves enough LRA headroom for the downstream
  make-it-hit tools to function. New CLAUDE.md rule: "LRA-driven drum
  bus preset choice".
- **New rule: use `--master-preset transparent` on a refmatched mix.**
  Running tonally-active master presets (`modern_rock`,
  `modern_rock_mb`, `pop` — each adds highshelf EQ + side highshelf +
  exciter) on top of a mix that already carries `compare_reference
  --apply` inverse-delta EQ compounds the tonal moves and pushes the
  top another +2-4 dB above the reference. The `transparent` preset
  is the right choice — does LUFS norm and ISP limit only, no
  spectral re-shaping. Verified on terido_v3: refmatched mix +
  transparent → all four streaming format-conformance verdicts green.
- **Clarified: reference-deck verdict is a tonal GUIDE, not a hard
  delivery gate.** The hard gates are LUFS, true peak, phase, punch —
  red there must block delivery. The reference-deck spectral delta is
  a second opinion; red there can mean the master is off OR the
  reference is just different (vocal mix, era, genre tilt). New
  full classification table in knowledge.md "Reference deck is a
  tonal GUIDE, not a hard delivery gate"; cross-referenced from
  CLAUDE.md decision-tree row and ground rule.

### Master chain feature completeness — multiband + M/S + width + vinyl elliptical + batch health

Field-test review on the freshly-shipped master_mix flagged five real gaps
between the v1 master toolset and a production-grade mastering kit. All
five closed in this batch.

**New chain steps (in `master_mix.py`)** — every chain preset can now wire
these in via a single config field; the existing presets were updated to
use them where appropriate:

- **Multiband compressor** (`multiband` field) — 3-band Linkwitz-Riley LR4
  split + per-band pedalboard.Compressor on the master. Reuses the band
  helpers from `apply_multiband_comp.py`. New chain preset
  `modern_rock_mb` puts a tight low / breathing mid+high preset on the
  master instead of glue comp.
- **M/S processing** (`ms` field) — independent mid/side EQ and gain on
  the master, identical pattern to the render_mix master.ms block but
  callable from the chain preset. `modern_rock`, `modern_rock_mb`, and
  `pop` presets now ship a +1 dB side highshelf at 7-8 kHz for the
  classic "wider top" mastering trick.
- **Stereo width control** (`stereo_width` field) — scalar that scales
  the side channel after M/S. 1.0 = no change, > 1 = wider, < 1 =
  narrower. `pop` ships 1.1, `modern_rock_mb` ships 1.05, `hip_hop`
  ships 0.95 (keeps the 808 centred).
- **Vinyl elliptical EQ** — sub-mono filter below 150 Hz on the side
  channel. Triggers automatically for the `vinyl_pre` format (via the
  new `vinyl_elliptical_hz` field on the format preset). Stops the
  cutter head from leaving the groove on wide bass — classic vinyl
  mastering requirement.

**New batch mode (in `master_health.py`)**:

- **`--all-formats`** scans the output dir for `master_<format>.wav`
  files and produces a per-format scorecard plus a cross-format summary
  table. Pairs naturally with `master_mix --all-formats`.
- Vinyl / no-limiter formats are now handled correctly — a true peak
  above the ceiling no longer flags red on `vinyl_pre`; it's expected
  (the cutter does its own limiting). The conformance report carries
  a `true_peak_note` explaining when downstream gear handles the limit.

**Tests**: 6 new pytest cases in `test_master.py` cover stereo-width
identity / mono / widening, vinyl elliptical sub-mono attenuation, M/S
side-gain effect on width, multiband shape preservation, and the
batch-mode summary. Total suite: 39 tests, all green.

**Field test on terido_v2**: re-mastered with the `modern_rock_mb` preset
across all formats. Spotify / Apple / YouTube / Tidal all landed at
target LUFS with delta 0.00; vinyl_pre output correctly applied the
sub-mono elliptical filter and the verdict came back green despite a
+1.72 dBTP true peak (correctly classified as no-limiter-format
expected behaviour).

### Mastering pipeline (master_mix + master_health) — separate phase

The project now covers the full mix→master→delivery flow as two distinct
phases with their own scorecards. The render_mix master chain stays in
its existing role (the mix-engineer's polish during render), and a new
independent mastering pass runs on the finished mix.wav.

**New tools:**
- `tools/master_mix.py` — mastering pass on stereo mix.wav. Chain:
  EQ → glue comp → harmonic exciter → soft/hard clipper → LUFS norm →
  ISP-aware true-peak limiter → post-limiter LUFS correction → optional
  16-bit dither.
- `tools/master_health.py` — master-level scorecard, distinct from
  mix_health. Checks format conformance (LUFS / true peak / 8x codec-ISP
  estimate), per-band phase coherence (sub-mono check, top-wide check),
  per-band M/S width profile, punch index (percentile-90 short envelope
  vs long envelope), compression-history detection, reference-deck
  comparison (multi-reference averaged).

**Format presets** (7): spotify, apple, youtube, tidal (all -14 to -16
LUFS, -1 dBTP, 24-bit), cd (-9 LUFS, -1 dBTP, 16-bit dithered),
vinyl_pre (-12 LUFS, no clipper / no limiter for the cutter), and
broadcast (-23 LUFS, EBU R128).

**Mastering chain presets** (5): gentle, modern_rock (default), pop,
hip_hop, transparent.

**Important fix discovered during field test**: pedalboard.Limiter
applies internal makeup gain, so the post-limiter integrated LUFS sits
~2-5 dB above the target. The chain now does a post-limiter LUFS
re-measure and a downward-only correction. Verified on the terido_v2
mix: spotify master lands at -14.00 LUFS (delta 0.00 from target),
all-green master_health scorecard.

**Punch index formula corrected**: the original `mean(short_env) /
mean(long_env)` formula in master_health doesn't discriminate squashed
from dynamic material because both envelopes converge on the same RMS
mean. Replaced with `percentile_90(short_env) / mean(long_env)` so the
metric actually compares transient peaks to sustained bed.

**Master_health LUFS measurement fixed**: was mono-summing the stereo
master before measuring (returns ~3 dB lower than BS.1770 spec). Now
measures on the native stereo signal as the standard requires.

**Docs**:
- CLAUDE.md workflow diagram now shows the explicit "mix phase ends"
  → "master phase begins" boundary; decision tree gains the master
  triggers; new ground rules: mix_health must gate master_mix, and
  every master_mix output must be followed by master_health.
- docs/knowledge.md gains a "Mastering Workflow and Philosophy" section
  covering format targets, codec ISP, punch index, per-band phase rules,
  compression history, and reference-deck workflow.
- README workflow diagram updated to show the master phase.
- BACKLOG.md: mastering moved to "Completed since this file was created";
  new BACKLOG entry "Advanced mastering features" for true codec
  roundtrip, ISRC embed, DDP, vinyl-specific elliptical EQ, stem
  mastering variants.

**Tests**: 16 new tests in `tests/test_master.py` covering format
presets, chain presets, master_mix end-to-end (LUFS targeting, TP
ceiling, CD bit depth, vinyl limiter skip), and master_health
(conformance round-trip, per-band phase coherence for mono and wide
signals, punch index discrimination, compression history detection).
Total suite: 33 tests, all green.

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
