# music-mix-agent — Claude session instructions

## What this project is

AI-assisted multi-track mixing pipeline for recorded stems (rock band, orchestral, etc.).
Claude acts as the agent: analyzes stems, reads the output data, proposes and applies processing.
All processing happens via Python CLI tools in `tools/`. Claude orchestrates them via Bash.

## Session start checklist

1. Read `docs/knowledge.md` — domain knowledge base (LUFS targets, per-instrument guidelines, trends).
2. Ask the user what session/folder they are working with today.
3. Check `output/` for any previous analysis runs on that session.
4. Ask what the goal is before running anything.

## Python environment

Use the project virtual environment. Set it up once:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Always invoke tools with the activated venv's Python:

```bash
python3 tools/<script>.py
```

## Available tools

| Tool | What it does | Key args |
|---|---|---|
| `tools/parse_session.py` | Parse DAW session file (.ptx, .als) into canonical session.json | `<session_file> --output-dir output/<session> --audio-dir <audio_dir>` |
| `tools/apply_gain.py --per-clip` | Clip gain: normalize each clip to consistent LUFS, then assemble full stem | `--per-clip session.json --track "NAME" --output-dir output/<session>` |
| `tools/apply_gain.py --per-channel` | Stem gain: apply single gain to assembled stem to reach LUFS target | `--per-channel assembled.wav --preset stem\|spotify\|apple\|amazon\|broadcast` |
| `tools/analyze.py` | Analyze a stem: LUFS, LRA, crest factor, transient density, spectral centroid, stereo balance/correlation/M-S width, 1/3-octave freq response, hum detection, 10-band text spectrogram + RMS waveform + PNG | `<file> --output-dir output/<session>/<track>` |
| `tools/align_phase.py` | Phase-align a target stem to a reference via cross-correlation | `--reference ref.wav --target tgt.wav --output-dir output/<session>` |
| `tools/apply_eq.py` | Apply EQ filter chain: notch, HP, LP, bandpass, peak, lowshelf, highshelf. Instrument presets. Auto-notch from hum detection. | `<file> --output-dir DIR [--preset NAME] [--filter JSON]... [--from-analysis analysis.json]` |
| `tools/apply_compression.py` | Apply dynamic range compression (pedalboard/JUCE). Parallel compression via --mix. Sidechain compression via --sidechain (custom envelope follower; pedalboard has no native sidechain). Instrument presets. | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--ratio N] [--attack MS] [--release MS] [--mix 0-1] [--sidechain FILE] [--sc-hp HZ] [--sc-lp HZ]` |
| `tools/apply_reverb.py` | Apply reverb to a stem. Algorithmic reverb (Freeverb) with pre-delay, HP/LP on return. Insert mode (dry+wet) or send mode (wet only). Presets: snare_plate, snare_plate_big, snare_gated, room_drums, guitar_room, hall_ambient. | `<file> --preset NAME [--send] [--pre-delay MS] [--wet 0-1] [--hp HZ] [--lp HZ] --output-dir DIR` |
| `tools/apply_gate.py` | Noise gate for drum bleed control. State machine (CLOSED/ATTACK/OPEN/HOLD/RELEASE) with RMS envelope follower and hysteresis. Presets: gate_kick, gate_snare_top, gate_snare_bottom, gate_tom, gate_room. | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--range DB] [--attack MS] [--hold MS] [--release MS] [--hysteresis DB]` |
| `tools/apply_transient.py` | Transient shaping: independently controls attack (+sharper/-softer) and sustain (+longer/-tighter) using fast/slow RMS envelope pair. Only meaningful on percussive stems — use analysis `transient_profile` to decide. Presets: transient_kick_punch, transient_kick_tight, transient_snare_crack, transient_snare_tight, transient_tom_tight. | `<file> --preset NAME [--attack DB] [--sustain DB] --output-dir DIR` |
| `tools/apply_amp.py` | Tube amp simulation + cabinet EQ for bass DI. Asymmetric soft clipping (even harmonics) + cabinet frequency response. Presets: ampeg_svt, ampeg_svt_driven, ampeg_slap, slap_bass, di_clean. | `<file> --preset NAME [--drive 0-1] [--asymmetry 0-1] [--hp HZ] [--lp HZ] [--low-shelf-hz HZ] [--low-shelf-db DB] [--mid-hz HZ] [--mid-db DB] [--mid-q Q] --output-dir DIR` |
| `tools/apply_saturation.py` | Harmonic saturation: tape (symmetric tanh, even+odd), tube (asymmetric tanh, even harmonics → warmth), clipper (cubic soft clip, odd harmonics → presence). RMS-normalized output. Parallel mode via --mix. Presets: sat_tape_subtle, sat_tape_drums, sat_tube_bass, sat_tube_guitar, sat_clipper_parallel. | `<file> --output-dir DIR [--preset NAME] [--mode tape\|tube\|clipper] [--drive 0-1] [--asymmetry 0-1] [--mix 0-1]` |
| `tools/apply_delay.py` | Delay/echo: normal (slapback, single echo, multi-tap with feedback) and pingpong (alternating L/R, mono→stereo). BPM-synced via --bpm + --division. HP/LP on wet signal. Send mode (--send) for bus return routing. Presets: delay_slapback_snare, delay_slapback_guitar, delay_pingpong_send, delay_pre_delay. | `<file> --output-dir DIR [--preset NAME] [--mode normal\|pingpong] [--delay-ms MS] [--feedback 0-0.95] [--mix 0-1] [--bpm BPM] [--division eighth\|dotted-eighth\|...] [--hp HZ] [--lp HZ] [--send]` |
| `tools/compare_reference.py` | Compare target mix against reference: 1/3-octave spectral delta (loudness-matched), LUFS/LRA/crest factor delta, spectral balance by region, ASCII two-sided bar chart, EQ recommendations for bands above --threshold. Outputs comparison.json + comparison.txt. | `reference.wav target.wav --output-dir DIR [--threshold DB]` |
| `tools/detect_masking.py` | Frequency masking detector: finds stem pairs competing in the same 1/3-octave band. All stems LUFS-normalized to -18 dBFS before comparison. Severity: CRITICAL (<3 dB gap), HIGH (3-6 dB), MODERATE (6-10 dB). Auto-discovers stems from session output dir by stage. Outputs masking_report.json + masking_report.txt with heatmap and ranked pair list. | `output/<session> --output-dir DIR [--stage raw\|eq\|comp\|fx] [--threshold DB]` or `stem1.wav stem2.wav ... --output-dir DIR` |
| `tools/render_mix.py` | Sum processed stems into a stereo mix. Hierarchical bus routing. Blend normalization for multi-mic guitars. Per-bus: volume, pan, comp_preset, saturation (tape), reverb_send. Master chain: glue comp + EQ + LUFS normalize + true peak limit. Stage rendering: `--stage raw\|eq\|comp\|fx` renders the mix using stem files from that processing stage (bus+master chain always runs). Output: `mix_stage_<stage>.wav`. | `output/<session> --generate-config` then `mix_config.json --render [--output mix.wav] [--stems] [--stage raw\|eq\|comp\|fx]` |

## Output structure

```
output/
└── <session>/
    ├── session.json                  <- parse_session output
    ├── mix_config.json               <- render_mix --generate-config output (edit before rendering)
    ├── analysis/                     <- session-level analysis (compare_reference, detect_masking)
    │   ├── masking_report.json       <- detect_masking output
    │   ├── masking_report.txt
    │   ├── comparison.json           <- compare_reference output
    │   └── comparison.txt
    ├── tracks/
    │   └── <track_name>/
    │       ├── assembled.wav             <- apply_gain --per-clip output (stage: raw)
    │       ├── assembled_gained.wav      <- apply_gain --per-channel output (if run)
    │       ├── assembled_aligned.wav     <- align_phase output (if run)
    │       ├── assembled_eq.wav          <- apply_eq output (stage: eq)
    │       ├── assembled_[aligned_]eq_comp.wav  <- apply_compression output (stage: comp)
    │       ├── assembled_eq_comp_<fx>.wav       <- reverb/amp output (stage: fx)
    │       ├── analysis.json             <- LUFS, LRA, crest factor, stereo, transient density, freq response, hum
    │       ├── spectrogram.png
    │       ├── spectrogram.txt           <- 10-band spectrogram + RMS waveform + freq response + stats summary
    │       ├── gain_report.json
    │       ├── align_report.json
    │       ├── eq_report.json
    │       └── comp_report.json
    ├── stems/                        <- render_mix --stems output (per-bus submixes at -18 LUFS)
    │   ├── stem_drums.wav
    │   ├── stem_bass.wav
    │   └── stem_guitar.wav
    └── mixes/
        ├── mix.wav                   <- render_mix --render output (LUFS normalized + true peak limited)
        ├── mix_report.json           <- render stats (LUFS, peak, active tracks, stage)
        └── stages/                   <- render_mix --stage renders for A/B comparison
            ├── mix_stage_raw.wav     <- stems unprocessed, bus+master chain applied
            ├── mix_stage_eq.wav
            ├── mix_stage_comp.wav
            └── mix_stage_fx.wav
```

**Session-level analysis always goes to `output/<session>/analysis/`** — never to the session root or mixes/ folder.

## mix_config.json bus and master fields

```json
"buses": {
  "drums": {
    "volume_db": 0.0,          // bus fader
    "pan": 0.0,                // -1.0 (L) to 1.0 (R), applied after volume
    "comp_preset": "comp_drum_bus",  // optional bus compressor preset
    "saturation": {"drive": 0.3},   // optional tape saturation (symmetric tanh)
    "parent_bus": null         // routes into this parent bus
  },
  "guitar": {
    "volume_db": -3.0,
    "pan": 0.0,
    "saturation": {"drive": 0.25},
    "reverb_send": {"preset": "guitar_room", "wet": 0.15},  // bus-level reverb send
    "parent_bus": null
  }
},
"master": {
  "lufs_target": -14.0,
  "true_peak_dbfs": -1.0,
  "comp": {                    // optional master glue compressor
    "threshold_db": -10.0,
    "ratio": 2.0,
    "attack_ms": 10.0,
    "release_ms": 300.0,
    "makeup_db": 0.0
  },
  "eq": [                      // optional master EQ (zero-phase, applied after comp)
    {"type": "highpass", "hz": 30},
    {"type": "highshelf", "hz": 12000, "db": 1.5}
  ]
}
```

Bus processing order: volume → pan → comp_preset → saturation → reverb_send
Master processing order: sum buses → glue comp → EQ → LUFS norm → true peak limiter

## Workflow

```
[full pipeline from DAW session]
parse_session -> apply_gain --per-clip -> analyze -> align_phase (drums) -> apply_eq -> apply_compression -> render_mix

[render_mix steps]
1. render_mix output/<session> --generate-config  -> edit mix_config.json
2. render_mix mix_config.json --render            -> mix.wav
```

**Gain staging logic:**
- `--per-clip` is the primary gain-staging step. It normalizes clip-level inconsistencies
  (different recording gain settings across sessions) AND assembles the full stem.
  Target: -18 LUFS per clip (set in config.toml [gain] per_clip_target_lufs).
- `--per-channel` is only needed on top of that for delivery normalization (Spotify, Apple Music, etc.)
  or when receiving pre-assembled stems. After a correct --per-clip pass, the stem is already
  at mix-ready levels — a second --per-channel pass is optional.

**DRUMS: never use per-clip normalization for gain staging.**
A drummer records in one continuous take; editorial clip cuts are the editor's work, not
separate recordings. Per-clip normalization on drum tracks creates artificial level jumps at
edit boundaries (e.g. stereo collapse at cut points from OH level mismatch).
Correct drum workflow: use `--per-clip` only to assemble the stem (for region placement),
then apply `--per-channel` on the assembled result for a single uniform gain pass.

- Never apply processing without reading the analysis first.
- Always read `spectrogram.txt` from output — it is the primary way to understand what is in a stem.
  The STATS SUMMARY block at the bottom of spectrogram.txt contains the new metrics — always read it.
- After applying any processing, re-analyze the output file to verify the result.
- Ask before processing multiple stems in bulk — do one first and confirm it is correct.

## Analysis interpretation — what to say after reading analysis.json

After reading analysis, always give a recommendation. The recommendation may be "no action needed" —
that is a valid and useful answer. Never leave analysis results without a verdict.

Read all of these fields from analysis.json and comment on each that is outside normal range:

| Field | Location in JSON | Normal range (rock mix stem) | Action if outside range |
|---|---|---|---|
| integrated_lufs | loudness.integrated_lufs | -22 to -14 LUFS (stem) | Too hot: reduce pre-clip gain. Too quiet: check assembly or delivery stage. |
| loudness_range_lu | loudness.loudness_range_lu | 5–16 LU (stem), 3–12 LU (mix) | < 3 LU: over-compressed; > 20 LU: may need compression before mix |
| crest_factor_db | loudness.crest_factor_db | 10–18 dB (healthy) | < 8 dB: brick-walled. > 20 dB: very dynamic — may need compression |
| stereo.balance_db | stereo.balance_db | ±1.5 dB | > ±3 dB: strong imbalance — check panning or channel assignment |
| stereo.lr_correlation | stereo.lr_correlation | +0.6 to +0.95 | Negative: out-of-phase (check mono compat). < 0.4: very wide |
| stereo.ms_width_ratio | stereo.ms_width_ratio | 0.1–0.4 | < 0.05: near-mono. > 0.6: very wide (check mono compat) |
| transient_density_per_sec | transient_density_per_sec | varies by instrument | Compare sections — sudden jumps indicate uneven playing |
| spectral_centroid_hz | spectral_centroid_hz | varies by instrument | Use to confirm EQ changes had the expected effect |
| transient_profile.transient_prominence_db | transient_profile.transient_prominence_db | > 8 dB (percussive) | Kick/snare/tom: < 4 dB → Attack+ may help. Distorted guitar / fingerstyle bass: always low → ignore. |
| transient_profile.transient_prominence_std_db | transient_profile.transient_prominence_std_db | < half of mean | std > mean on ANY instrument → uneven playing. Primary unevenness indicator for slap bass. Suggests compression. |
| transient_profile.decay_time_ms | transient_profile.decay_time_ms | kick < 150ms, snare < 100ms | Percussive only. Long decay → Sustain- may tighten. Overhead/room mics: ignore. |
| hum_detection.hum_detected | hum_detection.hum_detected | false | true: apply notch filters per hum_detection.harmonics |

**Rule:** if all metrics are within range and the spectrogram looks normal for the instrument type,
say so explicitly ("analysis looks clean — no action needed before next processing step").
Do not invent problems. Do not recommend processing without a specific reason from the data.

## Progress reporting during long operations

Many tools run for 10-30 seconds per stem. Keep the user informed:

- Before starting a tool: announce what you are running and on which file.
  Example: "Running analyze.py on KICK IN.05 (step 1 of 12)..."
- When processing multiple stems in sequence: show a counter.
  Example: "[3/12] Analyzing FLOOR TOM.05..."
- After each tool completes: report the key result in one line.
  Example: "KICK IN.05 done — LUFS -17.2, prominence 10.9 dB, decay 31ms"
- If a step is notably slow (render_mix, analyze on long stems): say so upfront.
  Example: "Rendering full mix — this takes ~30s..."

Never run a batch silently. The user cannot see tool call progress, only your text output.

## Reference comparison workflow

Use `tools/compare_reference.py` whenever a reference mix is available or after rendering a mix.

**When to run it:**
- User provides a reference track ("sound like this") — run immediately before processing starts.
  Establishes baseline targets for spectral balance, LUFS, and LRA.
- After rendering a mix (`render_mix.py --render`) — compare the rendered mix against the reference.
  Use the EQ recommendations to guide master EQ adjustments in mix_config.json.
- After a stage render (`--stage eq|comp|fx`) — compare stages A/B to hear what each processing
  step added or removed spectrally. Reference can be the raw stage or an external reference.

**How to interpret comparison.txt:**
- LUFS delta: the single most important number. If target is > 2 dB quieter than reference,
  the mix will sound worse on streaming platforms even with correct processing.
- Spectral balance (bottom/mids/top): quick three-number check. Delta > ±2 dB in any region
  means a clear tonal imbalance vs. the reference.
- [!] flagged bands in the chart: direct EQ targets. Translate directly into apply_eq.py
  `--filter` arguments or add to mix_config.json master EQ chain.
- The comparison is loudness-matched before computing the spectral delta — report the LUFS
  delta separately from the spectral recommendations, not as part of the EQ advice.

**Standard reference comparison command:**
```bash
python3 tools/compare_reference.py \
  <reference.wav> output/<session>/mixes/mix.wav \
  --output-dir output/<session>/analysis
```

**Frequency masking — run before EQ decisions:**
Use `tools/detect_masking.py` at the start of a session (before EQ) to see which stems
compete in the same frequency bands. Run on the comp stage for the most accurate picture.
The CRITICAL and HIGH pairs directly inform which stems need EQ cuts and where.

```bash
python3 tools/detect_masking.py \
  output/<session> --output-dir output/<session>/analysis --stage comp
```

Read `masking_report.txt` and state the top CRITICAL/HIGH pairs before proposing any EQ.
Classic rock mix patterns to expect: kick mics vs. bass DI at 60-120 Hz, snare vs. guitar
body at 200-400 Hz, guitar vs. vocal at 2-4 kHz.

## Progress checklist

When processing multiple stems or executing a multi-step plan, maintain a visible checklist
in your text output so the user always knows where things stand.

**Format:** print the full checklist before starting, then reprint it (updated) after each
completed step. Use `[ ]` for pending, `[x]` for done, `[>]` for in progress.

Example:
```
[ ] KICK IN  — EQ + comp
[x] KICK OUT — EQ
[>] KICK SUB — EQ (running...)
[ ] SN TOP   — EQ + comp + gate
[ ] BASS DI  — amp sim + sidechain comp
```

- Always show the checklist before the first tool call of a batch.
- Update after each stem/step completes — reprint with the new state.
- For single-stem operations this is not needed; only use it when 3 or more steps are planned.

## Session end summary

At the end of a mix session (after the final render and analysis), always write a stage
summary to `output/<session>/session_summary.md`. Create or overwrite the file using the
Write tool. Do this unprompted — no need to ask.

The file should contain three sections:

**1. Per-stem-group processing table** — what was applied at each stage:

| Stage | Stem group | What was done |
|---|---|---|
| eq | KICK IN | HP 80Hz, +3dB@3.5kHz click |
| comp | KICK IN | 4:1, att 4ms, rel 80ms, makeup 4dB |
| ... | ... | ... |

**2. Bus and master chain** — separate table for bus-level and master processing.

**3. Key metric deltas** — before vs. after for the most important metrics:
- Low/mid frequency gap
- LUFS target achieved
- Stereo width (ms_width_ratio, lr_correlation)
- Any hum eliminated

After writing the file, tell the user where it was saved.

## Ground rules

- One stem at a time until the user confirms the result is correct.
- State what you observe from the analysis before proposing any action.
- If a result looks wrong (clipping, unexpected LUFS), stop and diagnose before continuing.
- Keep `docs/knowledge.md` updated when new domain knowledge is found.
