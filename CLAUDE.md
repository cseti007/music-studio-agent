# music-mix-agent — Claude session instructions

## What this project is

AI-assisted multi-track mixing pipeline for recorded stems (rock band, orchestral, etc.).
Claude acts as the agent: analyzes stems, reads the output data, proposes and applies processing.
All processing happens via Python CLI tools in `tools/`. Claude orchestrates them via Bash.

## Session start checklist

1. Read `docs/knowledge.md` — domain knowledge base (LUFS targets, per-instrument guidelines, trends).
2. Ask the user what session/folder they are working with today.
3. Check `output/` for any previous analysis runs on that session.
4. **Run `tools/audit_session.py output/<session>/session.json --output-dir output/<session>/analysis`** — surfaces tracks that share identical source files (phase-coherent duplicates). Show the flagged groups to the user and ASK which to keep before generating mix_config. Common patterns: `<name>` + `<name>.L` + `<name>.R` triples (one stereo wav referenced three times), or `<name>` + `<name>.dup1.XX` pairs (editor accidentally cloned a track). Setting `active: false` on the suggested deactivate-list in mix_config prevents +6 dB phase-coherent doubling.
5. Ask the user which takes / mic-blend / dup-versions to use in the render (per `mix_config.json` `active` field). Don't decide unilaterally — the `render_mix --generate-config` output explicitly says "Set active=false for alternate takes you don't want (dup versions)"; surface that decision to the user.
6. **Ask the user what genre/style** the song is (modern_rock / classic_rock / pop / hip_hop / jazz_acoustic / other). The generic `volume_db: 0.0` defaults from `render_mix --generate-config` rarely match modern conventions — drums/bass are typically the foundation with guitar 3-4 dB below, vocal 2 dB above guitar. Use `--style NAME` on `--generate-config` to load genre-appropriate bus starting points from `tools/style_profiles/<name>.json` `default_bus_volume_db`. The agent and user should still iterate from there — the profile values are a *starting reference*, not a fixed answer.
7. Ask what the goal is before running anything (delivery-ready master, demo, mix-health gate against a reference, etc).

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
| `tools/audit_session.py` | Audit session.json for tracks that share identical source files (phase-coherent duplicates). Groups tracks by their source-file set, recommends a primary to keep and the rest to deactivate. Run at session start before generating mix_config. Outputs audit_report.json + audit_report.txt. | `<session.json> --output-dir output/<session>/analysis` |
| `tools/apply_gain.py --per-clip` | Clip gain: normalize each clip to consistent LUFS, then assemble full stem | `--per-clip session.json --track "NAME" --output-dir output/<session>/tracks` |
| `tools/apply_gain.py --per-channel` | Stem gain: apply single gain to assembled stem to reach LUFS target | `--per-channel assembled.wav --preset stem\|premix\|spotify\|apple\|amazon\|broadcast` |
| `tools/analyze.py` | Analyze a stem: LUFS, LRA, crest factor, transient density, spectral centroid, stereo balance/correlation/M-S width, 1/3-octave freq response, hum detection, 10-band text spectrogram + RMS waveform + PNG | `<file> --output-dir output/<session>/tracks/<track>` |
| `tools/batch_analyze.py` | Parallel wrapper around `analyze.py` using `multiprocessing.Pool`. Scans `<session>/tracks/*/assembled.wav` (or accepts an explicit `--files` list) and runs analyses across N workers. Output is identical to running `analyze.py` per stem. 5-6× faster than the serial loop on an 8-core machine. | `output/<session> [--workers N] [--skip-existing]` or `--files a.wav b.wav --output-dir DIR` |
| `tools/align_phase.py` | Phase-align a target stem to a reference via cross-correlation | `--reference ref.wav --target tgt.wav --output-dir output/<session>/tracks` |
| `tools/apply_eq.py` | Apply EQ filter chain: notch, HP, LP, bandpass, peak, lowshelf, highshelf. Instrument presets. Auto-notch from hum detection. | `<file> --output-dir DIR [--preset NAME] [--filter JSON]... [--from-analysis analysis.json]` |
| `tools/apply_compression.py` | Apply dynamic range compression (pedalboard/JUCE). Parallel compression via --mix. Sidechain compression via --sidechain (custom envelope follower; pedalboard has no native sidechain). Instrument presets. | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--ratio N] [--attack MS] [--release MS] [--mix 0-1] [--sidechain FILE] [--sc-hp HZ] [--sc-lp HZ]` |
| `tools/apply_reverb.py` | Apply reverb to a stem. Two engines: **algorithmic** Freeverb (presets: snare_plate, snare_plate_big, snare_gated, room_drums, guitar_room, hall_ambient) or **convolution** via `--ir <wav>` / `--ir-preset NAME` (built-in IR pack: plate_short, plate_long, room_tight, room_live, hall_concert, spring_guitar). Pre-delay can be tempo-synced via `--bpm + --pre-delay-division`. Sidechain ducking on the reverb tail via `--sidechain kick.wav` (classic "pumping reverb" pattern). Insert mode (dry+wet) or send mode (--send, wet only). | `<file> --output-dir DIR [--preset NAME \| --ir-preset NAME \| --ir WAV] [--send] [--pre-delay MS] [--bpm BPM --pre-delay-division eighth\|sixteenth\|...] [--wet 0-1] [--hp HZ] [--lp HZ] [--sidechain WAV] [--sc-depth DB] [--sc-hp HZ] [--sc-lp HZ]` |
| `tools/generate_irs.py` | Generates the IR pack into `tools/irs/` (six synthetic impulse responses — plate_short, plate_long, room_tight, room_live, hall_concert, spring_guitar). Synthetic so no licensing friction, ships with the repo. Run once at project setup. | `python tools/generate_irs.py` (no args) |
| `tools/apply_gate.py` | Noise gate for drum bleed control. State machine (CLOSED/ATTACK/OPEN/HOLD/RELEASE) with RMS envelope follower and hysteresis. Presets: gate_kick, gate_snare_top, gate_snare_bottom, gate_tom, gate_room. | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--range DB] [--attack MS] [--hold MS] [--release MS] [--hysteresis DB]` |
| `tools/apply_transient.py` | Transient shaping: independently controls attack (+sharper/-softer) and sustain (+longer/-tighter) using fast/slow RMS envelope pair. Only meaningful on percussive stems — use analysis `transient_profile` to decide. Presets: transient_kick_punch, transient_kick_tight, transient_snare_crack, transient_snare_tight, transient_tom_tight. | `<file> --preset NAME [--attack DB] [--sustain DB] --output-dir DIR` |
| `tools/apply_amp.py` | Tube amp simulation + cabinet EQ for bass DI. Asymmetric soft clipping (even harmonics) + cabinet frequency response. Presets: ampeg_svt, ampeg_svt_driven, ampeg_slap, slap_bass, di_clean. | `<file> --preset NAME [--drive 0-1] [--asymmetry 0-1] [--hp HZ] [--lp HZ] [--low-shelf-hz HZ] [--low-shelf-db DB] [--mid-hz HZ] [--mid-db DB] [--mid-q Q] --output-dir DIR` |
| `tools/apply_saturation.py` | Harmonic saturation: tape (symmetric tanh, even+odd), tube (asymmetric tanh, even harmonics → warmth), clipper (cubic soft clip, odd harmonics → presence). RMS-normalized output. Parallel mode via --mix. Presets: sat_tape_subtle, sat_tape_drums, sat_tube_bass, sat_tube_guitar, sat_clipper_parallel. | `<file> --output-dir DIR [--preset NAME] [--mode tape\|tube\|clipper] [--drive 0-1] [--asymmetry 0-1] [--mix 0-1]` |
| `tools/apply_delay.py` | Delay/echo: normal (slapback, single echo, multi-tap with feedback) and pingpong (alternating L/R, mono→stereo). BPM-synced via --bpm + --division. HP/LP on wet signal. Send mode (--send) for bus return routing. Presets: delay_slapback_snare, delay_slapback_guitar, delay_pingpong_send, delay_pre_delay. | `<file> --output-dir DIR [--preset NAME] [--mode normal\|pingpong] [--delay-ms MS] [--feedback 0-0.95] [--mix 0-1] [--bpm BPM] [--division eighth\|dotted-eighth\|...] [--hp HZ] [--lp HZ] [--send]` |
| `tools/compare_reference.py` | Compare target mix against reference: 1/3-octave spectral delta (loudness-matched), LUFS/LRA/crest factor delta, spectral balance by region, ASCII two-sided bar chart, EQ recommendations for bands above --threshold. Optional `--apply WAV` bakes the inverse-delta peak EQ chain (max 6 filters, ±6 dB cap) into a corrected WAV. Outputs comparison.json + comparison.txt. | `reference.wav target.wav --output-dir DIR [--threshold DB] [--apply OUT.wav] [--apply-phase minimum\|zero]` |
| `tools/detect_masking.py` | Frequency masking detector: finds stem pairs competing in the same 1/3-octave band. All stems LUFS-normalized to -18 LUFS before comparison; PSD is computed only on active frames (RMS > -45 dBFS) and pairs are time-gated (Jaccard co-activity < 0.15 suppressed). Severity: CRITICAL (<3 dB gap), HIGH (3-6 dB), MODERATE (6-10 dB). Auto-discovers stems from session output dir by stage. Outputs masking_report.json + masking_report.txt with heatmap and ranked pair list. | `output/<session> --output-dir DIR [--stage raw\|eq\|comp\|fx] [--threshold DB]` or `stem1.wav stem2.wav ... --output-dir DIR` |
| `tools/render_mix.py` | Sum processed stems into a stereo mix. Hierarchical bus routing. Blend normalization for multi-mic guitars. Per-bus: volume, pan, **eq (zero-phase)**, comp_preset, saturation (tape), parallel_saturation (guarded), reverb_send. Master chain: glue comp + EQ + clipper (guarded) + M/S (guarded) + LUFS normalize + true peak limit. Stage rendering: `--stage raw\|eq\|comp\|fx` renders the mix using stem files from that processing stage (bus+master chain always runs). Output: `mix_stage_<stage>.wav`. **`--generate-config --style NAME`** loads genre-appropriate `default_bus_volume_db` from `tools/style_profiles/<name>.json` (modern_rock / classic_rock / pop / hip_hop / jazz_acoustic) — without it, every bus starts at 0 dB which rarely matches modern conventions. | `output/<session> --generate-config [--style NAME]` then `mix_config.json --render [--output mix.wav] [--stems] [--stage raw\|eq\|comp\|fx]` |
| `tools/mix_health.py` | Session-level mix scorecard. Runs after render_mix and produces a green/yellow/red verdict across 7 checks: integrated LUFS vs target, true peak vs ceiling, LRA, M/S width, low-freq mono compatibility, tonal balance vs reference (optional), masking pairs (from masking_report.json), and stem pumping detection (from stems/). Outputs mix_health.json + mix_health.txt. **Run this last in the MIX phase — gate to the master phase.** | `output/<session> [--reference ref.wav] [--lufs-target -14] [--tp-ceiling -1.0] [--output-dir DIR]` |
| `tools/master_mix.py` | Mastering pass on a finished stereo mix.wav. Full chain: EQ → optional multiband → glue comp → exciter → optional M/S processing → optional stereo width → optional vinyl elliptical EQ → clipper → LUFS norm → ISP-aware limiter → post-limiter LUFS correction → optional dither. 7 format presets (spotify, apple, youtube, tidal, cd, vinyl_pre, broadcast) and 6 chain presets (gentle, modern_rock, modern_rock_mb, pop, hip_hop, transparent). `--all-formats` produces all delivery variants from one input. | `mix.wav --output-dir DIR [--format spotify\|...] [--all-formats] [--master-preset modern_rock\|modern_rock_mb\|...] [--target-lufs N] [--tp-ceiling N]` |
| `tools/style_check.py` | Grade a stereo mix against a named style profile — quantitative answer to "is this a modern_rock mix" without needing a reference track. Built-in profiles: `modern_rock`, `classic_rock`, `pop`, `hip_hop`, `jazz_acoustic`. Measures integrated LUFS, LRA, crest factor, and 5-band spectral RMS (tonal balance) at the profile's LUFS target, returns a traffic-light verdict (GREEN/YELLOW/RED) + 0-100 score + per-check deltas + EQ recommendations for off-target bands. Hard-fail rule: a RED on LUFS or LRA forces overall RED. | `mix.wav --style NAME --output-dir DIR` or `--list-styles` |
| `tools/build_chain.py` | Aggregate every `*_report.json` in a session's `tracks/<stem>/` folders into a single `mix_chain.json` recall sheet — the canonical record of what processing was applied to each stem, in what order, with what parameters. Non-invasive (only reads existing reports). Topo-sorts steps by input→output filename matching so a buggy historical path doesn't break ordering. | `output/<session>` |
| `tools/replay_chain.py` | Replay a `mix_chain.json` recall sheet — rebuild the entire mix from scratch by re-running every step (via subprocess) in recorded order, then `render_mix --render --stems`. Default behaviour is overwrite-in-place (back up first if you need the previous run). `--dry-run` prints the commands without executing; `--stem NAME` replays a single stem for debugging. | `<mix_chain.json \| session_dir> [--dry-run] [--stem NAME]` |
| `tools/master_health.py` | Master-level scorecard, complementary to mix_health. Checks: format conformance (LUFS / TP / codec-ISP estimate), per-band phase coherence (sub-mono / top-wide), per-band M/S width profile, punch index, compression-history detection, reference-deck comparison. `--all-formats` batch mode scans `master_<format>.wav` files in the output dir and produces a cross-format scorecard. Vinyl/no-limiter formats are handled correctly (TP > ceiling is expected and not flagged as red). | `[master.wav] --output-dir DIR [--format spotify\|...] [--all-formats] [--reference ref1.wav ...]` |
| `tools/bus_balance.py` | Per-bus loudness contribution report. For a `--render --stems` output, loads each `stems/stem_<bus>.wav`, applies bus volume_db (incl. parent chain) and measures effective LUFS in the mix. Marks top-level buses (the ones that actually sum into master). Use to answer "is the bass too loud vs drums?" with data instead of vibes. | `<mix_config.json>` |
| `tools/level_notes.py` | Per-note volume leveling on a target time range. Detects onsets, measures each note's attack peak, applies a short 95 ms boost envelope (5 ms pre-fade + 30 ms hold + 60 ms fade-out — fits between onsets so boosts don't overlap and overshoot). Only lifts quiet notes (peak below `--quiet-threshold-db`), never reduces loud ones. Safety scale is segment-only. Intended for uneven slap/finger bass takes where the player swings dynamically and per-clip gain can't help (multiple notes per clip). | `<input.wav> --output <out.wav> --end SEC [--start SEC] [--target-peak-db -4] [--quiet-threshold-db -6] [--max-boost-db 15]` |

### Make-it-hit tools — DATA-GATED, NOT DEFAULT

These tools add perceived loudness, weight, or width. **They are NOT default
processing steps.** Each one ships with a built-in `relevance_check` that
analyses the input and may set `recommend_skip: true` with reasons. When that
happens, the tool refuses to write audio (unless `--force` is passed) and
writes a report explaining why. **Honour the skip.** See "Make-it-hit decision
rules" below for when each one is allowed.

| Tool | What it does | Key args |
|---|---|---|
| `tools/apply_subharm.py` | Sub-bass harmonic synthesizer: generates 2nd/3rd harmonics of the 40-80 Hz fundamental so the low end translates to phones / laptops / BT speakers via the missing-fundamental psychoacoustic. Refuses if the stem has no sub content (sub_60hz < -35 dBFS) or sub is already squashed (band crest < 8 dB). Presets: subharm_subtle, subharm_kick, subharm_strong. | `<file> --output-dir DIR --preset NAME [--drive N] [--harmonic-mix 0-1] [--force]` |
| `tools/apply_haas.py` | Stereo widener via channel delay (5-25 ms). Mono-compat caveat — comb-filters when summed. Refuses on already-wide stems (ms_width > 0.3) or bass-heavy stems (mud risk). Presets: haas_guitar, haas_vocal_doubler, haas_synth_pad. | `<file> --output-dir DIR --preset NAME [--delay-ms MS] [--side L\|R] [--wet 0-1] [--force]` |
| `tools/apply_exciter.py` | HF harmonic generator (Aphex-style). HP + saturate + mix back in. Refuses on already-bright stems (air band > -40 dBFS or centroid > 4 kHz). Presets: exciter_vocal_air, exciter_acoustic_guitar, exciter_dull_mix. | `<file> --output-dir DIR --preset NAME [--hp-hz HZ] [--drive N] [--mix 0-1] [--force]` |
| `tools/apply_multiband_comp.py` | 3-band multiband compressor with Linkwitz-Riley LR4 crossovers. Independent threshold/ratio/attack/release per band. Refuses on already-squashed material (< 2 bands with crest >= 6 dB) or short signals (< 5s). Presets: mb_master_glue, mb_drum_bus, mb_bass. | `<file> --output-dir DIR --preset NAME [--low-thr DB] [--low-ratio N] [--mid-thr DB] ... [--force]` |

In addition, three make-it-hit features live INSIDE `render_mix.py` as master/bus chain options:

- **Master clipper** — soft cubic or hard clip before the brick-wall limiter. Configured under `master.clipper: {threshold_db, knee_db, mode}`. Refuses if sample peak < -10 dBFS or LRA < 4 LU (nothing to clip / already crushed).
- **M/S processing** — independent mid and side EQ + gain. Configured under `master.ms: {mid_eq, side_eq, mid_gain_db, side_gain_db}`. Refuses on near-mono mixes (width < 0.05) or when boosting side on already-wide mixes (width > 0.5).
- **Drum bus parallel saturation** — blend a tube/tape/clipper-saturated copy of the drum bus back in. Configured under `buses.drums.parallel_saturation: {mode, drive, mix}`. Refuses on bus crest < 10 dB, LRA < 4 LU, or non-drum buses.

The skip/apply decision and reason go into `mix_report.json` for the render and `<tool>_report.json` for each standalone tool.

## Output structure

```
output/
└── <session>/
    ├── session.json                  <- parse_session output
    ├── mix_config.json               <- render_mix --generate-config output (edit before rendering)
    ├── mix_chain.json                <- build_chain output: recall sheet of every per-stem processing step (replayable)
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
    ├── mixes/
    │   ├── mix.wav                   <- render_mix --render output (LUFS normalized + true peak limited)
    │   ├── mix_report.json           <- render stats: LUFS, true_peak_dbtp, sample_peak_dbfs, clipper/ms/parallel-sat status, ISP correction
    │   └── stages/                   <- render_mix --stage renders for A/B comparison
    │       ├── mix_stage_raw.wav     <- stems unprocessed, bus+master chain applied
    │       ├── mix_stage_eq.wav
    │       ├── mix_stage_comp.wav
    │       └── mix_stage_fx.wav
    └── masters/                      <- master_mix.py output (one WAV per delivery format)
        ├── master_spotify.wav        <- -14 LUFS, -1 dBTP, 24-bit
        ├── master_apple.wav          <- -16 LUFS
        ├── master_youtube.wav        <- -14 LUFS
        ├── master_tidal.wav          <- -14 LUFS
        ├── master_cd.wav             <- -9 LUFS, 16-bit dithered
        ├── master_vinyl_pre.wav      <- -12 LUFS, sub-mono below 150 Hz, no limiter
        ├── master_<format>_report.json   <- per-format chain log
        └── master_health_<format>.{json,txt}  <- master_health scorecard per format
```

**Session-level analysis always goes to `output/<session>/analysis/`** — never to the session root or mixes/ folder.

**Why analysis files live in three places (don't try to consolidate):**
Each analysis file is co-located with the asset it describes — this is intentional scoping, not clutter.
1. **Per-stem analysis** (`analysis.json`, `spectrogram.{png,txt}`, `*_report.json`) lives **next to the stem's audio** in `tracks/<track>/`. Opening a single stem's folder shows its audio + every analysis and processing report for it in one place.
2. **Session-level / cross-stem analysis** (`audit_report`, `masking_report`, `comparison`, `mix_health`) lives in `analysis/` because it spans multiple stems and doesn't belong to any single one.
3. **Master-level analysis** (`master_<fmt>_report.json`, `master_health_<fmt>.{json,txt}`) lives in `masters/` **next to the matching `master_<fmt>.wav`** — same co-locate principle as the per-stem case.

Moving any of these into a single flat `analysis/` would either break the audio-next-to-analysis property or force a duplicated mirror directory tree.

## mix_config.json bus and master fields

```json
"buses": {
  "drums": {
    "volume_db": 0.0,          // bus fader
    "pan": 0.0,                // -1.0 (L) to 1.0 (R), applied after volume
    "eq": [                    // optional per-bus EQ (zero-phase, applied after volume/pan, BEFORE comp). Same filter schema as master.eq (peak / highshelf / lowshelf / highpass / lowpass)
      {"type": "peak", "hz": 5000, "q": 1.2, "db": -2.0}  // e.g. cut drum cymbals here without touching guitar/vocal presence
    ],
    "comp_preset": "comp_drum_bus",  // optional bus compressor preset. Use comp_drum_bus_gentle if the render mix_health LRA falls below 4 LU (the default 4:1 preset crushes dynamics, which then blocks the master clipper and drum bus parallel-sat relevance checks downstream).
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

Bus processing order: volume → pan → **eq (per-bus, zero-phase)** → comp_preset → saturation → parallel_saturation (guarded) → reverb_send
Master processing order: sum buses → glue comp → clipper (guarded) → M/S (guarded) → EQ → LUFS norm → ISP-aware true-peak limiter

## Workflow

The full pipeline is two phases — MIX, then MASTER. Each phase ends with a
required scorecard (mix_health, master_health). The master phase only starts
when mix_health is green or 1-yellow.

```
[MIX PHASE — analysis triggers in <brackets>]
parse_session
  -> apply_gain --per-clip
     <analyze each new assembled.wav>                                       [Required]
     <compare_reference if user gave a reference, once>                     [Required if ref]
     <detect_masking --stage raw to set EQ priorities>                      [Required]
  -> align_phase (drums)
     <analyze each new assembled_aligned.wav>                               [Required]
  -> apply_eq
     <analyze each *_eq.wav — confirm spectral move>                        [Required]
  -> apply_compression
     <analyze each *_eq_comp.wav — confirm crest + pumping flag>            [Required]
     <detect_masking --stage comp — compare to baseline>                    [Optional]
  -> (make-it-hit, only if data justifies it: subharm / haas / exciter
      / multiband / clipper / parallel_sat / M/S — see decision rules)
     <analyze after each — verify metric + pumping>                         [Required]
  -> render_mix                                              -> mix.wav
     <mix_health.py output/<session> [--reference ref.wav]>                 [Required]
     <if not green: address issues, re-render, re-run mix_health>           [Required loop]

[MASTER PHASE — runs on mix.wav after mix_health passed]
  -> master_mix mix.wav --format <preset>   OR   --all-formats
     <master_health.py master_<fmt>.wav --format <fmt> [--reference deck...]>  [Required per format]
     <if not green: tweak --master-preset or chain settings, re-master>        [Required loop]
  -> delivery: ship master_<format>.wav files

[mix render commands]
1. render_mix output/<session> --generate-config  -> edit mix_config.json
2. render_mix mix_config.json --render --stems    -> mix.wav + stems/stem_<bus>.wav  (ALWAYS --stems)
3. mix_health.py output/<session>                 -> scorecard (REQUIRED)

[master commands]
4. master_mix mix.wav --output-dir output/<session>/masters --all-formats
                                                  -> master_<format>.wav per format
5. master_health.py master_<fmt>.wav --format <fmt> --output-dir DIR
                                                  -> scorecard per format (REQUIRED)
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
| frequency_bands_crest_db.* | frequency_bands_crest_db (per band) | 8-15 dB healthy | < 6 dB: band is squashed (avoid multiband / parallel sat on that band). > 18 dB: band is loose (multiband can help). |
| pumping.pumping_detected | pumping.pumping_detected | false | true is a **suspicion, not a verdict** — manually disambiguate: is it (a) comp/multiband/clipper artifact, or (b) musical strumming/groove? See "Interpreting pumping_detected" below. |

**Rule:** if all metrics are within range and the spectrogram looks normal for the instrument type,
say so explicitly ("analysis looks clean — no action needed before next processing step").
Do not invent problems. Do not recommend processing without a specific reason from the data.

### Interpreting pumping_detected

The pumping detector measures low-frequency envelope modulation (1-5 Hz). It
fires on both real comp artifacts AND on naturally periodic musical material
(strumming, hi-hat patterns, repeated kick hits at song tempo). It cannot
distinguish them from envelope statistics alone — that's the agent's job.

When `pumping_detected: true` appears, before reverting or softening any
upstream step, run this checklist:

1. **Did the flag appear AFTER a compression / multiband / clipper step?**
   Compare the analysis JSON from before and after that step. If pumping
   was `false` before and `true` after, the step caused it — soften the
   release or reduce the ratio and retry.

2. **Is `pump_rate_hz` close to the song tempo's quarter or eighth note?**
   At 120 BPM: quarter = 2.0 Hz, eighth = 4.0 Hz, dotted quarter = 1.33 Hz.
   At 82 BPM (typical rock ballad): quarter = 1.37 Hz. If pump_rate matches
   the groove pulse, it's likely musical — strumming, kick-snare backbeat,
   or hi-hat pattern showing up in the envelope, NOT a comp artifact.

3. **What stem is it on?**
   - Guitar (especially rhythm): periodic strumming pulse — usually musical
   - Bass: usually follows the kick pattern — musical pulse
   - Drum buses (per-stem mode): kick pattern — musical pulse
   - Vocal, master mix, sustained pad: more suspect — comp artifact more likely
   - Per-instrument with clearly uniform decay everywhere: comp artifact

4. **Does the `modulation_depth_db` exceed `lf_excess_db` by a lot?**
   - High depth + moderate excess (e.g. depth 18 dB, excess 5 dB) often = musical
   - High depth + high excess (e.g. depth 8 dB, excess 30 dB) = comp artifact
   - Synthetic continuous-noise pumping test signals show excess > 20 dB.

If the conclusion is "musical pulse, not artifact": **say so explicitly in the
verdict and do not revert**. Note it in the session summary so the next
analysis pass doesn't re-flag it as a problem.

If the conclusion is "comp artifact": revert or soften the offending step,
re-render, re-analyze, confirm the flag clears.

## Analysis tool decision tree — when to run what

This is the source of truth for when each analysis tool MUST run vs. when it's
optional. Do not wait for the user to ask — these are obligations triggered by
events in the workflow. "Required" means you stop and run it before doing
anything else; "optional" means run it if there's a specific question to answer.

| Trigger event | Required / Optional | Run this | What to read |
|---|---|---|---|
| Session opened — `session.json` exists, mix_config not yet generated | **Required, once** | `audit_session.py session.json` | Duplicate groups → which tracks to set `active: false` in mix_config. Always show the report to the user and ASK before deciding which copy to keep. |
| A new `assembled.wav` (or `assembled_aligned.wav`) just landed | **Required** | `analyze.py` on that file | LUFS, hum, transient_profile, frequency_bands, frequency_bands_crest_db, stereo, pumping |
| User provided a reference mix at session start | **Required**, once | `compare_reference.py reference target_or_raw_mix` | LUFS delta (target), spectral balance deltas (EQ goals), LRA delta (compression target) |
| Session opened, before any EQ work | **Required** | `detect_masking.py output/<session> --stage comp` (or `--stage raw` if no comp yet) | CRITICAL + HIGH pairs → primary EQ cut targets |
| Output from `apply_eq` / `apply_compression` / `apply_gate` / `apply_amp` / `apply_saturation` / `apply_transient` / `apply_reverb` / `apply_delay` just landed | **Required** | `analyze.py` on that output | Did the targeted metric move the right way? `pumping_detected` flipped? |
| Output from a make-it-hit tool just landed (`apply_subharm`, `apply_haas`, `apply_exciter`, `apply_multiband_comp`, or a render with `master.clipper` / `master.ms` / `buses.*.parallel_saturation`) | **Required, double check** | `analyze.py` on the output **AND** verify the tool's own `relevance_check` result | (a) targeted metric moved in the intended direction, (b) `pumping.pumping_detected` is still false, (c) `relevance_check.recommend_skip` was honoured |
| `render_mix --render` finished (mix.wav written) | **Required** | `mix_health.py output/<session> [--reference ref.wav if user gave one]` | Green/yellow/red verdict across LUFS, true peak, LRA, M/S width, mono compat, tonal balance, masking, stem pumping |
| `mix_health` returned yellow or red verdicts | **Required loop** | Address each non-green item, re-render, re-run `mix_health.py` | Same — until green or "1 yellow max" |
| `mix_health` passed (green or 1-yellow) — moving from mix to master | **Required transition** | `master_mix mix.wav --output-dir output/<session>/masters --format <preset>` (or `--all-formats`) | Mastering pass per delivery target |
| `master_mix` finished (per format) | **Required** | `master_health.py master_<fmt>.wav --format <fmt>` | format conformance + phase + punch + compression history per delivery |
| `master_health` returned yellow or red | **Required loop, but read which section** | Tweak `--master-preset` or chain params, re-run `master_mix`, re-run `master_health`. **Hard gates** (LUFS, true peak, phase, punch) red = must fix. **Reference deck red** alone = tonal advisory only; ship if the hard gates are green (see knowledge.md "Reference deck is a tonal GUIDE"). | Same — until green on hard gates |
| Between processing stages (eq → comp → fx), want to see if masking improved | Optional | `detect_masking.py output/<session> --stage <stage>` | Compare critical/high counts to the earlier run |
| After rendering, want to compare against reference for master EQ tweaks | Optional | `compare_reference.py reference mixes/mix.wav --output-dir output/<session>/analysis [--apply ...]` | Spectral delta in the rendered mix — feeds master EQ |
| Mix or master is "done" but no reference track was provided; want a style-aware sanity check (does this sound like the intended genre) | Optional | `style_check.py mix.wav --style <modern_rock\|classic_rock\|pop\|hip_hop\|jazz_acoustic> --output-dir output/<session>/analysis` | 0-100 score + GREEN/YELLOW/RED verdict + per-band EQ recommendations vs. the genre's expected tonal balance, LUFS, LRA, and crest. **Read borderline ([OK*]) bands explicitly** — these are GREEN but at severity ≥ 0.7 (within 30% of the YELLOW threshold). A band sitting at +2.3 dB delta with ±2.5 dB tolerance is GREEN, but it's also "barely GREEN" — call it out so the user knows the verdict is on the edge, not comfortably inside spec. |
| After `render_mix --render --stems`, want objective per-bus loudness data (e.g. "is the bass actually too loud relative to drums?") | Optional | `bus_balance.py <mix_config.json>` | Effective LUFS contribution per bus (volume_db applied + parent-chain summed). Top-level buses (`[T]`) are the ones summed into master; sub-buses shown for reference. Use to settle perception arguments with measurement. |
| Source stem has uneven per-note dynamics (slap/finger bass take with wildly varying note levels — per-clip gain can't help because each clip has many notes) | Optional | `level_notes.py <input.wav> --output <out.wav> --end SEC [--start SEC]` | Detects onsets, lifts only the quiet notes via a short ~95 ms boost envelope (no overlap between notes, no reduction of loud notes). Safety-scaled to the modified segment only. Run on the assembled stem before EQ/comp. |
| Process budget hit (4 processing steps on the same stem) | **Required STOP** | `analyze.py` on current state | Stop. Read what the chain actually achieved. Ask the user before adding a 5th step. |

**Operational rules that follow from the table:**

1. **Never skip a "Required" trigger.** If the trigger fires and you didn't run the analysis, you are guessing — that is the failure mode this table exists to prevent.
2. **Always read the JSON, not just the .txt summary.** The .txt is for the user to skim; the JSON is what your decisions must reference.
3. **Quote specific fields when proposing a next step.** "I see `loudness.crest_factor_db: 4.1` and `pumping.pumping_detected: true` — the comp went too hard, reverting and retrying with a 2:1 ratio" beats "the comp seems too much".
4. **Re-run analysis even if you "know" what changed.** The point of the re-analyze loop is to catch the cases where you were wrong about what changed.

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

## Make-it-hit decision rules

The make-it-hit tools (subharm, haas, exciter, multiband, master clipper,
M/S, drum bus parallel sat) exist to **add perceived loudness, weight, or
width**. They are powerful and easy to overuse — agents pattern-match to
"more processing = better mix" and stack them. **DO NOT.** Use only when
the analysis data justifies it. The 2026 rock-mix best practice is "don't
over-process — keep the band's raw energy".

**Required data evidence before applying each tool:**

| Tool | Required evidence (from analyze.py or render output) |
|---|---|
| Master clipper | Sample peak ≥ -10 dBFS AND LRA ≥ 4 LU (the two conditions the code's relevance_check actually enforces). Below either, the clipper has no headroom to recover or just adds fatigue to an already-flat mix. |
| Sub-bass synth (`apply_subharm`) | `frequency_bands.sub_60hz_rms_db` >= -35; `frequency_bands_crest_db.sub_60hz_crest_db` >= 8; stem is bass or kick |
| Drum bus parallel sat | Drum bus crest factor > 10 dB; LRA > 4 LU; user explicitly asked for "punchier drums" or a "fatter kit" |
| Spectral exciter | `spectral_centroid_hz` < 4000; `frequency_bands.air_8khz_plus_rms_db` < -40 |
| M/S width | `stereo.ms_width_ratio` < 0.2 if side boost; do NOT boost side if width > 0.5 |
| Haas widener | `stereo.ms_width_ratio` < 0.3; NOT on bass / low-centroid stems |
| Multiband comp | At least 2 bands with `frequency_bands_crest_db` >= 6 dB; duration > 5s |
| compare_reference --apply | Largest delta band > 2 dB AND it's a tonal/balance issue (not an obvious EQ omission earlier in the chain) |

**If the data does not match, DO NOT apply the tool.** The `relevance_check`
in each tool's report will say `recommend_skip: true` with specific reasons.
That is the source of truth — do not override with `--force` unless the user
explicitly asks.

**Process budget:** a single stem chain should not exceed 4 processing
steps total (typically: gain → EQ → comp → one fx). If you are about to add
a 5th step, **stop and re-examine** whether the earlier steps actually
solved the problem. More processing on already-processed audio compounds
phase shift, transient smearing, and artifacts.

**Re-analyze after every make-it-hit step:**

1. Apply the tool.
2. Immediately run `analyze.py` on the output.
3. Compare the targeted metric:
   - Sub-bass synth on a bass DI → sub_60hz_rms_db should rise ~3-6 dB
   - Master clipper → integrated_lufs should rise 1-3 dB without LRA collapsing
   - Multiband → per-band crest should tighten in the targeted band
   - Exciter → spectral_centroid_hz should rise, air_8khz_plus_rms_db should rise
4. If the metric did not move in the intended direction, **revert** —
   the tool either didn't help or just shifted the problem.
5. Additionally check `analyze.pumping.pumping_detected`. If it **flips to
   true** after a comp / multiband / clipper step (i.e. was `false` before
   the step and `true` after), the step is the cause — revert or soften
   and retry. If it was already `true` before the step (musical pulse —
   common on guitar / drum buses), the step did not cause it; do not
   revert on that ground. See "Interpreting pumping_detected" for the
   musical-vs-artifact disambiguation checklist.

**Do not chain make-it-hit tools blindly.** A typical "make it hit harder"
session adds at most ONE of: clipper, multiband, or parallel-sat per bus.
Stacking all three creates fatigue, not punch.

**LRA-driven drum bus preset choice.** If after `render_mix --render` the
mix LRA sits below 4 LU and the render log shows the clipper and
parallel-sat guards firing ("SKIPPED — LRA X LU < 4"), the upstream drum
bus compressor is the load-bearing cause. Switch the drums bus from
`comp_drum_bus` (4:1, -10 dB) to **`comp_drum_bus_gentle`** (2:1, -8 dB,
15 ms attack, 150 ms release), re-render, re-check. The gentle preset
preserves enough LRA headroom for the downstream make-it-hit tools to
function. Do this before reaching for the master clipper / parallel sat
manually; you want the relevance check to PASS, not be force-overridden.

**Master preset choice on a refmatched mix.** If you ran
`compare_reference.py --apply` on the mix.wav, the resulting
`mix.wav` already carries up to six inverse-delta peak EQ filters (the
spectral correction toward the reference). Running a tonally-active
master preset (`modern_rock`, `modern_rock_mb`, `pop` — all of which
add highshelf EQ + side highshelf + exciter) on top of an already-
refmatched mix **compounds the tonal moves**, pushing the top another
+2-4 dB above the reference. Use **`--master-preset transparent`** on
refmatched mixes — let the LUFS normalisation + ISP-aware limiter do
their job without re-shaping the tonal balance the refmatch step
already settled. Picked for v3: refmatched mix → `transparent` master
preset → all four streaming format-conformance verdicts green.

## Reproducibility — mix_chain.json (recall sheet)

After a session is finished (mix_health green / master delivered), generate a
`mix_chain.json` recall sheet so the entire mix is reproducible from the
canonical session inputs:

```bash
python3 tools/build_chain.py output/<session>
# Writes output/<session>/mix_chain.json
```

`build_chain` aggregates every `*_report.json` under `tracks/<stem>/` into a
single JSON that lists, per stem, the exact ordered chain of processing
steps with their arguments. Steps are topologically sorted by input→output
filename so the recorded order matches the actual processing order.

To rebuild the mix from a chain:

```bash
python3 tools/replay_chain.py output/<session>/mix_chain.json
# Re-runs every step in subprocess; finishes with render_mix --render --stems
```

Useful flags:
- `--dry-run` — print the commands without executing (sanity check the chain)
- `--stem "KICK IN.05"` — replay one stem only (debugging)

**When to run `build_chain`:**
- After `mix_health.py` passes (mix is "done") — before moving to master.
- After a v2 / v3 iteration finishes, so each version has its own recall.
- Before deleting any session-wide audio — the chain is the smallest possible
  record of "how this mix was made" (a few hundred KB JSON vs. gigabytes of WAV).

**Default behaviour is overwrite-in-place** — replay writes into the same
`output/<session>/` directory the chain references. Back up first if you
want to keep the previous run intact. (We chose this over a "_replay"
sibling dir to avoid having every tool's path-baked references in the
chain go stale.)

The chain is a faithful record, not a fixer — if the original run had a bug
(e.g. an align_phase output written to an accidentally-nested path), the
recall sheet reproduces it. Edit the chain JSON by hand if you need to
patch a historical mistake before replay.

## Ground rules

- One stem at a time until the user confirms the result is correct.
- State what you observe from the analysis before proposing any action.
- If a result looks wrong (clipping, unexpected LUFS), stop and diagnose before continuing.
- Keep `docs/knowledge.md` updated when new domain knowledge is found.
- **Every `render_mix --render` MUST be followed by `mix_health.py`.** No exceptions. If `mix_health` returns any RED verdict, fix it and re-render; if it returns more than 1 YELLOW, address them. Only declare the mix "done" when `mix_health` shows all green (or at most 1 yellow with a reasoned justification).
- **Always pass `--stems` to `render_mix --render`.** Stems are required: (a) `mix_health.py` uses `stems/` for per-stem pumping detection — without them the pumping check is silently skipped, and (b) per-bus submixes (`stems/stem_drums.wav`, `stems/stem_bass.wav`, `stems/stem_<bus>.wav`, all LUFS-normalized to -18) are deliverables the user expects alongside `mix.wav`. Run as `render_mix mix_config.json --render --stems`, not bare `--render`.
- **The mix phase MUST gate the master phase.** Don't start `master_mix` until `mix_health` is green or 1-yellow. Mastering a broken mix wastes work and hides problems under a louder ceiling.
- **Every `master_mix` output (per format) MUST be followed by `master_health.py`** with the same `--format` flag, to verify the delivery target was actually hit. RED on a **hard gate** (LUFS / true peak / phase / punch) means re-master, not "ship anyway". RED on the **reference-deck section alone** is a tonal advisory, not a hard gate — investigate, document, and ship if the hard gates are green. See knowledge.md "Reference deck is a tonal GUIDE, not a hard delivery gate" for the full table.
- Follow the **"Analysis tool decision tree — when to run what"** section above as obligations, not suggestions. Skipping a required analysis trigger means you are guessing — and guessing wrong is more expensive than running a 30-second analysis.
