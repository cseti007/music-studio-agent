# Current policy

Follow the current workflow below and `docs/knowledge.md` before historical
session examples. Existing instrument recipes and relevance thresholds are
heuristics, not industry requirements or evidence of perceived quality. Never
let a green score replace listening approval. Project/user instructions take
precedence over this guidance.

# music-studio-agent — Claude session instructions

## What this project is

AI-assisted multi-track mixing pipeline for recorded stems (rock band, orchestral, etc.).
Claude acts as the agent: analyzes stems, reads the output data, proposes and applies processing.
All processing happens via Python CLI tools in `tools/`. Claude orchestrates them via Bash.

## Session start checklist

1. Read `docs/knowledge.md`. Distinguish requirements, recommendations,
   heuristics, preferences, and observations from an earlier session.
2. Declare listening capability: direct audition available, external human
   feedback available, or measurements only. A playback link or an audio file
   on disk does not mean the agent heard it. Record the tool and excerpts when
   direct audition succeeds; report failures accurately.
3. Establish source session, artistic intent, takes, reference versions, and
   requested deliverables. Carry forward existing approvals within their scope.
4. Prefer consolidated, time-aligned stems. DAW parsing preserves a limited
   clip layout and cannot reproduce plugins, routing automation, or all fades.
   Audit paths and layouts before processing. Do not infer redundant takes
   from names, or switch to continuous reconstruction from clip counts.
5. Preserve original levels, channel relationships, and edits. Normalization,
   auto-trim, new crossfades, and editorial changes need a stated purpose.
6. Make a section map: intro, representative quiet/dense passages, transitions,
   solos, and ending, with actual timestamps. Describe intended contrast and
   list uncertain judgments rather than pretending to know the arrangement.
7. Render an initial balance as a draft. Style profiles and presets provide
   optional hypotheses; they cannot demonstrate genre authenticity or quality.

## Listening, evidence, and completion

- Separate three evidence types in every decision: measured observation,
  inferred cause, and heard/user-reported result. Never describe an inference
  as something heard. Tests verify software behavior, not musical success.
- Without direct audition, use the user's listening feedback to guide musical
  revisions. Continue independent analysis and render reviewable alternatives;
  do not stall authorized work, but keep subjective conclusions provisional.
- Compare representative sections at matched loudness with
  `tools/prepare_audition.py`. State offsets when comparing trimmed versions.
  The tool uses linear gain and common peak attenuation, with no limiting.
  It rejects incompatible sample rates, silence, and out-of-range excerpts.
- For each consequential change, record the problem, timestamp, hypothesis,
  intervention, measurements, matched comparison, feedback, and keep/revert
  decision. Change one coherent group of settings per comparison. A technical
  repair may proceed on objective evidence; an unreviewed creative move stays
  provisional. Do not stack effects just because a metric suggests relevance.
- Evaluate kick/snare/toms with overheads and rooms as one kit; inspect bass
  interaction, guitar microphone blends, vocal phrase consistency, depth, and
  transitions in context. Full-song LUFS and correlation cannot resolve these.
- A comment such as "balance feels right" approves that balance in the heard
  excerpt. It does not approve timbre, dynamics, the full song, or a later master.
  Preserve Take 2/double preferences across revisions, but renew audio approval
  when the actual render changes. Record criticism as a revision request.
- First renders are drafts. Use `tools/review_delivery.py` on the exact export
  before reporting delivery readiness. Technical reports always leave listening
  pending; only scoped, externally supplied human feedback can satisfy the
  listening portion of delivery review. Direct agent audition remains useful
  evidence for decisions but cannot impersonate user/engineer sign-off.
- Exporting drafts is allowed while feedback is pending. Say "rendered for
  review" or "technical checks passed; listening pending" as appropriate.
  Never claim professional quality, commercial success, or artist equivalence
  from a preset, score, export, or test suite.

Listening evidence is a JSON list, oldest first. Each record has `artifact_hash`
(from the reviewed source), `reviewer` (the actual person), `source` (message or
session-note identifier), `quote` (their actual words), `scope`, and `decision`.
Scopes are `vocal_balance`, `section`, `full_song`, and `codec_roundtrip`.
Decisions are `approved` or `revision_requested`. No default positive answer is
allowed. Feedback on an excerpt is associated with its source hash from
`audition.json` and stays narrow. Whole-song approval applies to the full file.
The tool validates scope and version, not whether the evidence is authentic;
never manufacture a quote, reviewer, or approval source to pass the check.

## Python environment

The project uses the `music-mix-agent` conda environment. **Prefer activating
the env once at session start and using plain `python` from then on** — every
`conda run -n music-mix-agent python` call pays ~1.5–2 s of conda resolution
overhead on top of Python's own ~3 s cold-start. For a 20-step replay_chain
that's ~30–40 s of pure environment-setup time.

```bash
# Once per shell (saves ~1.5-2s per tool invocation)
conda activate music-mix-agent
python tools/<script>.py ...

# Or, if you can't activate (one-off invocation from outside an activated shell)
conda run -n music-mix-agent python tools/<script>.py ...
```

**Parallelising batched calls.** For "apply tool X on every stem in a session"
workflows, use the OS-level parallelism that's already there — don't sleep
behind a single Python interpreter:

```bash
# Run an already-decided per-stem command in parallel (-P 4 = 4 workers).
# sh -c defers $(dirname ...) until xargs substitutes each path.
ls output/<session>/tracks/*/assembled.wav | xargs -P 4 -I {} sh -c \
    'python tools/apply_eq.py "$1" --preset NAME --output-dir "$(dirname "$1")"' _ {}

# batch_analyze.py already does multiprocessing internally
python tools/batch_analyze.py output/<session> --workers 8
```

## Available tools

| Tool | What it does | Key args |
|---|---|---|
| `tools/apply_dynamic_eq.py` | Blend a bounded bell cut under a linked RMS detector. External sidechain must match the input timeline; no trigger means dry bypass. Use for an auditioned frequency-overlap hypothesis, not automatic spectral matching. | `input.wav --output-dir DIR --frequency-hz N [--sidechain-path trigger.wav] [--max-cut-db N] [--threshold-db N]` |
| `tools/prepare_audition.py` | Export paired excerpts with matched loudness and source hashes; linear gain only, no audition performed. | `before.wav after.wav --output-dir NEW_DIR [--start S] [--after-start S] [--duration S]` |
| `tools/review_delivery.py` | Check exact export plus scoped human feedback; missing/stale/narrow approval cannot complete delivery. Exit 1 means review is incomplete or failed. `--bit-depth 32` accepts 32-bit PCM or float. `codec_roundtrip` evidence carries the hash of the reviewed WAV export, not of the decoded codec file. | `audio.wav --output-dir DIR --tp-ceiling N [--sample-rate N] [--bit-depth N] [--evidence records.json] [--target-lufs CONTRACTUAL_N] [--require-codec-review]` |
| `tools/apply_automation.py` | Apply explicit channel-linked gain rides with linear interpolation in dB, preserving timing and floating-point headroom. Records replay metadata. | `<input.wav> --output-dir DIR --points curve.json` |
| `tools/parse_session.py` | Parse DAW session file (.ptx, .als) into canonical session.json | `<session_file> --output-dir output/<session> --audio-dir <audio_dir>` |
| `tools/audit_session.py` | Audit session.json for identical resolved source paths and clip layouts. Findings are duplicate candidates for review, not proof that a track should be muted. Run at session start before generating mix_config. Outputs audit_report.json + audit_report.txt. | `<session.json> --output-dir output/<session>/analysis` |
| `tools/apply_gain.py --per-clip` | Assemble clips at original levels and positions. `--normalize` and crossfades are opt-in. Continuous mode reconstructs source takes and changes edits; use only as an approved editorial alternative. `--normalize-per-source` levels those placements only when requested. `--crossfade-ms` defaults to 0 ms in per-clip mode (edits preserved) and 50 ms in continuous mode; contiguous cuts within the same take use equal-gain fades, other joins equal-power. `--no-normalize` is a deprecated no-op alias. `config.toml` is read from the project root and resolved values are recorded. | `--per-clip session.json --track "NAME" --output-dir output/<session>/tracks [--normalize] [--source-mode per-clip\|continuous] [--crossfade-ms MS] [--interloper-head-ms MS] [--interloper-tail-ms MS] [--normalize-per-source --source-target-lufs LUFS]` |
| `tools/apply_gain.py --per-channel` | Stem gain: apply single gain to assembled stem to reach LUFS target | `--per-channel assembled.wav --preset stem\|premix\|spotify\|apple\|amazon\|broadcast` |
| `tools/analyze.py` | Analyze a stem: LUFS, LRA, per-channel sample/true peak and crest factor, transient density, spectral centroid, stereo balance/correlation/M-S width, 1/3-octave freq response (mono downmix), hum detection (narrow persistent mains lines in the quietest passages), 10-band text spectrogram + RMS waveform + PNG. Unmeasurable values are null; files with non-finite samples are rejected. | `<file> --output-dir output/<session>/tracks/<track>` |
| `tools/batch_analyze.py` | Parallel wrapper around `analyze.py` using `multiprocessing.Pool`. Scans `<session>/tracks/*/assembled.wav` (or accepts an explicit `--files` list) and runs analyses across N workers. Output is identical to running `analyze.py` per stem. 5-6× faster than the serial loop on an 8-core machine. With `--files`, each result folder is named after the file (or its parent folder when file names repeat). | `output/<session> [--workers N] [--skip-existing]` or `--files a.wav b.wav --output-dir DIR` |
| `tools/align_phase.py` | Delay/polarity-align a target stem to a reference via cross-correlation of a shared active window. Refuses (exit 1, nothing written) when the absolute correlation is below `--min-correlation` (default `[align] min_correlation = 0.3` in config.toml). Use only for a supported alignment hypothesis; acoustic arrival delays are preserved by default. | `--reference ref.wav --target tgt.wav --output-dir output/<session>/tracks [--max-delay-ms MS] [--min-correlation R]` |
| `tools/apply_eq.py` | Apply EQ filter chain: notch, HP, LP, bandpass, peak, lowshelf, highshelf. Instrument presets (`--list-presets` lists only presets with EQ filters; a non-EQ preset is an error). Notches from analysis hum detection via `--from-analysis` — check the hum finding first. | `<file> --output-dir DIR [--preset NAME] [--filter JSON]... [--from-analysis analysis.json]` |
| `tools/apply_compression.py` | Apply dynamic range compression (pedalboard/JUCE). Parallel compression via --mix. Stereo channels share one linked gain curve. Sidechain compression via --sidechain (custom envelope follower; the sidechain must match the input sample rate and length). Explicit CLI values override preset values. Instrument presets. | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--ratio N] [--attack MS] [--release MS] [--mix 0-1] [--sidechain FILE] [--sc-hp HZ] [--sc-lp HZ]` |
| `tools/apply_reverb.py` | Apply reverb to a stem. Two engines: **algorithmic** Freeverb (presets: snare_plate, snare_plate_big, snare_gated, room_drums, guitar_room, hall_ambient) or **convolution** via `--ir <wav>` / `--ir-preset NAME` (built-in IR pack: plate_short, plate_long, room_tight, room_live, hall_concert, spring_guitar). Pre-delay can be tempo-synced via `--bpm + --pre-delay-division`. Sidechain ducking on the reverb tail via `--sidechain kick.wav` (classic "pumping reverb" pattern). Insert mode (dry+wet) or send mode (--send, wet only). The convolution return is level-matched to Freeverb at the same room_size/damping/width (`convolution_makeup_db` in the report), so preset `wet` values mean the same for both engines. Sidechain must match input length and rate. `--output-dir` is required. | `<file> --output-dir DIR [--preset NAME \| --ir-preset NAME \| --ir WAV] [--send] [--pre-delay MS] [--bpm BPM --pre-delay-division eighth\|sixteenth\|...] [--wet 0-1] [--hp HZ] [--lp HZ] [--sidechain WAV] [--sc-depth DB] [--sc-hp HZ] [--sc-lp HZ]` |
| `tools/generate_irs.py` | Generates the IR pack into `tools/irs/` (six synthetic impulse responses — plate_short, plate_long, room_tight, room_live, hall_concert, spring_guitar). Synthetic so no licensing friction, ships with the repo. Run once at project setup. | `python tools/generate_irs.py` (no args) |
| `tools/apply_gate.py` | Noise gate for drum bleed control. State machine (CLOSED/ATTACK/OPEN/HOLD/RELEASE) with RMS envelope follower and hysteresis. Presets: gate_kick, gate_snare_top, gate_snare_bottom, gate_tom, gate_room. | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--range DB] [--attack MS] [--hold MS] [--release MS] [--hysteresis DB]` |
| `tools/apply_transient.py` | Transient shaping: independently controls attack (+sharper/-softer) and sustain (+longer/-tighter) from the dB difference between a fast RMS envelope and a slower one-pole envelope, so the effect does not depend on the hit's absolute level. Only meaningful on percussive stems — use analysis `transient_profile` to decide. Presets: transient_kick_punch, transient_kick_tight, transient_snare_crack, transient_snare_tight, transient_tom_tight. | `<file> --preset NAME [--attack DB] [--sustain DB] --output-dir DIR` |
| `tools/apply_amp.py` | Tube-style amp simulation + cabinet EQ for bass DI. 4x-oversampled asymmetric soft clipping (adds even harmonics) with DC blocking + cabinet frequency response. Presets: ampeg_svt, ampeg_svt_driven, ampeg_slap, slap_bass, di_clean. | `<file> --preset NAME [--drive 0-1] [--asymmetry 0-1] [--hp HZ] [--lp HZ] [--low-shelf-hz HZ] [--low-shelf-db DB] [--mid-hz HZ] [--mid-db DB] [--mid-q Q] --output-dir DIR` |
| `tools/apply_saturation.py` | Harmonic saturation: tape (symmetric tanh, odd harmonics), tube (asymmetric tanh, even + odd harmonics, DC-blocked at 10 Hz), clipper (cubic soft clip, odd harmonics). RMS-normalized output. Parallel mode via --mix. Explicit CLI values override preset values. Presets: sat_tape_subtle, sat_tape_drums, sat_tube_bass, sat_tube_guitar, sat_clipper_parallel. | `<file> --output-dir DIR [--preset NAME] [--mode tape\|tube\|clipper] [--drive 0-1] [--asymmetry 0-1] [--mix 0-1]` |
| `tools/apply_delay.py` | Delay/echo: normal (slapback, single echo, multi-tap with feedback) and pingpong (alternating L/R, mono→stereo). BPM-synced via --bpm + --division. HP/LP filter the summed wet signal once (repeats do not darken progressively). Send mode (--send) for bus return routing. Explicit CLI values override preset values. Presets: delay_slapback_snare, delay_slapback_guitar, delay_pingpong_send, delay_pre_delay (wet-only send). | `<file> --output-dir DIR [--preset NAME] [--mode normal\|pingpong] [--delay-ms MS] [--feedback 0-0.95] [--mix 0-1] [--bpm BPM] [--division eighth\|dotted-eighth\|...] [--hp HZ] [--lp HZ] [--send]` |
| `tools/apply_deesser.py` | Frequency-specific sidechain compressor for vocal sibilance control. Detection band is 5-8 kHz (5500-8500 default), gain reduction is full-band. Chain placement: AFTER compression (the comp amplifies sibilance peaks, so the de-esser catches them at the comp's output). Presets: deesser_smooth, deesser_aggressive, deesser_male_lead, deesser_female_lead. `relevance_check` skips when the sibilance band peak is below -25 dBFS (nothing to de-ess). | `<file> --output-dir DIR [--preset NAME] [--threshold DB] [--ratio N] [--attack MS] [--release MS] [--detect-low HZ] [--detect-high HZ] [--force]` |
| `tools/apply_pitch_correct.py` | Vocal pitch correction via librosa.pyin + psola PSOLA shifting + scale quantisation. Detects voice f0 frame-by-frame, snaps to the nearest scale degree of the chosen key/mode, blends original→quantised by `strength` (0=no correction, 1=full Auto-Tune snap). 7 modes: major, minor, harmonic_minor, dorian, mixolydian, chromatic, natural_minor. Presets: pitch_correct_subtle, pitch_correct_pop, pitch_correct_hard_tune. Requires the `psola` package. | `<file> --output-dir DIR --scale-root NOTE --scale-mode MODE [--strength 0-1] [--preset NAME] [--fmin HZ] [--fmax HZ]` |
| `tools/compare_reference.py` | Compare target mix against reference: 1/3-octave spectral delta (level-matched on the median band delta, `level_match_method`), LUFS/LRA/crest factor delta, worst-channel dBTP, spectral balance by region, ASCII two-sided bar chart, EQ hypotheses for bands above --threshold. Optional `--apply WAV` bakes an inverse-delta EQ (adjacent same-sign bands merged into one filter, max 6 filters, each and the combined response capped at ±6 dB; the set is scaled down (`auto_eq_scale`) rather than worsen any band). The corrected WAV is float and not peak-normalized (`apply_output_peak_dbfs`). Unmeasurable values are null. Outputs comparison.json + comparison.txt. | `reference.wav target.wav --output-dir DIR [--threshold DB] [--apply OUT.wav] [--apply-phase minimum\|zero]` |
| `tools/detect_masking.py` | Candidate-overlap detector: finds stem pairs with comparable energy in the same 1/3-octave band. All stems LUFS-normalized to -18 LUFS before comparison, which discards the actual mix balance (stated in the report as `normalization_note`). Levels are 1/3-octave band power in dB re full scale (`band_level_unit`), computed only on active frames (RMS > -45 dBFS), with a -50 dB band-power floor; pairs are time-gated by the overlap coefficient (shared active frames / active frames of the shorter part; < 0.15 suppressed). In session mode, tracks marked inactive in mix_config are skipped and the fx stage uses the mix_config file (or the chain leaf, never a `*_send.wav`). Findings are hypotheses to audition in context. Severity: CRITICAL (<3 dB gap), HIGH (3-6 dB), MODERATE (6-10 dB). **NOTE:** the CLI `--threshold` default is 6.0, which reports only HIGH+CRITICAL — pass `--threshold 10` to surface the MODERATE band too. Auto-discovers stems from session output dir by stage. Outputs masking_report.json + masking_report.txt with heatmap and ranked pair list. | `output/<session> --output-dir DIR [--stage raw\|eq\|comp\|fx] [--threshold DB]` or `stem1.wav stem2.wav ... --output-dir DIR` |
| `tools/render_mix.py` | Sum processed stems into a stereo mix. Hierarchical bus routing. Blend normalization for multi-mic guitars. **Optional per-bus auto-trim (`--auto-trim`):** every bus carries `auto_trim_db` (calibration) alongside `volume_db` (style/user taste). Effective gain = `auto_trim_db + volume_db`. Calibration is computed during `--generate-config --auto-trim` (and refreshable via `--recompute-autotrim`) by measuring each bus's dry-sum LUFS in topological order so every bus's dry-sum output lands at -18 LUFS regardless of stem count. `volume_db` then layers a pure relative offset on top — drum dry-sum still sits at -18, but `vocal_lead` with `+2.0` sits at -16. Pan: stereo sources and buses use a balance law with unity at centre; mono sources use constant-power panning. Per-bus: volume, pan, **eq (zero-phase)**, comp_preset (stereo-linked), saturation (**guarded — tape sat refuses if bus crest < 8 dB or LRA < 4 LU**), parallel_saturation (guarded), reverb_send. **Master chain — premaster mode (default):** glue comp + EQ + peak normalize to `peak_target_dbfs` (default -3 dBFS). NO clipper, NO M/S, NO LUFS-target normalization, NO brick-wall limiter. Project default: mastering (`master_mix.py`) handles delivery processing; intentional artistic mix processing may differ. Legacy combined mix+master chain via `master.premaster_mode: false` opt-in (clipper / M/S / lufs_target / true_peak_dbfs only applied in this mode). Premaster `mix.wav` is written as 32-bit float; when any reverb is configured the render is padded by 3 s so tails are not cut. Unknown comp/reverb preset names fail before stems are loaded. Stage rendering: `--stage raw\|eq\|comp\|fx` renders the mix using stem files from that processing stage (bus+master chain always runs). Output: `mix_stage_<stage>.wav`; with `--stems`, stage stems go to `stems/<stage>/`. **`--generate-config --style NAME`** loads optional starting `default_bus_volume_db` / `default_bus_pan` from `tools/style_profiles/<name>.json` (classic_rock, hip_hop, jazz_acoustic, modern_rock, pop, punchy_modern_rock, tool_inspired); these are project preferences, not genre standards. A branched or partially recorded track folder makes generate-config warn and fall back to the filename ladder. | `output/<session> --generate-config [--style NAME]` then `mix_config.json --render [--output mix.wav] [--stems] [--stage raw\|eq\|comp\|fx]`. Use `mix_config.json --recompute-autotrim` only for intentional recalibration; it changes the balance. |
| `tools/mix_health.py` | Technical peak check plus advisory loudness, phase, masking, and dynamics measurements. Premaster peak gate: RED above 0 dBTP or a sample peak above 0 dBFS, YELLOW between -1 and 0 dBTP, GREEN at or below -1 dBTP. LRA is advisory only. Listening stays pending. | `output/<session> [--reference ref.wav] [--lufs-target N] [--tp-ceiling N] [--output-dir DIR]` |
| `tools/master_mix.py` | Mastering pass on a finished stereo mix.wav. Full chain: EQ → optional multiband (linked per band) → glue comp (stereo-linked) → exciter → optional M/S processing → optional stereo width → optional vinyl elliptical EQ (zero-phase, opt-in) → clipper (monotone soft knee) → LUFS norm → true-peak brickwall limiter (pedalboard `BrickwallLimiter`: linked, 5 ms lookahead, no makeup gain) → 8x-verified safety trim → optional dither. There is no post-limiter loudness correction: on dense material loudness may land below target and `loudness_target_met` reports it. Ceilings: format default (-1 dBTP for streaming), lowered to -2 dBTP when the target is louder than -14 LUFS unless `--tp-ceiling` is given (`tp_ceiling_source` in the report). `cd` (-9 LUFS) is an artistic choice, not a standard; `broadcast` -2 dBTP is a house choice (R128 max is -1). `vinyl_pre`: no LUFS target, peak-normalized to -3 dBTP, no limiter/clipper; sub-mono only with `--vinyl-elliptical [HZ]`. 7 format presets (spotify, apple, youtube, tidal, cd, vinyl_pre, broadcast) and 11 chain presets: **base** (gentle, modern_rock, modern_rock_mb, pop, hip_hop, transparent) + **spatial family** (modern_rock_spatial = sub-mono <150 Hz + +2 dB side shelf @ 8k; **modern_rock_spatial_v10** = sub-mono <200 Hz + +1 dB peak @ 2.5k side + +3 dB shelf @ 8k side + stereo_width 1.05 — historical spatial experiment; **modern_rock_spatial_v9** = +1.5 dB top shelf + exciter mix 0.12 for crisp Leprous/Wheel direction; **modern_rock_spatial_dark** = SAME spatial benefits as v10 but ALL top emphasis dialed back, for ear-fatigue cases; **modern_rock_spatial_noclip** = no-clipper variant; compare actual settings before attributing differences). `--all-formats` produces every variant from one input; use it only when several deliverables were requested (see Workflow step 5). | `mix.wav --output-dir DIR [--format spotify\|...] [--all-formats] [--master-preset modern_rock\|modern_rock_spatial\|modern_rock_spatial_dark\|...] [--target-lufs N] [--tp-ceiling N] [--vinyl-elliptical [HZ]]` |
| `tools/style_check.py` | Measure similarity to project style preferences. Scores cannot establish musical quality or genre authenticity. Profiles describe masters: with `--input-kind auto` (default; premaster when worst-channel TP <= -2.5 dBTP) or `premaster`, LUFS/LRA/crest are N/A. Unmeasurable metrics are N/A and excluded from the score; differences are reported neutrally. CLI succeeds when measurement completes, even for a red similarity score. | `mix.wav --style NAME --output-dir DIR [--input-kind auto\|premaster\|master]` or `--list-styles` |
| `tools/build_chain.py` | Aggregate every `*_report.json` in a session's `tracks/<stem>/` folders into a single `mix_chain.json` recall sheet — the canonical record of what processing was applied to each stem, in what order, with what parameters. Non-invasive (only reads existing reports). Topo-sorts steps by input→output filename matching. Reports without an operation record become `unrecorded` steps with chain-level `warnings` and `verified_operations: false` instead of an error. | `output/<session> [--output PATH]` |
| `tools/replay_chain.py` | Replay a `mix_chain.json` recall sheet — rebuild the entire mix from scratch by validating and re-running supported operations in dependency order, then `render_mix --render --stems`. Recorded operations replay into a `.replay-*` scratch folder with recording suppressed; the WAV is replaced only when its content hash matches, otherwise the original audio and `.operation.json` stay untouched and the candidate's location is reported. Legacy steps and the final render still overwrite. A changed tool module (engine) is a warning ("outputs unverified"); a changed input, source or preset is an error. Unrecorded or legacy steps are rejected before the mix config is validated. `--dry-run` validates recorded calls, data hashes and routing and writes nothing; `--stem NAME` replays a single stem. | `<mix_chain.json \| session_dir> [--dry-run] [--stem NAME]` |
| `tools/master_health.py` | Master-level scorecard, complementary to mix_health. Checks: delivery properties, loudness and 4x/8x waveform true-peak estimates, per-band phase coherence (sub-mono / top-wide), per-band M/S width profile, punch index, compression-history detection, reference-deck comparison. `--all-formats` batch mode scans `master_<format>.wav` files in the output dir and produces a cross-format scorecard. Peak safety also applies to vinyl/no-limiter formats. Codec encode/decode is unavailable; artistic metrics remain advisory. | `[master.wav] --output-dir DIR [--format spotify\|...] [--all-formats] [--reference ref1.wav ...]` |
| `tools/bus_balance.py` | Per-bus loudness contribution report. For a `--render --stems` output, measures each `stems/stem_<bus>.wav` (float, already at its rendered level, before the master chain) and reports LUFS and peak. Marks top-level buses (the ones that actually sum into master). Use to answer "is the bass too loud vs drums?" with data, then confirm by listening. | `<mix_config.json>` |
| `tools/level_notes.py` | Per-note volume leveling on a target time range. Detects onsets, measures each note's attack peak, applies a short 95 ms boost envelope (5 ms pre-fade + 30 ms hold + 60 ms fade-out — fits between onsets so boosts don't overlap and overshoot). Only lifts quiet notes (peak below `--quiet-threshold-db`), never reduces loud ones. No rescaling: audio outside the range is unchanged and overs stay in the float output (reported as `output_peak_dbfs` / `output_exceeds_0dbfs`). Intended for uneven slap/finger bass takes where the player swings dynamically and per-clip gain can't help (multiple notes per clip). | `<input.wav> --output <out.wav> --end SEC [--start SEC] [--target-peak-db -4] [--quiet-threshold-db -6] [--max-boost-db 15]` |
| `tools/find_clicks.py` | Click / sharp-transient forensic. **Sweep mode** scans `mixes/mix.wav` for high inter-sample steps (>= 0.10 default) and lists them ranked by magnitude (with clustering to collapse nearby clicks into single events). **Trace mode** (`--time <sec>`) walks every chain stage at the given timestamp (`source:` files mapped through session.json → every WAV in the track folder ordered by operation records → stem → mix) and prints the max inter-sample step at each, so you can attribute an audible click to the exact stage that introduces it — engineer slip-edit at a clip boundary, polarity inversion, comp ceiling clipping, or a real source recording issue. Always trace BEFORE blaming the chain; many "clicks" turn out to be the source itself (pre-limited bounce, recording clip, or a known anti-phase pedal DI). | `output/<session> [--time SEC] [--threshold 0.10] [--top 20]` |

### Optional creative tools - measurement prompts, audition required

These tools add perceived loudness, weight, or width. **They are NOT default
processing steps.** Each one ships with a built-in `relevance_check` that
analyses the input and may set `recommend_skip: true` with reasons. When that
happens, the tool refuses to write audio (unless `--force` is passed) and
writes a report explaining why. Respect the skip by default; an override needs a
documented reason (see "Creative processing decision rules" below).

| Tool | What it does | Key args |
|---|---|---|
| `tools/apply_subharm.py` | Sub-bass harmonic synthesizer: synthesizes 2nd/3rd harmonics of the 40-80 Hz fundamental by phase multiplication of the analytic signal (no fundamental leak; `--drive` compresses the harmonic envelope as tanh(drive*env)/drive) so the low end translates to phones / laptops / BT speakers via the missing-fundamental psychoacoustic. Refuses if the stem has no sub content (sub_60hz < -35 dBFS) or sub is already squashed (band crest < 8 dB). Presets: subharm_subtle, subharm_kick, subharm_strong. | `<file> --output-dir DIR --preset NAME [--drive N] [--harmonic-mix 0-1] [--force]` |
| `tools/apply_haas.py` | Stereo widener via channel delay (5-25 ms). Mono-compat caveat — comb-filters when summed. Refuses on already-wide stems (ms_width > 0.3) or bass-heavy stems (mud risk). Presets: haas_guitar, haas_vocal_doubler, haas_synth_pad. | `<file> --output-dir DIR --preset NAME [--delay-ms MS] [--side L\|R] [--wet 0-1] [--force]` |
| `tools/apply_exciter.py` | HF harmonic generator. HP + 4x-oversampled symmetric tanh (odd harmonics) + mix back in. Refuses on already-bright stems (air band > -40 dBFS or centroid > 4 kHz). Presets: exciter_vocal_air, exciter_acoustic_guitar, exciter_dull_mix. | `<file> --output-dir DIR --preset NAME [--hp-hz HZ] [--drive N] [--mix 0-1] [--force]` |
| `tools/apply_multiband_comp.py` | 3-band multiband compressor with Linkwitz-Riley LR4 crossovers. Independent threshold/ratio/attack/release per band, stereo-linked within each band. Refuses on already-squashed material (< 2 bands with crest >= 6 dB) or short signals (< 5s). Presets: mb_master_glue, mb_drum_bus, mb_bass. | `<file> --output-dir DIR --preset NAME [--low-thr DB] [--low-ratio N] [--mid-thr DB] ... [--force]` |
| `tools/apply_octaver.py` | Sub-octave generator: pitch-shifts a stem down by N octaves (librosa phase-vocoder), band-passes the shifted signal (default 30-120 Hz), mixes back into the original at a low level (default 0.18-0.25, ~-15 to -12 dB). Adds psychoacoustic weight to bass DI without competing with the upper-bass body. **NOT `relevance_check`-guarded yet** — apply selectively. Typical use: bass CLEAN sub-band (`<bass>_lpsplit.wav`) → sub-octave layer added back as a separate mix track. | `<bass.wav> --output-dir DIR --octaves -1 --hp-hz 30 --lp-hz 90 --mix 0.18` |

In addition, three creative options live INSIDE `render_mix.py` as master/bus chain options. The master clipper and M/S apply only in the legacy `premaster_mode: false` chain; premaster mode ignores them (master_mix has its own):

- **Master clipper** — monotone soft knee or hard clip before the brick-wall limiter. Configured under `master.clipper: {threshold_db, knee_db, mode}`. Refuses if sample peak < -10 dBFS or LRA < 4 LU (nothing to clip / already crushed).
- **M/S processing** — independent mid and side EQ + gain. Configured under `master.ms: {mid_eq, side_eq, mid_gain_db, side_gain_db}`. Refuses on near-mono mixes (width < 0.05) or when boosting side (side EQ boost or `side_gain_db > 0`) on already-wide mixes (width > 0.5).
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
    │       ├── <output>.operation.json   <- one recall record per processed WAV (see Reproducibility)
    │       ├── analysis.json             <- LUFS, LRA, crest factor, stereo, transient density, freq response, hum
    │       ├── spectrogram.png
    │       ├── spectrogram.txt           <- 10-band spectrogram + RMS waveform + freq response + stats summary
    │       ├── gain_report.json
    │       ├── align_report.json
    │       ├── eq_report.json
    │       └── comp_report.json
    ├── stems/                        <- render_mix --stems output (float bus submixes preserving their rendered levels)
    │   ├── stem_drums.wav
    │   ├── stem_bass.wav
    │   ├── stem_vocal_lead.wav      <- only when vocal_lead bus has active tracks
    │   ├── stem_vocal_bg.wav        <- only when vocal_bg bus has active tracks
    │   ├── stem_vocal.wav           <- vocal parent (sums vocal_lead + vocal_bg)
    │   ├── stem_guitar.wav
    │   └── <stage>/                 <- render_mix --stage <stage> --stems output (kept separate from the main render)
    ├── mixes/
    │   ├── mix.wav                   <- render_mix --render output; premaster mode (default): 32-bit float, peak-normalized to peak_target_dbfs, no limiter
    │   ├── mix_report.json           <- render stats: LUFS, true_peak_dbtp, sample_peak_dbfs, parallel-sat status (clipper/ms/limiter only in legacy mode), **bus_peaks**, **master_peaks**, **phase_warnings**
    │   └── stages/                   <- render_mix --stage renders for A/B comparison
    │       ├── mix_stage_raw.wav     <- stems unprocessed, bus+master chain applied
    │       ├── mix_stage_eq.wav
    │       ├── mix_stage_comp.wav
    │       └── mix_stage_fx.wav
    └── masters/                      <- master_mix.py output (one WAV per requested delivery format)
        ├── master_spotify.wav        <- -14 LUFS, -1 dBTP, 24-bit
        ├── master_apple.wav          <- -16 LUFS
        ├── master_youtube.wav        <- -14 LUFS
        ├── master_tidal.wav          <- -14 LUFS
        ├── master_cd.wav             <- -9 LUFS (artistic choice), -2 dBTP, 16-bit dithered
        ├── master_vinyl_pre.wav      <- no LUFS target, peak -3 dBTP, no limiter; sub-mono only with --vinyl-elliptical
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
    "auto_trim_db": -2.5,      // calibration: brings bus dry-sum to -18 LUFS at volume_db=0. WRITTEN BY --generate-config --auto-trim / --recompute-autotrim (0 otherwise) — do not edit manually. Effective gain = auto_trim_db + volume_db.
    "volume_db": 0.0,          // style/user offset on top of auto_trim_db. Style profile populates this (modern_rock: drums 0, vocal_lead +2, guitar -3).
    "pan": 0.0,                // -1.0 (L) to 1.0 (R), applied after volume; balance law with unity at centre on stereo buses
    "eq": [                    // optional per-bus EQ (zero-phase, applied after volume/pan, BEFORE comp). Same filter schema as master.eq (peak / highshelf / lowshelf / highpass / lowpass)
      {"type": "peak", "hz": 5000, "q": 1.2, "db": -2.0}  // e.g. cut drum cymbals here without touching guitar/vocal presence
    ],
    "comp_preset": "comp_drum_bus_gentle",  // optional, stereo-linked bus compressor preset (generate-config proposes comp_drum_bus_gentle for drums; remove it if the bus needs no compression). comp_drum_bus is 4:1 and much heavier.
    "saturation": {"drive": 0.3},   // optional tape saturation (symmetric tanh). GUARDED: relevance check refuses if bus crest < 8 dB or LRA < 4 LU; pass {"drive": 0.3, "force": true} to override.
    "parent_bus": null         // routes into this parent bus
  },
  "guitar": {
    "auto_trim_db": -4.2,
    "volume_db": -3.0,
    "pan": 0.0,
    "saturation": {"drive": 0.25},
    "reverb_send": {"preset": "guitar_room", "wet": 0.15},  // bus-level reverb send
    "parent_bus": null
  }
},
"master": {
  // PREMASTER MODE (default, recommended). render_mix produces a CLEAN
  // mix.wav handoff to mastering. NO limiter / LUFS-norm / clipper / M/S
  // baked in — those are mastering's job. See "Mix vs master separation"
  // in docs/knowledge.md.
  "premaster_mode": true,
  "peak_target_dbfs": -3.0,    // peak normalize the sum to this target

  // Optional glue + EQ are still allowed in premaster mode (subtle bus
  // shaping is mix-phase work, not mastering):
  "comp": {                    // gentle master glue compressor (1-2 dB GR)
    "threshold_db": -10.0,
    "ratio": 1.5,
    "attack_ms": 30.0,
    "release_ms": 300.0,
    "makeup_db": 0.0
  },
  "eq": [                      // subtle tonal shaping (NOT loudness-focused)
    {"type": "highpass", "hz": 30},
    {"type": "peak", "hz": 2500, "q": 1.0, "db": 0.5}
  ]
  // The following keys are IGNORED in premaster mode (warning logged) —
  // they belong to master_mix.py:
  //   "lufs_target", "true_peak_dbfs", "clipper", "ms"
  // To use the legacy combined mix+master chain set:
  //   "premaster_mode": false
},
"reverb_buses": {              // optional: shared reverb buses (vocal mixing standard)
  "vocal_plate": {
    "preset": "vocal_plate",   // any preset from apply_reverb.PRESETS
    "wet": 1.0,                // wet level inside the reverb bus (1.0 = pure wet for send/return)
    "return_volume_db": -8.0,  // how loud the reverb return enters the master sum
    "return_pan": 0.0
  },
  "vocal_hall": {
    "preset": "vocal_hall_wide",
    "wet": 1.0,
    "return_volume_db": -12.0,
    "return_pan": 0.0
  }
}
// Per-track sends to reverb buses are declared on the track entry:
//   {"name": "LEAD VOX", "bus": "vocal_lead", "reverb_sends": [
//      {"bus": "vocal_plate", "level_db": -6},
//      {"bus": "vocal_hall",  "level_db": -18}
//   ]}
//
// Per-track polarity flip — invert the signal before summing into the bus.
// Use when a track is recorded anti-phase relative to its bus mates (most
// commonly: BASS DI PEDAL relative to BASS DI CLEAN when the pedal chain
// inverts polarity). Detection: see the "Bass DI PEDAL polarity inversion"
// rule below.
//   {"name": "BASS DI PEDAL", "bus": "bass", "polarity_flip": true, ...}
```

Bus processing order: volume → pan → **eq (per-bus, zero-phase)** → comp_preset → saturation → parallel_saturation (guarded) → reverb_send (per-bus insert)
Vocal chain order (per-stem, 2026 best practice): subtractive EQ (cuts) → compression → **de-esser (after comp on purpose — comp amplifies sibilance)** → additive EQ (boosts) → [pitch correction] → saturation → reverb sends to shared `reverb_buses`
Master processing order, premaster mode (default): sum buses → glue comp (stereo-linked) → EQ → peak normalize to `peak_target_dbfs`.
Master processing order, legacy `premaster_mode: false`: sum buses → glue comp → clipper (guarded) → M/S (guarded) → EQ → LUFS norm → true-peak brickwall limiter (8x-verified)

## mix_report.json — what render_mix writes back

Every `render_mix --render` writes `mix_report.json` next to `mix.wav`. The
fields are the agent's primary feedback on whether the mix is healthy. The
example below is a legacy-mode render (`after_lufs_norm` / `after_limiter` are
populated); in premaster mode those are null, `limiter` is null, and
`integrated_lufs` is whatever the peak-normalized premaster measures:

```json
{
  "integrated_lufs": -14.0,
  "true_peak_dbtp": -3.44,
  "sample_peak_dbfs": -3.68,
  "bus_peaks": {
    "drums": {
      "sum_in": 4.5,           // peak after summing all drum tracks into the bus
      "after_vol_pan": -3.5,   // after bus volume_db + pan
      "after_eq": null,        // null = stage skipped (no per-bus EQ on this bus)
      "after_comp": -3.5,      // after comp_preset
      "after_sat": null,
      "after_parallel_sat": null,
      "after_reverb": null,
      "final": -3.5,           // bus output sample peak
      "true_peak_final": -3.5, // 4x-oversampled true peak on bus output
      "verdict": "[WARN]"      // worst of sample+TP against -6/-1 dBFS thresholds
    }
  },
  "master_peaks": {
    "sum_in": 0.5,              // master sum of all top-level buses (pre-process)
    "after_comp": -0.1, "after_clipper": null, "after_ms": null,
    "after_eq": 0.1, "after_lufs_norm": 2.5, "after_limiter": -3.5,
    "final_sample_peak": -3.5, "final_true_peak": -3.4,
    "verdict": "[WARN]"
  },
  "phase_warnings": [
    {
      "bus": "bass",
      "track_a": "BASS DI CLEAN",
      "track_b": "BASS DI PEDAL",
      "correlation": -0.72,    // negative = anti-correlated (cancellation / polarity)
      "overlap_sec": 312,      // seconds both tracks are simultaneously active
      "kind": "anti-correlated (cancellation / polarity)"
    }
  ]
}
```

Verdict thresholds (DAW-style, applied identically by `_peak_verdict`
in `render_mix.py` and `_headroom_verdict` in `analyze.py` to buses, stems and
the legacy master; the premaster `master_peaks.verdict` uses its own bands so the
default -3 dBFS handoff reads `[OK]`: `[OK]` <= -2, `[WARN]` -2..-1, `[CLIP]` > -1
on the worse of sample and true peak):

- `[OK]` — peak < -6 dBFS AND true peak < -6 dBTP
- `[WARN]` — -6 ≤ peak < -1 dBFS OR -6 ≤ true peak < -1 dBTP
- `[CLIP]` — legacy warning label for peak >= -1 dBFS or true peak >= -1 dBTP.
  This threshold is not proof of clipping; floating-point buses may exceed
  0 dBFS without clipping. Inspect source/export samples and nonlinear stages.

`phase_warnings` fires for any pair of active tracks on the same bus with
`|corr| ≥ 0.4` over their mutual activity window. Most warnings are
expected multi-mic patterns (overhead L/R, kick-in + kick-sub, guitar
amp + cab on the same DI). The interesting case is two tracks of the
same instrument signal split between paths (e.g. DI clean + pedal chain
DI of a bass) — `audit_session.py` cannot see those because the source
files differ. Compare polarity and timing alternatives against the original
blend in context. Use `align_phase.py` only for a supported alignment hypothesis;
a large absolute correlation alone does not justify changing timing.

## Workflow

1. Validate inputs and preserve the DAW edit or consolidated-stem timeline.
2. Assemble at original levels. Level clips only to correct identified accidental
   recording-level differences; link related microphone/stereo decisions.
3. Analyze, then establish balance and pan. EQ, dynamics, alignment and effects
   are optional interventions with specific hypotheses, not mandatory stages.
4. Render and inspect `mix_health`. Technical failures must be resolved; LUFS,
   LRA, width, masking and tonal targets are advisory. Compare matched excerpts
   and obtain listening feedback before treating the mix as approved.
5. Master to the requested delivery profile. Preserve creative bus processing
   where intended and avoid accidental duplicate limiting. Do not automatically
   generate one loudness-normalized master for every streaming service.
6. Run `master_health` on the exported WAV. Verify file format, measured peaks,
   and any contractual loudness requirement. Read `loudness_target_met` from the
   mastering report; a missed preference is not permission to overcompress.
7. Health tools do not encode codecs. If required, use an available encoder,
   decode and measure the actual result, and arrange codec listening review.
   An 8x waveform peak estimate is not an encoder simulation or codec audition.
8. Build and validate recall. Run `tools/review_delivery.py` with the agreed
   ceiling, format requirements, and actual listening records. Full-song human
   approval must match the export hash; a green style score never grants it.

When investigating a click, compare the source, assembled stem, processed stem,
contributing buses, and final export at the same timestamp. A peak or correlation
threshold identifies a candidate cause, not a diagnosis. Preserve source material
and audition the smallest corrective change.

## Analysis interpretation — what to say after reading analysis.json

After reading analysis, always give a recommendation. The recommendation may be "no action needed" —
that is a valid and useful answer. Never leave analysis results without a verdict.

Read all of these fields from analysis.json and comment on each that is outside normal range:

| Field | Location in JSON | Normal range (rock mix stem) | Action if outside range |
|---|---|---|---|
| integrated_lufs | loudness.integrated_lufs | -22 to -14 LUFS (stem) | Too hot: reduce pre-clip gain. Too quiet: check assembly or delivery stage. null = not measurable (silent/too short). |
| headroom_verdict | loudness.headroom_verdict | `[OK]` (peak < -6 dBFS) | `[WARN]` (-6 to -1 dBFS) is acceptable on processed stems; `[CLIP]` (≥ -1 dBFS or TP ≥ -1 dBTP) is a warning label, not proof of clipping: intermediate files are float. Check whether a later fixed-point export or nonlinear stage will be driven by it before changing gain. |
| loudness_range_lu | loudness.loudness_range_lu | 5–16 LU (stem), 3–12 LU (mix) | < 3 LU: over-compressed; > 20 LU: may need compression before mix |
| crest_factor_db | loudness.crest_factor_db | 10–18 dB (typical) | Peak and RMS of the loudest channel. < 8 dB: heavily limited. > 20 dB: very dynamic — compression is one option, not a requirement |
| stereo.balance_db | stereo.balance_db | ±1.5 dB | > ±3 dB: strong imbalance — check panning or channel assignment. null = one channel silent |
| stereo.lr_correlation | stereo.lr_correlation | +0.6 to +0.95 | Negative: out-of-phase (check mono compat). < 0.4: very wide. null = one channel silent |
| stereo.ms_width_ratio | stereo.ms_width_ratio | 0.1–0.4 | < 0.05: near-mono. > 0.6: very wide (check mono compat) |
| transient_density_per_sec | transient_density_per_sec | varies by instrument | Compare sections — sudden jumps indicate uneven playing |
| spectral_centroid_hz | spectral_centroid_hz | varies by instrument | Use to confirm EQ changes had the expected effect |
| transient_profile.transient_prominence_db | transient_profile.transient_prominence_db | > 8 dB (percussive) | Kick/snare/tom: < 4 dB → Attack+ may help. Distorted guitar / fingerstyle bass: always low → ignore. |
| transient_profile.transient_prominence_std_db | transient_profile.transient_prominence_std_db | < half of mean | std > mean on ANY instrument → uneven playing. Primary unevenness indicator for slap bass. Suggests compression. |
| transient_profile.decay_time_ms | transient_profile.decay_time_ms | kick < 150ms, snare < 100ms | Percussive only. Long decay → Sustain- may tighten. Overhead/room mics: ignore. |
| hum_detection.hum_detected | hum_detection.hum_detected | false | Requires narrow persistent lines within ±0.5 Hz of at least 2 mains harmonics in the quietest passages ("Not analysed" under 8 s of audio). true: audition candidate notches from hum_detection.harmonics in a quiet passage before applying them |
| frequency_bands_crest_db.* | frequency_bands_crest_db (per band, mono downmix) | 8-15 dB typical | < 6 dB: band is already squashed (avoid more multiband / parallel sat there). > 18 dB: band is very dynamic (multiband is one option). |
| pumping.pumping_detected | pumping.pumping_detected | false | true is a **suspicion, not a verdict** — manually disambiguate: is it (a) comp/multiband/clipper artifact, or (b) musical strumming/groove? See "Interpreting pumping_detected" below. |
| vocal.sibilance.peak_db | vocal.sibilance.peak_db (vocals only) | -25 to -35 dBFS typical | > -25 dBFS: sibilance may be loud — listen; a de-esser after the comp step is a candidate. < -35: little sibilance content, de-esser will skip via relevance_check. |
| vocal.pitch.cents_std | vocal.pitch.cents_std (vocals only) | < 15 cents (well intoned) | RMS cents from per-note targets after removing the global tuning offset (`tuning_offset_cents`); vibrato and tracker jitter are smoothed out; random notes would give about 29. 15-25: noticeable — listen to the flagged phrases. > 25: surface to the user; pitch correction is their choice. Also read `fraction_over_25_cents` and `cents_mad`. |
| vocal.plosive.events_per_minute | vocal.plosive.events_per_minute (`events_count` is the total) | 0-3 per minute | Low-frequency bursts within 12 dB of the stem's loud level. > 10 per minute: many p/b bursts — audition a vocal HP (80-120 Hz) or clip-gain on the worst ones. |

**Rule:** if all metrics are within range and the spectrogram looks normal for the instrument type,
say so explicitly ("analysis looks clean — no action needed before next processing step").
Do not invent problems. Do not recommend processing without a specific reason from the data.

### Interpreting pumping_detected

The detector measures envelope modulation and can flag strumming, repeating
hits, vocal phrasing, or compression artifacts. It cannot distinguish those
from statistics alone. A new flag after processing is a threshold crossing,
not proof that processing introduced an audible defect.

Compare the same passage before/after at matched loudness. Examine the groove,
active notes, gain envelope, and release behavior. Tempo-related modulation may
be intentional or compressor-driven; the tempo match alone settles neither.
Use direct listening or actual human feedback to decide whether to keep, soften,
or revert the change. If listening is unavailable, report the candidate cause
and the excerpt to review without declaring an artifact or a clean result.

## Analysis tool decision tree - when to run what

Use analysis to answer a specific question. Numerical thresholds generate
hypotheses; they do not authorize automatic corrective processing.

| Trigger | Check | Decision |
|---|---|---|
| New session | Source/clip audit and assembled-stem analysis | Confirm layout, channel relationships, usable inputs and duplicate candidates. |
| User supplies references | Loudness-matched comparison of corresponding passages | Establish intended tone and dynamics; account for arrangement differences. |
| Suspected masking, sibilance, pitch or pumping issue | Relevant analysis plus audition | Process only an identified audible problem; preserve intentional effects. |
| Processing changed audio | Before/after measurement and matched audition | Keep, revise or bypass the treatment according to the stated goal. |
| Mix rendered | `mix_health.py` and the exported audio | Resolve technical failures; treat musical metrics as advisory. |
| Master exported | `master_health.py` with the agreed format | Verify file properties, peaks and any contracted loudness requirement. |
| Health report is yellow or red | Inspect the named check | Fix technical violations; investigate artistic differences without chasing an all-green score. |

Preserve an approved bus balance. Recompute auto-trim only when intentional
recalibration is requested, since it changes relative levels. Export bus stems
when needed for deliverables or diagnostics; their levels are preserved.

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

Use a user-provided or otherwise authorized reference when available. Record
its exact version, comparison passage, and what it demonstrates (for example,
drum impact or vocal depth). Research articles cannot replace audio references.
If no reference is available, state that limitation; do not invent one.

`tools/compare_reference.py` measures loudness and spectral differences.
Compare similar musical sections at matched loudness. Differences can come
from arrangement, tuning, instrumentation, or production intent. Its suggested
EQ filters are hypotheses: do not apply inverse spectral matching automatically.
A quieter mix is not inherently worse, and playback normalization does not
establish an artistic loudness target.

Use `tools/detect_masking.py` to locate possible overlap, then check simultaneous
activity and the actual mix balance. Its normalized stem comparison changes
relative levels and cannot by itself establish audible masking or required EQ.

Use `tools/prepare_audition.py` for audible A/B files. Compare a minimally
processed baseline and each candidate at a common loudness. Inspect kick/snare
attacks, bass articulation, vocal consonants, cymbal decay, mono translation,
and quiet-to-dense transitions. Collect focused feedback on unresolved choices.

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

Report the current state clearly: draft, technically checked with listening
pending, revision requested, or approved for the specified delivery. Include:

- Actual source/take decisions, applied processing, and retained preferences.
- Links to the exact rendered versions and any matched audition excerpts.
- Measured results and contractual checks, separately from artistic judgments.
- Direct audition performed, human feedback and its scope, and unresolved issues.
- Recall validation performed (dry-run or actual replay), including engine-version
  mismatches. New tool code invalidates old engine hashes; do not rewrite old
  manifests to pretend a prior render used the new engine.

Do not automatically create a narrative document unless the user requested one.
Keep machine-readable processing, audition, and delivery-review records with the
session. Show test results as software verification, never as a sound-quality score.

## Creative processing decision rules

Start from a stated listening problem or artistic intention. Relevance checks
are conservative tool safeguards, not evidence that an effect improves music.
A documented override may be justified within existing user authorization;
never force a process solely to pass a score or a downstream tool's guard.

Compare before/after metrics and matched audio for each consequential change.
A new pumping flag means the detector crossed its threshold; it does not prove
an audible artifact. Check whether the signal follows the musical pulse and
whether the change improved the intended result. Without listening, describe
candidate causes and keep the choice provisional.

Keep chains as simple as the job permits. There is no universal four-process
limit, and processing should not be added to satisfy a genre recipe. If further
processing is needed, review the earlier choices before stacking another effect.
Low LRA alone cannot identify an upstream compressor as the cause. Do not
change drum compression just to make a clipper relevance check pass.

## Production finishing review

Follow the production review below when the user asks for a more
modern, finished, or impactful record. These are conditional experiments, not a
mandatory processing chain or a promise of commercial success.

1. Establish the intended attention and contrast in each section. Use actual
   timestamps; mark section labels inferred from activity as provisional.
2. Preserve the approved lead take and double preference. Review phrase levels,
   consonants, breaths, and effect tails. Bounded energy-derived gain rides are
   candidates, not a substitute for word-by-word listening or pitch judgment.
   Match overall lead loudness before comparing a consistency treatment.
3. Review drum microphone relationships before adding processing. Try a quiet
   parallel shell bus for body while retaining the original attacks. Do not
   automatically include overheads/cymbals in a heavily compressed parallel path.
   Sample reinforcement requires an identified need, suitable cleared or own
   samples, velocity handling, and phase/timing review. It is never obligatory.
4. Review kick/bass and guitar/vocal overlap only where both sources are active.
   Prefer a small, bounded dynamic EQ experiment over automatic permanent cuts.
   Preserve sidechain timing and link stereo control. A silent trigger must
   leave the signal unchanged; check detector activity and actual cut depth.
5. Shape space by section. Duck the wet vocal return if it interferes with active
   words, retaining the dry vocal and allowing tails to emerge in gaps. Compare
   at the same dry/wet balance before claiming better intelligibility.
6. Use a few deliberate details: selected phrase-end delay throws, a build in
   room energy, or a swell from existing material. Keep conspicuous options
   separate from the main revision until reviewed. Do not claim tempo sync
   without verified tempo, meter, and placement.
7. Check exposed edits, sustained notes, double consonants, drum fills, and the
   ending. Do not quantize, tune, replace hits, or remove breaths solely because
   the user named a genre. Report which requested refinements remain unverified
   when direct listening is unavailable.
8. Export the revised mix and master in a new version folder, with source hashes,
   settings, section decisions, and matched before/after excerpts. Technical
   checks and musical approval remain separate. Never overwrite an approved
   source or inherit full-song approval from an earlier balance comment.

### Room and depth review

For an assertive production revision, evaluate instrument character and section
contrast before adding more natural room. Parallel snare distortion, a shaped
snare tail, and a separate bass midrange drive path are candidate treatments.
Use `apply_saturation.py --oversample 4` (or 8) for stronger nonlinear processing;
`--input-gain-db` changes drive while the wet path is RMS matched. Check actual
alias reduction and peak behavior; oversampling does not establish sound quality.
Keep an unchanged baseline and compare shortened snare tails or new textures as
separate options. Do not treat absence of a reference song as a reason to stop.

- Compare recorded room microphones before choosing synthetic ambience. Preserve
  acoustic arrival delays by default; waveform correlation alone does not justify
  shifting a room microphone or changing its polarity. Check the combined kit in
  stereo and mono, including low-frequency weight and cymbal decay.
- Compare natural-room and short shared-room alternatives separately. Use
  instrument sends and wet returns for selective depth; do not automatically add
  room reverb across the stereo master. Keep the approved dry-vocal balance.
- Treat room filtering, compression, ducking, and section rides as conditional
  choices. Document return levels and audition attacks, consonants, and exposed
  tails at matched loudness. Increased width is not proof of better depth.
- Preset names and Freeverb room_size values do not establish a physical room,
  authentic plate algorithm, or decay in seconds. Measure an impulse response
  when a decay claim matters, reporting the estimator and filtering. Distinguish
  algorithmic rooms from captured spaces; record provenance for external IRs.

## Reproducibility - mix_chain.json

New file-processing calls save `<output-stem>.operation.json` next to their WAV.
These records (schema 2) contain complete callable arguments with resolved
defaults (e.g. the actual `target_lufs`), data `dependencies` (inputs, sources of
the recorded track, presets), `engine` hashes of the tool modules actually loaded,
and the decoded-audio hash (samples and format, excluding container timestamps). Repeated tools retain separate records. Keep these files with the
session; the traditional `eq_report.json` aliases remain summaries of the latest
operation only.

Run `tools/build_chain.py output/<session>` then
`tools/replay_chain.py output/<session>/mix_chain.json --dry-run`.
Replay validates supported calls and dependencies before processing, stops on the
first error, and checks recorded output hashes. A data-dependency mismatch is an
error; an engine (tool code) mismatch is a warning and the outputs count as
unverified. Recorded operations replay into a `.replay-*` scratch folder and
replace the WAV only on a hash match; on a mismatch the original audio and its
operation record are kept. Legacy steps and the final render overwrite.
`--stem NAME` replays that stem without rendering the full mix. Schema 1 records
remain readable (their tool hashes count as engine).

`build_chain.py` turns reports without an operation record into `unrecorded`
steps with chain-level `warnings` and `verified_operations: false`; replay rejects
them with a request to rerun that tool. Legacy replay is rejected by default; `--allow-legacy` explicitly requests a
best-effort run without equivalence guarantees. Legacy reports may omit
parameters or whole operations. They cannot recover
information that was never recorded. Unsupported or incomplete chains require
rebuilding from the original inputs with explicit settings. Do not claim a legacy
recall is bit-identical merely because its commands completed.

## Ground rules

- State observations, uncertainty and the intended audible improvement.
- Preserve the user's artistic decisions and existing approvals.
- Diagnose missing sources, nonfinite samples, invalid routing and export clipping.
- Check rendered mixes and exported masters with the relevant health tool.
  Resolve technical failures against the agreed specification. LUFS, LRA,
  spectral similarity, phase and punch heuristics do not replace listening.
- Export bus stems when requested or needed for diagnosis; they preserve mix
  levels and are not automatically normalized to -18 LUFS.
- Keep a reproducible operation record. Do not describe unrecorded changes as
  verified recall or legacy reports as exact reconstruction.
- Add knowledge only with source, scope, evidence type and verification date.
  Internet research must not silently modify an approved session's decisions.
