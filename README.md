# music-studio-agent

AI-assisted multi-track mixing pipeline. An LLM agent analyzes your recorded stems,
interprets the data, and applies processing — EQ, compression, gating, reverb, saturation,
delay, transient shaping, amp simulation — via Python CLI tools. You stay in control:
the agent proposes, you approve.

Works with any LLM: Claude, ChatGPT, Gemini, local models via Ollama, etc.

---

## Requirements

- Python 3.11+
- All dependencies in `requirements.txt` (includes `pedalboard`, which requires a C++ compiler on Linux/macOS)

## Installation

```bash
git clone https://github.com/cseti007/music-studio-agent.git
cd music-studio-agent

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verify:

```bash
python3 -c "import pedalboard, librosa, pyloudnorm; print('OK')"
```

---

## Usage

### Option A — Claude Code (recommended)

Claude Code reads `CLAUDE.md` automatically as the agent system prompt.

```bash
# in the project root
claude
```

The agent will ask which session you are working on and guide you through the full pipeline.

### Option B — Any other LLM (ChatGPT, Gemini, Ollama, etc.)

Pass the contents of `CLAUDE.md` to your agent as the system prompt or initial instructions.
It contains everything the agent needs to know: available tools, workflow logic, analysis rules,
and ground rules. The agent should take it from there.

---

## Preparing your recording session

The pipeline reads a DAW session file directly — no stem-bouncing required.
But a few things need to be set right *before* you hand the session to the
agent so it can do its job.

### DO before handoff

**1. Name your tracks clearly.** The agent infers each track's bus
assignment from its name. Use instrument + microphone position:

```
KICK IN, KICK OUT, KICK SUB
SN TOP, SN BOTTOM
HIHAT, RIDE, CRASH
RACK TOM 1, RACK TOM 2, FLOOR TOM
OH AEA L, OH AEA R, OH U87 L, OH U87 R
ROOM CLOSE L, ROOM CLOSE R, ROOM FAR L, ROOM FAR R
BASS DI, BASS DI PEDAL
GTR <player> FENDER, GTR <player> ORANGE, GTR <player> DI, GTR <player> 57
LEAD VOX, LEAD VOCAL, VOX, VOC
BG VOX L, BG VOX R, BACKING VOCAL, HARMONY HIGH, HARMONY LOW
DOUBLE LEAD, AD-LIB, WHISPER, TALK
```

Unrecognised track names get flagged and the agent will ask which bus they
belong to before continuing.

**2. Do your editorial cuts.** Trim out false starts, talkback, miscued
takes, and anything else you don't want in the final mix. Clip boundaries
become part of the session metadata and get assembled in order.

**3. Comp (best-take assembly) inside the DAW.** If you want to stitch
the strongest sections from several takes into one ideal take, do that
in your DAW. The pipeline reads clip positions as they are — it does
not perform clip-level comping.

**4. For A/B alternate takes**, put them on *separate* tracks (e.g.
`BASS DI` and `BASS DI Take2`). Same-track stacked clips get assembled
sequentially. Separate-track lets you switch via `mix_config.json` →
`"active": false` later.

**5. Stereo pairs**: a stereo mic (OH AEA, ROOM CLOSE, etc.) can live on
one stereo track *or* as two mono tracks with `.L` / `.R` suffixes —
both work. Don't mix the two conventions for the same mic.

**6. Set the session tempo and time signature.** The agent estimates BPM
with librosa, but a session-tempo-correct project converges faster (and
unlocks BPM-synced delays / reverbs).

**7. Pro Tools**: just save the `.ptx`. **Ableton**: `File → Collect All
and Save` after final save. This collects every used sample into
`<name> Project/Samples/` next to the `.als`. Without this, the parser
sees absolute paths that won't resolve on the target machine.

### DON'T do before handoff

| Don't | Why |
|---|---|
| **Pre-apply EQ on tracks** | The agent runs `apply_eq` with instrument-specific presets (`kick_in`, `snare_top`, `bass_di`, etc.). Pre-EQ stacks invisibly and can't be undone. |
| **Pre-apply compression / limiting / gating** | The agent runs `apply_compression`, `apply_gate` with reasoning. Pre-comped tracks have damaged crest factor and trigger false pumping alarms. |
| **Pre-apply reverb / delay / saturation** | The agent runs `apply_reverb`, `apply_delay`, `apply_saturation`. Wet signal isn't reversible. |
| **Normalize tracks or apply gain rides** | The agent's `apply_gain --per-clip` normalises each clip to a consistent LUFS target. Pre-normalised material defeats this. |
| **Apply master-bus EQ / limiting** | The agent's `master_mix.py` handles format-specific mastering (Spotify, Apple, CD, etc.). |
| **Apply pitch correction / Melodyne / Auto-Tune** | Vocal toolkit is on the backlog — when shipped, the agent will handle this. Manual edits constrain its choices. |
| **Enable Ableton's auto-warp on a fixed-tempo recording** | Turn warp off on tracks that were recorded to click. Warp markers can misalign multi-mic drum takes by milliseconds. |

### Special cases

**Click / scratch track**: keep it in the session (it helps the agent
understand tempo), but name it clearly (`CLICK`, `GUIDE`, `METRONOME`, etc.).
The agent will set `active: false` on it during the render — otherwise
a click track that extends past the song's end will lengthen the mix
with silence.

**Drum mics with bleed**: leave the bleed in. The agent's `apply_gate`
handles drum bleed control with instrument-specific presets, and the
`align_phase` step phase-aligns multi-mic captures sub-sample. Manual
fade-outs between hits defeat both.

**Vocal recordings**: out of scope for now (no vocal toolkit yet —
backlog item #4). You can still hand the session to the agent; it will
detect vocal tracks, deactivate them in `mix_config`, and ship the
instrumental.

### Folder layout to hand off

After `Collect All and Save` (Ableton) or saving the `.ptx` (Pro Tools),
the folder you hand the agent should look like:

```
sessions/<songname>/
├── <songname>.als            ← Ableton session file (or .ptx for Pro Tools)
├── Samples/
│   ├── Recorded/             ← Ableton-recorded audio
│   ├── Imported/             ← External samples used in the session
│   └── ...
└── (optional) reference.wav  ← A "make it sound like this" reference track
```

Copy this directory to the machine running music-studio-agent (USB, scp,
rsync — whatever's convenient). Then tell the agent:

> "New session at `~/sessions/songname/songname.als`. Modern rock. Goal: Spotify-ready master."

(Substitute genre and goal as appropriate.) The agent takes it from there.

### What the agent will ask before starting

Before the pipeline runs, the agent will ask 2-3 things you should be
ready for:

1. **Genre / style** — for `--style modern_rock` (or `classic_rock`,
   `pop`, `hip_hop`, `jazz_acoustic`). Sets genre-appropriate bus
   volume defaults.
2. **Which takes/mics to keep** if there are duplicate-source-file
   groups (the `audit_session.py` flag) or alternate takes on separate
   tracks.
3. **Goal** — streaming delivery, demo, mix-health gate against the
   reference track, etc. Determines target LUFS and which masters get
   rendered.

You don't have to know these answers in advance — the agent shows what
it sees and explains the trade-offs.

### "Less is more" — the cardinal rule

The less you process pre-handoff, the better the pipeline can do its
job and the more reproducible the result is. The `mix_chain.json` recall
sheet captures every step the pipeline takes — but only steps that are
*part of the pipeline*. Pre-processing inside your DAW isn't in the
chain. If you want to rebuild the mix in three months on a different
machine using only the raw DAW session + `mix_chain.json`, that only
works if the DAW session is genuinely raw.

---

## Workflow overview

```
stems (WAV files)
    |
    v
parse_session        -- parse DAW session (.ptx / .als) into session.json
audit_session        -- detect tracks sharing identical source files
                        (phase-coherent duplicates) at session start
apply_gain --per-clip -- normalize clips + assemble full-length stems
analyze              -- LUFS, LRA, crest, transients, spectrum, stereo,
                        hum, pumping, per-band crest, true peak (4x
                        oversampled), onsets[], tempo_bpm, estimated_key,
                        envelopes (RMS / LUFS short-term / spectral flux)
batch_analyze        -- parallel multiprocessing wrapper around analyze
                        (5-6x faster than the serial loop on 8 cores)
level_notes          -- per-note volume leveling on a target time range
                        (opt-in: fixes uneven slap/finger bass takes)
detect_masking       -- find frequency conflicts between stems (time-gated)
align_phase          -- phase-align drum mics to kick reference (sub-sample)
apply_eq             -- notch hum, carve frequencies, instrument presets
                        (minimum-phase by default, zero-phase for mastering)
apply_compression    -- dynamics control, parallel + sidechain options
apply_gate           -- drum bleed control
apply_transient      -- attack/sustain shaping for percussive stems
apply_amp            -- tube amp + cabinet sim for bass DI
apply_saturation     -- tape/tube/clipper harmonic saturation
apply_reverb         -- algorithmic (Freeverb) OR convolution (--ir <IR.wav>)
apply_delay          -- slapback, echo, ping-pong

vocal pipeline       -- vocal-stem-aware chain (best-practice order):
                        subtractive EQ → comp → DE-ESSER → additive EQ
                        → [pitch correction] → reverb sends:
  apply_deesser      -- frequency-specific sidechain (5-8 kHz detect,
                        full-band gain reduction). Runs AFTER the comp on
                        purpose — the comp amplifies sibilance, the de-esser
                        catches it. Presets per voice type.
  apply_pitch_correct-- librosa.pyin pitch detect + PSOLA shift + scale
                        quantisation. 7 modes (major, minor, dorian, etc.),
                        strength blends original→quantised.
  (vocal EQ / comp / reverb presets are part of the standard preset library)

make-it-hit tools    -- guarded by data-driven relevance_check (won't fire
                        if the input doesn't justify the trade-off):
  apply_subharm      -- sub-bass harmonic synthesizer (small-speaker translation)
  apply_haas         -- Haas stereo widener (mono mic-pair detection)
  apply_exciter      -- HF harmonic generator (Aphex-style, refuses on bass/kick)
  apply_multiband_comp -- 3-band Linkwitz-Riley split + per-band compressor

compare_reference    -- spectral + LUFS comparison against a reference track,
                        optional --apply for auto inverse-delta EQ chain
render_mix           -- sum to stereo, bus routing, master chain
                        (master.clipper / master.ms / parallel_sat — all guarded)
bus_balance          -- per-bus LUFS contribution report (opt-in
                        diagnostic — "is bass too loud vs drums?" objectively)
mix_health           -- green/yellow/red scorecard. REQUIRED before mastering.

  ── mix phase ends here ── master phase begins ──

master_mix           -- stereo mix → mastered.wav per delivery format.
                        7 format presets (spotify -14, apple -16, youtube,
                        tidal, cd 16-bit, vinyl_pre, broadcast -23) and 6
                        chain presets (gentle, modern_rock, modern_rock_mb,
                        pop, hip_hop, transparent). --all-formats batch.
                        modern_rock_mb adds 3-band multiband + M/S; pop
                        adds bright EQ + width 1.1; transparent does
                        only LUFS norm + ISP limit (use on refmatched
                        mixes to avoid tonal compounding).
master_health        -- master scorecard: format conformance (LUFS, true
                        peak, 8x codec-ISP estimate), per-band phase
                        coherence (sub mono check, top wide), M/S width
                        profile, punch index, compression-history detect,
                        reference-deck comparison. REQUIRED per format.

  ── reference-free style grading (optional, no ref track needed) ──

style_check          -- grade a mix against one of 5 built-in genre
                        profiles (modern_rock, classic_rock, pop,
                        hip_hop, jazz_acoustic). 0-100 score, traffic-
                        light verdict, EQ recommendations.

  ── reproducibility (run after the session is done) ──

build_chain          -- aggregate every per-stem *_report.json into
                        a single mix_chain.json recall sheet
replay_chain         -- re-run every step from a mix_chain.json,
                        rebuilds the entire mix from scratch
```

All tools are standalone CLI scripts — run them in any order, re-run individual steps,
or skip stages that are not needed for your session. Make-it-hit tools are
gated by `relevance_check`: each one analyses its input first and refuses
to write audio when the input data doesn't justify the processing.

---

## Project structure

```
tools/               CLI processing tools (one file per processor)
tools/presets/       Instrument-specific EQ / comp / amp / etc. preset JSONs
tools/style_profiles/ Genre profiles consumed by style_check.py
tools/irs/           Synthetic impulse-response pack for convolution reverb
docs/knowledge.md    Domain knowledge base (LUFS targets, instrument guidelines,
                     make-it-hit philosophy, pumping disambiguation)
tests/               Smoke tests (pytest) for DSP + relevance-check logic
config.toml          Pipeline configuration (LUFS targets, alignment settings)
CLAUDE.md            Agent system prompt (auto-loaded by Claude Code; paste manually for other LLMs)
output/              Generated during a session (excluded from git)
BACKLOG.md           Deferred feature ideas
CHANGELOG.md         Project history
```

---

## Tests

A minimal pytest suite covers the load-bearing DSP and relevance-check logic.
Run before committing changes that touch tools/:

```bash
conda run -n music-studio-agent pytest tests/ -v
# or with venv:
pytest tests/ -v
```

---

## Configuration

Edit `config.toml` to adjust global defaults:

```toml
[gain]
per_clip_target_lufs = -18.0     # clip gain normalization target
per_channel_preset = "stem"      # default delivery preset

[align]
max_delay_ms = 20.0              # phase alignment search range
```

---

## Supported DAW session formats

- Pro Tools (.ptx) — track names, clip regions, sample rate
- Ableton Live (.als) — audio tracks, clip names, BPM

For sessions without a DAW file, skip `parse_session` and run `analyze` + `apply_gain --per-channel`
directly on your pre-assembled stems.

---

## License

MIT
