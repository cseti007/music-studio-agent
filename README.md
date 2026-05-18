# music-mix-agent

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
git clone https://github.com/your-username/music-mix-agent.git
cd music-mix-agent

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
conda run -n music-mix-agent pytest tests/ -v
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
