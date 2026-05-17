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
apply_gain --per-clip -- normalize clips + assemble full-length stems
analyze              -- LUFS, transients, spectrum, stereo, hum, pumping,
                        per-band crest, true peak (4x oversampled)
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
mix_health           -- green/yellow/red scorecard. Run after every render.
                        REQUIRED to declare a mix delivery-ready.
```

All tools are standalone CLI scripts — run them in any order, re-run individual steps,
or skip stages that are not needed for your session. Make-it-hit tools are
gated by `relevance_check`: each one analyses its input first and refuses
to write audio when the input data doesn't justify the processing.

---

## Project structure

```
tools/               CLI processing tools (one file per processor)
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
