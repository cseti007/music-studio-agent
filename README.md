# music-studio-agent

AI-assisted multi-track mixing pipeline. An LLM agent analyzes your recorded stems,
interprets the data, and applies processing — EQ, compression, gating, reverb, saturation,
delay, transient shaping, amp simulation — via Python CLI tools. You stay in control:
the agent proposes, you approve.

Works with any LLM: Claude, ChatGPT, Gemini, local models via Ollama, etc.

Rendered audio starts as a draft. Agents must disclose whether they can directly
audition it, compare changes at matched loudness, and preserve the scope of user
feedback. Health checks and style scores never certify musical quality.
`prepare_audition.py` creates comparison excerpts; `review_delivery.py` checks
the current file and its recorded human listening approval before delivery.
Shared instructions are in `CLAUDE.md`, with `AGENTS.md` as the agent entry point.

---

## Requirements

- Python 3.11+
- All dependencies in `requirements.txt` (`pedalboard>=0.9.25` provides the true-peak brickwall limiter; `psola` is needed only by `apply_pitch_correct.py`)
- Optional system tools: `ptftool` for Pro Tools `.ptx` parsing (path via `PTFTOOL_PATH`, default `/tmp/ptformat/ptftool`); `ffmpeg` for codec round-trip checks

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

Consolidated, time-aligned stems are the most reliable input. The DAW parser
extracts a limited audio clip layout; it does not reproduce plugins, automation,
or complete session playback. Check the parsed layout before assembly.

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
| **Pre-apply corrective EQ on tracks** | EQ in the pipeline is an optional, recorded intervention with a stated reason. Pre-EQ stacks invisibly and can't be undone. Intentional sound-design processing is fine if you say so. |
| **Pre-apply compression / limiting / gating** | Dynamics in the pipeline are optional and auditioned. Pre-comped tracks have reduced crest factor and confuse the pumping detector. |
| **Pre-apply reverb / delay / saturation** | Unless it is part of the sound you want, keep it out: wet signal isn't reversible. |
| **Normalize tracks** | `apply_gain --per-clip` assembles clips at their original levels by default and levels clips only on request (`--normalize`). Pre-normalised material hides the real level relationships. |
| **Apply master-bus EQ / limiting** | `master_mix.py` handles mastering to the requested delivery profile. |
| **Apply pitch correction / Melodyne / Auto-Tune** (unless that is the intended sound) | `apply_pitch_correct.py` exists but tuning is the artist's choice; manual edits constrain later choices. |
| **Enable Ableton's auto-warp on a fixed-tempo recording** | Turn warp off on tracks that were recorded to click. The parser accepts warped clips only when the warp is tempo-neutral and rejects tempo automation it cannot reproduce. |

### Special cases

**Click / scratch track**: keep it in the session (it helps the agent
understand tempo), but name it clearly (`CLICK`, `GUIDE`, `METRONOME`, etc.).
The agent will set `active: false` on it during the render — otherwise
a click track that extends past the song's end will lengthen the mix
with silence.

**Drum mics with bleed**: leave the bleed in. Gating (`apply_gate`) and
alignment (`align_phase`) are optional, auditioned choices; acoustic arrival
delays between mics are preserved by default. Manual fade-outs between hits
remove information the mix may need.

**Vocal recordings**: vocal tools exist (de-esser, pitch correction, vocal
presets, shared reverb buses). Whether vocals are in scope is decided per
project; tell the agent if the deliverable is an instrumental.

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

1. **Genre / style and references** — optional `--style` starting points
   (`classic_rock`, `hip_hop`, `jazz_acoustic`, `modern_rock`, `pop`,
   `punchy_modern_rock`, `tool_inspired`) set initial bus volumes/pans.
   They are project preferences, not genre standards; your references
   and listening decide.
2. **Which takes/mics to keep** if there are duplicate-source-file
   groups (the `audit_session.py` flag) or alternate takes on separate
   tracks.
3. **Goal** — streaming delivery, demo, vinyl, broadcast, etc. Determines
   which masters get rendered and any contractual loudness or peak limits.

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
audit_session        -- flag tracks sharing identical source files / clip
                        layouts (duplicate candidates to review)
apply_gain --per-clip -- assemble full-length stems at original levels
                        (--normalize and continuous take reconstruction
                        are opt-in editorial choices)
analyze              -- LUFS, LRA, crest, transients, spectrum, stereo,
                        hum (narrow mains lines in quiet passages), pumping
                        candidates, per-band crest, per-channel sample and
                        true peak, onsets[], tempo_bpm, estimated_key,
                        envelopes (RMS / LUFS short-term / spectral flux)
batch_analyze        -- parallel multiprocessing wrapper around analyze
                        (5-6x faster than the serial loop on 8 cores)
level_notes          -- per-note volume leveling on a target time range
                        (opt-in: fixes uneven slap/finger bass takes)
detect_masking       -- candidate frequency overlap between stems (band
                        power, co-activity gated; audition before cutting)
align_phase          -- optional delay/polarity alignment for a supported
                        hypothesis; refuses low-confidence estimates
apply_eq             -- notches, carving, instrument presets
                        (minimum-phase by default, zero-phase optional)
apply_dynamic_eq     -- bounded bell cut under a detector (self or sidechain)
apply_automation     -- explicit linked gain rides from a curve file
apply_compression    -- stereo-linked dynamics, parallel + sidechain options
apply_gate           -- drum bleed control
apply_transient      -- attack/sustain shaping for percussive stems
apply_amp            -- tube amp + cabinet sim for bass DI
apply_octaver        -- sub-octave generator for bass weight (pitch-shift
                        down + bandpass + mix-back; helps translation on
                        small speakers / phones / BT)
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

optional creative tools -- guarded by conservative relevance_check
                        safeguards; audition required either way:
  apply_subharm      -- 2nd/3rd harmonic synthesis of the sub fundamental
  apply_haas         -- Haas stereo widener (mono-compatibility caveat)
  apply_exciter      -- oversampled HF harmonic generator
  apply_multiband_comp -- 3-band Linkwitz-Riley split, linked per band

compare_reference    -- spectral + loudness comparison against a reference;
                        optional --apply bakes a merged, capped inverse EQ
                        (a hypothesis to audition, never automatic)
render_mix           -- sum to stereo, bus routing; premaster mode (default)
                        writes a float, peak-normalized mix.wav with no
                        limiter (legacy mode adds clipper / M/S / limiter)
bus_balance          -- per-bus LUFS/peak of the rendered stems (diagnostic)
mix_health           -- technical peak gate + advisory loudness, phase,
                        masking and dynamics measurements. Run before
                        mastering; listening review stays separate.
prepare_audition     -- loudness-matched before/after excerpts with hashes
find_clicks          -- click forensic: sweep the mix, or trace one
                        timestamp from source files through every stage

  ── mix phase ends here ── master phase begins ──

master_mix           -- stereo mix → master per requested delivery format.
                        True-peak brickwall limiter with an 8x-verified
                        ceiling (-1 dBTP, -2 dBTP above -14 LUFS).
                        7 format presets (spotify -14, apple -16, youtube,
                        tidal, cd 16-bit, vinyl_pre (no LUFS target, peak
                        -3 dBTP), broadcast -23) and 11
                        chain presets — base (gentle, modern_rock,
                        modern_rock_mb, pop, hip_hop, transparent) + spatial
                        family (modern_rock_spatial, _v9, _v10, _dark,
                        _noclip — sub-mono + side-band emphasis tuned for
                        modern prog metal; _dark drops top emphasis for
                        ear-fatigue cases; _noclip is a clipper-bypass
                        diagnostic). --all-formats only when several
                        deliverables were requested. modern_rock_mb
                        adds 3-band multiband + M/S; pop adds bright EQ +
                        width 1.1; transparent does only LUFS norm + the
                        true-peak limiter.
master_health        -- master scorecard: format conformance (LUFS, 4x/8x
                        waveform true-peak estimates; not a codec
                        simulation), per-band phase
                        coherence (sub mono check, top wide), M/S width
                        profile, punch index, compression-history detect,
                        reference-deck comparison. Run per exported format.
review_delivery      -- delivery gate: exact export properties plus scoped
                        human listening evidence tied to the file hash

  ── reference-free style similarity (optional) ──

style_check          -- similarity to one of 7 project style profiles.
                        Measures, never grades quality; loudness checks
                        are N/A on premaster input.

  ── reproducibility ──

build_chain          -- collect per-output .operation.json records (and
                        flag unrecorded legacy reports) into mix_chain.json
replay_chain         -- validate and re-run recorded operations; replaces
                        outputs only when the content hash matches
generate_irs         -- regenerate the synthetic IR pack (deterministic)
```

All tools are standalone CLI scripts — run them in any order, re-run individual steps,
or skip stages that are not needed for your session. The optional creative tools
run a `relevance_check` first and refuse to write audio (unless `--force`) when the
input clearly doesn't suit the processing; passing the check is not evidence the
effect improves the music.

---

## Project structure

```
tools/               CLI processing tools (one file per processor)
tools/presets/       Instrument-specific EQ / comp / amp / etc. preset JSONs
tools/style_profiles/ Genre profiles consumed by style_check.py
tools/irs/           Synthetic impulse-response pack for convolution reverb
docs/knowledge.md    Domain knowledge base with evidence levels (delivery specs,
                     instrument heuristics, pumping disambiguation)
tests/               pytest suite: DSP behavior, regressions, recall, workflow
config.toml          Pipeline configuration (read from the project root)
CLAUDE.md            Agent instructions (auto-loaded by Claude Code; paste manually for other LLMs)
AGENTS.md            Short entry point for other agent hosts
ruff.toml            Lint config (pyflakes-level checks, run in CI)
output/              Generated during a session (excluded from git)
```

---

## Tests

The pytest suite verifies software behavior (DSP math, regressions, recall,
listening-evidence workflow), not musical quality. Run before committing changes
that touch tools/:

```bash
conda run -n music-mix-agent pytest tests/ -v   # conda env name: music-mix-agent
# or with venv:
pytest tests/ -v
ruff check tools tests
```

---

## Configuration

Edit `config.toml` to adjust global defaults:

```toml
[gain]
per_clip_target_lufs = -18.0     # used only with apply_gain --normalize
per_channel_preset = "stem"      # default apply_gain --per-channel preset

[align]
max_delay_ms = 20.0              # alignment search range
segment_duration_sec = 10.0      # correlation window
min_correlation = 0.3            # refuse low-confidence alignment

[analyze]
default_target_lufs = -18.0      # reference for recommended_gain_db
```

Tools read `config.toml` from the project root regardless of the working
directory, and record the resolved values in each operation record.

---

## Supported DAW session formats

- Pro Tools (.ptx) — track names, clip regions, sample rate
- Ableton Live (.als) — audio tracks, clip names, BPM

For sessions without a DAW file, skip `parse_session` and run `analyze` + `apply_gain --per-channel`
directly on your pre-assembled stems.

---

## License

MIT
