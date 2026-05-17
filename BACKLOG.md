# Backlog

Deferred ideas — discussed but not yet implemented. Each entry includes the
motivation, a rough scope sketch, and the condition under which it becomes
worth doing. Ordered by how readily it fits the existing architecture.

---

## Completed since this file was created

These items used to live here and have been shipped:

- ~~Dedicated mastering tools (master_mix + master_health, multi-format
  delivery)~~ — done (commit batch after f2eea7a). Pipeline is now
  end-to-end mix → master with format presets for Spotify, Apple,
  YouTube, Tidal, CD, vinyl pre-master, and broadcast.

---

## 1. Volume / clip automation

**What.** Per-track time-based gain envelope. Linear interpolation between
keyframes. New field in `mix_config.tracks[*]`:

```json
"automation": [
  {"time_s": 0,    "gain_db": -3},
  {"time_s": 45,   "gain_db": -3},
  {"time_s": 47,   "gain_db":  0},
  {"time_s": 120,  "gain_db":  0}
]
```

**Why.** Last meaningful classical-mixing gap in the pipeline. Modern 2026
mixing best practice — vocal rider, "verse softer / chorus louder", post-
chorus dropouts — is per-clip time-based. Current `render_mix` only supports
a single static `volume_db` per track.

**Scope.** ~80 lines in `render_mix.py`. Sample-accurate gain envelope
generated from the keyframe list, applied to the track buffer before the
existing pan + bus routing. CLAUDE.md gets a new section explaining when
to use automation vs. compression for level shaping.

**Triggers.** Worth doing when:
- A session arrives with vocal that needs section-level rider
- A live-rec session has section dynamics that comp+limiting alone can't tame
- The user explicitly asks for "automate the volume" or "ride the vocal"

---

## 2. Pop / hip-hop / orchestral preset packs

**What.** New preset JSON files for non-rock-band sessions.

- **Pop** — vocal_lead, vocal_bg, kick_pop, snare_pop, synth_lead
- **Hip-hop** — kick_808, snare_clap, hi-hat_trap, vocal_rap
- **Orchestral** — strings_section, brass_section, woodwinds, timpani

**Why.** The current 46 presets are rock-band-focused (kick_in, snare_top,
guitar_clean, bass_di, etc.). CLAUDE.md mentions "rock band, orchestral"
but there is zero orchestral content. An agent given an orchestral session
today will reach for `guitar_clean` presets that don't fit.

**Scope.** ~30-50 new JSON files. Per genre, an EQ-preset pack plus comp-
preset pack plus maybe one or two FX presets (e.g. trap hi-hat decay).

**Triggers.** Worth doing when:
- User explicitly mentions a non-rock session
- Or as a one-time investment to broaden the agent's competence area

---

## 3. CI / GitHub Actions

**What.** `.github/workflows/test.yml` that runs the existing pytest smoke
suite on every push and PR.

**Why.** Defends the repo against regressions when changes touch `tools/`.
The smoke tests (17 tests, < 1s) already exist; this just runs them
automatically.

**Scope.** ~30 lines of YAML. Install Python 3.11, install requirements,
run pytest. Optionally a lint step (ruff or flake8).

**Triggers.** Worth doing when:
- Another contributor joins and we don't want manual test-running to be a
  per-PR ritual
- Public release / GitHub stars start arriving and broken `master` would
  be embarrassing

---

## 4. Vocal toolkit (de-esser, pitch correction, vocal presets)

**What.** A vocal-stem-aware set of tools and presets.

- `apply_deesser.py` — frequency-specific sidechain compressor on the 5-8 kHz
  sibilance region
- `apply_pitch_correct.py` — Melodyne/Auto-Tune style; librosa or world
  vocoder under the hood
- Vocal EQ / comp presets — `vocal_lead_pop`, `vocal_lead_rock`, etc.

**Why.** The current repo has zero vocal-specific tooling. The 2026 streaming
target for vocal momentary LUFS (-10 to -8 with full mix at -14) cannot be
hit reliably without vocal-aware processing.

**Scope.** Each tool ~150-250 lines + relevance check. De-esser is the
quickest win (~100 lines, just a sidechained band-comp); pitch correction
is the heavy item (formant preservation, phase coherence on shifted blocks
is real DSP work).

**Triggers.** Worth doing when:
- User starts mixing sessions that contain vocals
- A specific vocal sibilance problem comes up

---

## 5. Drum sample replacement / augmentation

**What.** `apply_drum_replace.py` that onset-detects the live drum mic and
layers a chosen sample on each hit, mixing the sample under the live signal
at a chosen level. Optionally with velocity-tracking — louder hits get a
louder sample layer.

**Why.** A classic rock-production technique (Slate Trigger 2, Drumagog,
Addictive Trigger). Lets a weak drum recording inherit the body and attack
of a great sample without losing the live take's timing and feel. Especially
useful for kick reinforcement.

**Scope.** ~150 lines + a minimal sample library (5-10 kick samples,
5-10 snare samples). The onset detection layer is already present in
`analyze.py`. The new work is loading samples, applying velocity scaling,
sub-sample alignment of the sample peak to the detected onset, and
crossfading.

**Triggers.** Worth doing when:
- A session arrives with a weak kick or snare recording
- The user explicitly asks for sample reinforcement

---

## 6. Advanced mastering features

**What.** Extensions to the existing `master_mix.py` / `master_health.py`:

- **True codec roundtrip**: encode → decode → measure (ffmpeg-backed Ogg
  Vorbis, AAC, MP3) instead of the 8×-oversampling proxy. Exact post-codec
  true peak rather than a conservative estimate.
- **ISRC / metadata embed**: write ISRC code, artist, album, track title,
  ISWC into the WAV / FLAC headers. Required for distributors (DistroKid,
  Tunecore, CD Baby).
- **DDP image export** for CD plant delivery (track marks, PQ codes, EAN).
- **Vinyl-specific master**: dedicated cutter pre-master with elliptical
  EQ in the low end (sum mono below 200 Hz to prevent groove jumps),
  RIAA pre-emphasis option, side-A / side-B time limits.
- **Stem mastering**: instead of one master, output vocal-up, vocal-down,
  instrumental, TV mix variants from the same chain.

**Why.** These cover the "professional delivery" tail of the mastering
workflow that the current tools don't reach. The current pipeline ships a
master_<format>.wav that's accurate against streaming platforms; commercial
delivery often needs more.

**Scope.** ffmpeg dependency for true codec roundtrip (~80 lines plus
subprocess plumbing). ISRC / metadata via the `mutagen` library
(~50 lines). DDP image is genuinely complex (~400 lines, custom
binary format). Vinyl pre-master can layer on existing chain
(~100 lines). Stem mastering is straightforward if you already have
the stems separately (~120 lines).

**Triggers.** Worth doing when:
- A commercial release is being prepared
- A specific delivery target needs metadata or DDP
- A vinyl cut is on the schedule

---

## 7. Spatial / Dolby Atmos output

**What.** Object-based render path. `render_mix --atmos` produces an ADM BWF
file with bed channels + objects (per-track positional metadata). Optionally
binaural stereo render for headphone preview.

**Why.** Apple Music ships every track in Dolby Atmos by default since 2023.
The 2026 streaming landscape increasingly assumes immersive support is
table-stakes for premium tier. Stereo-only delivery limits where the mix
plays well.

**Scope.** Large — this is the biggest item on the backlog. The pedalboard /
scipy stack doesn't have an ADM writer; would need either a new dependency
(`pyadm`, an experimental library) or hand-rolled BWF chunks. Object panning
math is straightforward (HRTF / vector-based amplitude panning) but
verifying the render against Apple's Spatial Audio renderer requires their
toolset (closed-source).

**Triggers.** Worth doing when:
- User delivers to Apple Music / streaming platforms with Atmos support
- A specific session asks for an immersive mix
- Don't do this as a generic upgrade — the complexity is large and rock-
  band-stem-mixing rarely benefits from Atmos
