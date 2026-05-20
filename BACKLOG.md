# Backlog

Deferred ideas — discussed but not yet implemented. Each entry includes the
motivation, a rough scope sketch, and the condition under which it becomes
worth doing. Ordered by how readily it fits the existing architecture.

---

## Completed since this file was created

These items used to live here and have been shipped:

- ~~Style-aware mix evaluation (style profiles + style_check.py)~~ — done.
  Five built-in profiles (`modern_rock`, `classic_rock`, `pop`, `hip_hop`,
  `jazz_acoustic`) with LUFS / LRA / crest / 5-band tonal balance targets.
  Calibrated against real-world streaming masters; numbers are wideband
  band-RMS at the profile's LUFS target (NOT iZotope-TBC PSD-curve values).
  Tool emits 0-100 score + traffic-light verdict + EQ recommendations.
- ~~Reproducibility: mix_chain.json recall sheets~~ — done. `build_chain.py`
  aggregates every per-stem `*_report.json` into a single chain; `replay_chain.py`
  rebuilds the full mix from it via subprocess.
- ~~Dedicated mastering tools (master_mix + master_health, multi-format
  delivery)~~ — done (commit batch after f2eea7a). Pipeline is now
  end-to-end mix → master with format presets for Spotify, Apple,
  YouTube, Tidal, CD, vinyl pre-master, and broadcast.
- ~~Master chain feature completeness (multiband / M/S / stereo width /
  vinyl elliptical / batch master_health)~~ — done (commit f9f5884).
  Added six chain preset (`modern_rock_mb`), per-format sub-mono filter
  for vinyl, cross-format batch scorecard.
- ~~Vinyl-specific master with elliptical EQ in the low end~~ — done
  inside the master chain feature batch. The `vinyl_pre` format
  triggers a 4th-order sub-mono filter (side channel HP at 150 Hz)
  automatically; sample-side energy below 150 Hz is removed so the
  cutter head can track the groove.

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

**Why.** The current 47 presets are rock-band-focused (kick_in, snare_top,
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
The smoke tests (39 tests, ~1.2s) already exist; this just runs them
automatically.

**Scope.** ~30 lines of YAML. Install Python 3.11, install requirements,
run pytest. Optionally a lint step (ruff or flake8).

**Triggers.** Worth doing when:
- Another contributor joins and we don't want manual test-running to be a
  per-PR ritual
- Public release / GitHub stars start arriving and broken `master` would
  be embarrassing

---

## 4. Vocal toolkit (de-esser, pitch correction, vocal presets, vocal reverb)

**What.** A vocal-stem-aware set of tools, presets, and reverb integration.

- `apply_deesser.py` — frequency-specific sidechain compressor on the 5-8 kHz
  sibilance region
- `apply_pitch_correct.py` — Melodyne/Auto-Tune style; librosa or world
  vocoder under the hood
- Vocal EQ / comp presets — `vocal_lead_pop`, `vocal_lead_rock`,
  `vocal_lead_ballad`, etc. (the horgonyt_fel session needed a manual
  `HP@80 + cut@250 + boost@4k` chain because none of the existing
  presets fit)
- **Vocal reverb presets** — `vocal_plate` (1.5s decay, HP@300, LP@8k),
  `vocal_chamber` (8 ms pre-delay, 0.6s decay), `vocal_hall_wide`
  (60 ms pre-delay, 2-3s decay)
- **Reverb-bus architecture in render_mix** — currently every bus has
  its own optional `reverb_send`, but the studio standard is dedicated
  reverb buses (plate, room, hall) that multiple tracks/buses send
  into at different levels. This becomes load-bearing once vocal mixing
  is in scope (vocal typically sends to two reverb buses simultaneously).
  Spec sketch:

  ```json
  "reverb_buses": {
    "plate": {"preset": "vocal_plate", "wet": 1.0},
    "hall":  {"preset": "hall_ambient", "wet": 1.0}
  },
  "tracks": [
    {"name": "Vocal", "reverb_sends": [
      {"bus": "plate", "level": 0.2},
      {"bus": "hall",  "level": 0.15}
    ]}
  ]
  ```

**Why.** The current repo has zero vocal-specific tooling. The 2026
streaming target for vocal momentary LUFS (-10 to -8 with full mix at
-14) cannot be hit reliably without vocal-aware processing. The
horgonyt_fel session was the first vocal-bearing session and exposed
the gap: the vocal stem was processed with a hand-rolled custom EQ +
comp chain because no presets exist; it went into the mix dry because
the reverb-bus architecture isn't there.

**Scope.** Each tool ~150-250 lines + relevance check. De-esser is the
quickest win (~100 lines, just a sidechained band-comp); pitch correction
is the heavy item (formant preservation, phase coherence on shifted blocks
is real DSP work). Vocal reverb presets are trivial (3 new JSON files).
The reverb-bus architecture in render_mix is the largest piece — about
100 lines of routing code (independent reverb-bus rendering, return to
master) plus mix_config schema extension.

**Triggers.** Worth doing when:
- User starts mixing sessions that contain vocals (the horgonyt_fel
  vocal stem was processed manually; revisiting it would benefit from
  proper vocal-toolkit support)
- A specific vocal sibilance problem comes up

**Status.** User explicitly deferred this in the current scope — vocal
work is out of scope until vocal-bearing sessions become regular. Pick
up when that happens.

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
- **Vinyl pre-master extensions**: the basic sub-mono filter at 150 Hz
  is already shipping on the `vinyl_pre` format. Remaining: RIAA
  pre-emphasis option, side-A / side-B time limits.
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

## 7. VST3 / AU plugin support

**What.** A `tools/apply_vst.py` generic runner that loads a VST3 (or AU
on macOS) plugin via pedalboard's already-shipped `pedalboard.load_plugin()`
API, sets parameters from CLI flags or a JSON preset, and renders the
processed WAV. Optionally extend the existing `apply_eq` / `apply_compression`
/ `apply_reverb` tools with a `--vst <path>` flag that bypasses the
built-in DSP and routes through a user-supplied plugin.

**Why.** Specific situations where the built-in DSP is the limiting
factor:
- **Mastering quality jump** — iZotope Ozone 11, FabFilter Pro-L 2, Pro-MB,
  Pro-Q 4 dynamic EQ outperform the current `master_mix.py transparent`
  preset on absolute pro-tier delivery.
- **Distinctive character** — Soundtoys Decapitator (saturation),
  Valhalla VintageVerb (reverb), Slate VMR (channel strip), Waves SSL
  G-Bus give signature colours the built-in math approximations can't
  fully reproduce.
- **Dynamic / multiband EQ** — Pro-Q 4's dynamic mode + per-band mid-side
  processing is in a different league than scipy biquads.

**Scope.**
- **MVP** (~200 lines): `tools/apply_vst.py` standalone runner.
  `--plugin <path>` `--param NAME=VALUE` (repeatable). Output: WAV +
  report.json with the parameter dictionary that was applied.
- **Preset layer** (~150 lines): per-plugin JSON presets under
  `tools/presets/vst_<plugin>_<name>.json`. `--list-vst-plugins` scans
  system-typical VST3 dirs.
- **Integration** (~300 lines): `--vst <path>` flag on existing tools;
  `mix_chain.json` schema gains a `vst` step type; `replay_chain` does
  graceful fallback to the built-in DSP when the plugin isn't available
  on the target machine (the chain still completes, just sounds
  different — flagged in the run log).

**Trade-offs to remember.**
- **Reproducibility cost**: a chain step that depends on FabFilter Pro-Q
  can't be exactly replayed on a machine without Pro-Q installed.
  Mitigation: the chain `vst` step always records the parameter snapshot,
  so a built-in EQ approximation can be auto-generated from the captured
  params (lossy but valid).
- **Plugin ownership**: the repo ships only the wrapper. The user has to
  own and install the actual VSTs. Pro plugin bundles (Waves Mercury,
  iZotope Music Production Suite) cost $500-$2000.
- **pedalboard Linux stability**: `pedalboard.load_plugin()` has been
  reported to stall on certain VST3s under JUCE 7 on Linux. MVP must
  smoke-test on a known-good free plugin (Surge XT, TAL-Reverb-4) to
  establish that the path works on the user's machine before claiming
  generic support.

**Triggers.** Worth doing when:
- A specific session has a quality requirement the built-in DSP can't
  reach (e.g. final commercial master needing Ozone 11).
- The user acquires a pro plugin bundle and wants CLI-driven access to
  it for batch processing.
- A specific plugin's character (Decapitator's grit, Pro-Q's dynamic
  EQ) becomes a recurring need across sessions.

**Status.** Explored 2026-05-18. User has only a few free plugins and
no concrete use case yet, so deferred. The pedalboard infrastructure is
already in place (no new dependencies needed), so when the trigger
fires the MVP is genuinely 1 day of work.

---

## 8. Config-driven guitarist-prefix detection

**What.** `tools/render_mix.py` currently has a hardcoded
`_GUITARIST_PREFIXES = ["GTR 1", "GTR LACI", "GTR TERKA", "GTR"]` list
used by `_detect_bus()` and `_guitarist_prefix()` to assign multi-guitarist
tracks to per-player sub-buses (gtr_1, gtr_laci, gtr_terka).

The first three entries are names from the project's reference test
session. Any other session with different guitarist labels (e.g.
"GTR JOHN", "GTR PAUL") falls through to the generic "GTR" bucket,
which loses the per-player sub-bus structure.

**Why.** For the project to work cleanly on arbitrary sessions, the
guitarist-prefix list should be either:
- Auto-detected from track names at `--generate-config` time (scan
  for unique `GTR <NAME>` patterns).
- Specified via the `mix_config.json` or an environment-level config file.

**Scope.** ~30 lines of changes in `render_mix.py`:
- New auto-detection function `_discover_guitarist_prefixes(tracks)`
- The existing `_GUITARIST_PREFIXES` becomes a fallback default
- `generate_config()` calls the discovery function once, stores the
  result either in `mix_config.json` or passes it through as a parameter.

**Triggers.** Worth doing when:
- A second session arrives with different guitarist names.
- Someone forks the repo and runs into a "my GTR JOHN track went to
  the generic guitar bus" issue.

**Status.** Identified during the public-readiness review on 2026-05-19.
Functional for the original session, gracefully degrades for others
(generic guitar bus still works, just loses sub-bus granularity).

---

## 9. Spatial / Dolby Atmos output

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
