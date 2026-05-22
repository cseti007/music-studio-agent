# Backlog

Deferred ideas — discussed but not yet implemented. Each entry includes the
motivation, a rough scope sketch, and the condition under which it becomes
worth doing. Ordered by how readily it fits the existing architecture.

---

## Completed since this file was created

These items used to live here and have been shipped:

- ~~Vocal toolkit Phases 1-4~~ — done. `_detect_bus` routes vocal-lead/vocal-bg
  keywords to dedicated buses; `apply_deesser.py` runs the sidechain band-comp
  AFTER the main compressor (per the 2026 best-practice ordering);
  `apply_pitch_correct.py` uses librosa.pyin + psola for scale-aware
  correction; 13 new vocal preset JSONs (5 EQ pairs cut/boost + 4 comp + 3
  reverb); reverb-bus architecture in render_mix supports shared `reverb_buses`
  with per-track `reverb_sends`; analyze.py writes a `vocal` block with
  sibilance / plosive / pitch / vibrato / breath metrics; all 7 style profiles
  gained vocal_lead and vocal_bg default_bus_volume_db values. **Phase 5
  (apply_vocal_align + further test coverage) is still open below.**

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

## 4. Vocal toolkit — Phase 5 (alignment + heavier test coverage)

**Status.** Phases 1-4 of the vocal-toolkit plan **shipped** (see the
Completed section above). What remains:

**What's left in this entry.**

- `tools/apply_vocal_align.py` — onset-based alignment for lead vocal +
  double-tracks. Different from `align_phase.py` (cross-correlation):
  for vocal doubles the phrasing isn't perfectly coherent, so phonetic
  onsets are the reliable anchor. ~200 lines.
- Heavier integration test coverage: render a tiny synthetic vocal
  session end-to-end through the full chain (subtractive EQ → comp →
  deesser → additive EQ → optional pitch correct → reverb-bus send) and
  assert the spectrum lands where the chain should put it.
- `apply_vocal_align.py` integration into the chain ordering in CLAUDE.md
  decision tree.

The original plan (full 5-phase description, 2026 best-practice research,
sources, etc.) is still useful as background — kept below for reference.

### Background — full pipeline plan (researched 2026)

(Phases 1-4 of this plan have shipped; the rest below is historical
context for Phase 5 and any future refinements.)

**What.** A complete vocal-stem-aware pipeline: stem detection, vocal-
specific analysis fields, a chain of processing tools applied in the
modern-best-practice order, shared reverb buses, and per-genre vocal
presets across all 7 style profiles. Designed in 5 phases so an MVP can
ship after Phase 1+2 (~5 days) and the heavier pitch-correction work
can wait until a session actually needs it.

### Best-practice vocal chain order (per-stem)

Researched against 2026 sources (Music Guy Mixing, iZotope, Sonarworks,
UAD, Patrik Skoog). The non-obvious bits:

```
volume → pan
  → subtractive EQ        (HP @ 80 Hz, mud cut @ 200-400 Hz, harshness notches)
  → compression           (3-4:1 leveling)
  → de-esser              (catches comp-amplified sibilance — placed AFTER comp on purpose)
  → additive EQ           (presence @ 3-4 kHz, air @ 12 kHz — placed AFTER comp so boosts aren't squashed)
  → [pitch correction]    (optional, comes before reverb sends)
  → saturation            (character processor, optional)
  → reverb sends          (to shared vocal_plate / vocal_hall buses — never insert reverb on the vocal channel itself)
```

Two key reversals vs. a naïve chain:
1. **De-esser AFTER compression**, not before. The comp amplifies
   sibilance peaks; the de-esser catches them at the comp's output where
   they're worst.
2. **EQ split into subtractive (pre-comp) and additive (post-comp).**
   Cuts before comp keep its detection clean; boosts after comp prevent
   the boost from being squashed by gain reduction.

### Phase 1 — MVP (~3 days)

| Component | Detail |
|---|---|
| `_detect_bus` vocal-keywords | New `_VOCAL_KEYWORDS = ["LEAD VOX", "LEAD VOC", "VOX", "VOC", "BG VOX", "BACKING", "HARMONY", "DOUBLE", "AD-LIB", "ADLIB", "WHISPER", "TALK"]`. Returns `vocal_lead` or `vocal_bg` sub-bus, parent `vocal`. |
| `tools/apply_deesser.py` (~250 lines) | Frequency-band-specific sidechain comp. Detection band 5-8 kHz, gain reduction full-band. Presets: `deesser_smooth`, `deesser_aggressive`, `deesser_male_lead`, `deesser_female_lead`. `relevance_check`: skip if sibilance band peak < -25 dBFS. |
| 5 new vocal EQ presets | `eq_vocal_lead_rock_cut/_boost`, `eq_vocal_lead_pop_cut/_boost`, `eq_vocal_lead_ballad_cut/_boost`, `eq_vocal_bg_cut/_boost`, `eq_vocal_double_cut/_boost`. Each split into a "subtractive" preset (pre-comp) and an "additive" preset (post-comp). |
| 4 new vocal comp presets | `comp_vocal_lead_rock` (4:1, -18 dB, 5/100 ms, +6 makeup), `comp_vocal_lead_pop` (6:1, -16, 3/80, +6 — heavier), `comp_vocal_lead_ballad` (3:1, -20, 8/120, +4 — gentler), `comp_vocal_bg` (3:1, -20, 10/150). |
| 3 algorithmic + 3 IR vocal reverb presets | Algorithmic: `vocal_plate` (1.5s decay, HP @ 300, LP @ 8k), `vocal_chamber` (8 ms pre-delay, 0.6s), `vocal_hall_wide` (60 ms pre-delay, 2-3s). IR pack additions: `vocal_plate_emt140`, `vocal_chamber_small`, `vocal_room_close` (added to `tools/generate_irs.py`). |
| Style profile vocal bus defaults | All 7 profiles gain `vocal_lead` and `vocal_bg` in `default_bus_volume_db`. Pop = +4/-1 (vocal-dominated); rock = +2/-2; ballad = +2.5/-2; hip-hop = +3/-3; jazz = +2/-2. |

### Phase 2 — Reverb-bus architecture (~1 day)

The studio-standard "send to shared reverb bus" pattern. Today every bus
has its own `reverb_send` (insert-style); for vocal mixing we need
multiple sources sending to one shared plate / hall / chamber bus at
different levels. Schema extension:

```json
"reverb_buses": {
  "vocal_plate": {"preset": "vocal_plate", "wet": 1.0, "return_volume_db": -6, "return_pan": 0.0},
  "vocal_hall":  {"preset": "vocal_hall_wide", "wet": 1.0, "return_volume_db": -12, "return_pan": 0.0}
},
"tracks": [
  {
    "name": "LEAD VOX",
    "bus": "vocal_lead",
    "reverb_sends": [
      {"bus": "vocal_plate", "level_db": -6},
      {"bus": "vocal_hall",  "level_db": -18}
    ]
  }
]
```

`render_mix.py` change: ~100 lines. New rendering pass for each
`reverb_bus` (one convolution / Freeverb instance, fed by summed pre-
fader sends from contributing tracks), then return-bus mix into master.

### Phase 3 — Vocal-specific analyse fields (~1 day)

New top-level keys in `analysis.json` for vocal stems. Inferred from the
existing analyse-decision-tree pattern:

| Field | What | Decides |
|---|---|---|
| `sibilance.peak_db` | 5-8 kHz transient peak | de-esser threshold |
| `sibilance.density_per_sec` | sibilance events per second | de-esser preset (smooth vs aggressive) |
| `plosive.events_count` | sub-100 Hz transient bursts | high-pass / plosive removal |
| `pitch.mean_hz` | librosa.pyin avg fundamental | vocal range detection (male / female / other) |
| `pitch.cents_std` | intonation stability | auto-tune intensity |
| `vibrato.rate_hz` | 4-7 Hz amplitude modulation | vocal style classification |
| `breath.silence_ratio` | silent-frame ratio | gate threshold |

### Phase 4 — Pitch correction (~1-2 days)

`tools/apply_pitch_correct.py`. After web research the implementation
is more accessible than first thought — there is a pip-installable
`psola` package that handles the PSOLA pitch-shifting (used in the
JanWilczek/python-auto-tune reference repo).

```python
import librosa, psola
y, sr = librosa.load("vocal.wav", sr=None)
f0, _, _ = librosa.pyin(y, fmin=80, fmax=600, sr=sr)
f0_quantized = quantize_to_scale(f0, root_note='A', mode='minor')   # ~30 lines of scale logic
y_corrected = psola.vocode(y, sample_rate=sr, target_pitch=f0_quantized,
                           fmin=80, fmax=600)
```

CLI: `--scale-root A --scale-mode minor --strength 0.7` (0=no
correction, 1=full quantize). Skip when `pitch.cents_std < 25` (already
in tune). Defer the "Deep Autotuner" NN approach (arxiv 1902.00956,
2002.05511) — overkill for MVP.

### Phase 5 — Polish (~1 day)

- `apply_vocal_align.py` — lead + double-tracks alignment. Unlike
  `align_phase.py` (which uses cross-correlation), vocal double-tracks
  benefit from **onset-based** alignment because the phrasing isn't
  perfectly coherent — phonetic onsets are the reliable anchor points.
- Tests for everything (`TestVocalDeesser`, `TestVocalPitchCorrect`,
  `TestReverbBus`, `TestVocalDetectBus`).
- Docs: CLAUDE.md gains a "Vocal pipeline" workflow section + tools-
  table rows + new decision-tree triggers (e.g. "vocal stem
  `sibilance.peak_db > -20` → run apply_deesser"). knowledge.md gains a
  "Vocal Mixing" top-level section with genre-specific aesthetics
  (pop = bright + heavy comp + plate + slap; rock = present + moderate
  comp + room; ballad = natural + soft comp + lush hall; hip-hop = pitched
  + hard-comp + chamber + ad-libs panned).

### Why this matters

The current repo has zero vocal-specific tooling. The 2026 streaming
target for vocal momentary LUFS (-10 to -8 with full mix at -14) can't
be hit reliably without vocal-aware processing. The horgonyt_fel session
was the first vocal-bearing session and exposed the gap: the vocal stem
was processed with a hand-rolled custom EQ + comp chain because no
presets exist; it went into the mix dry because the reverb-bus
architecture isn't there.

### Scope summary

| Phase | Days | Notes |
|---|---|---|
| 1 (MVP — deesser + presets + bus detection) | 3 | Functional vocal mixing |
| 2 (Reverb-bus architecture) | 1 | Shared-bus reverb routing |
| 3 (Vocal-specific analyze fields) | 1 | Decision-tree completeness |
| 4 (Pitch correction) | 1-2 | `psola` package shortens this vs. building PSOLA from scratch |
| 5 (Polish + alignment + docs) | 1 | Tests, docs, vocal-align |
| **Total** | **~7-8 days** | **MVP after 1+2 = 4 days** |

### Triggers

Worth doing when:
- A vocal-bearing session arrives that needs delivery-ready vocal mixing
- The horgonyt_fel session is being revisited
- A specific vocal sibilance / pitch problem comes up

### Web-research sources (2026 best practices)

- [Music Guy Mixing — Vocal Chain Order](https://www.musicguymixing.com/vocal-chain/)
- [iZotope — Crafting a basic vocal chain](https://www.izotope.com/en/learn/crafting-a-basic-vocal-chain)
- [Sonarworks — Chaining vocal effects plugins](https://www.sonarworks.com/blog/learn/how-to-chain-multiple-vocal-effects-plugins-effectively)
- [Universal Audio — Top vocal chains](https://www.uaudio.com/blogs/ua/top-uad-vocal-chains)
- [Patrik Skoog — How to Mix Vocals](https://www.patrikskoogmusic.com/guides/how-to-mix-vocals-eq-compression-saturation)
- [PSOLA algorithm explanation](https://engprojects.tcnj.edu/autotuner16/2016/04/11/the-psola-algorithm/)
- [JanWilczek/python-auto-tune (PYIN + PSOLA reference impl)](https://github.com/JanWilczek/python-auto-tune)

**Status.** Plan researched and detailed 2026-05-21 (this update). Ready
to implement Phase 1 when a vocal-bearing session is on deck.

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

---

## 10. Docs refactor — split CLAUDE.md + knowledge.md into topical files

**What.** CLAUDE.md is currently 800+ lines, docs/knowledge.md is 1500+ lines.
Both are single monolithic files that have grown organically and are hard
to navigate. Split into topical sub-files:

```
CLAUDE.md                       # tight operational guide (200-300 lines max)
  → links to:
docs/
  gain_staging.md               # per-clip vs per-channel, autotrim, headroom
  mixing_conventions.md         # panning, balance, premaster handoff
  mastering.md                  # master_mix chain, formats, master_health
  analysis_decision_tree.md     # the big table; quoted by CLAUDE.md
  vocal_pipeline.md             # de-esser, pitch correct, vocal chain order
  forensic.md                   # find_clicks, polarity, source forward
  knowledge.md                  # remaining: domain ref, frequency bands
```

**Why.** The May 2026 panning miss happened partly because the agent skimmed
a 2300-line doc set and the panning convention literally wasn't in there.
Smaller topical files are easier to fully ingest and easier to add to. The
agent decision tree should be a focused 100-line table, not buried in a
800-line operational doc.

**Scope.** Medium. 2-3 hours of editing — most content already exists, just
needs reorganising. The links from CLAUDE.md to the sub-files need to be
clear and discoverable. Tests don't change. Memory entries don't change
(they reference concepts, not file paths). The risk is that the next
contributor doesn't notice the new structure and adds back to CLAUDE.md
inline.

**Triggers.** Worth doing when:
- CLAUDE.md crosses ~1000 lines or knowledge.md crosses ~2000 lines (close
  to that now)
- A user reports that the agent missed something the docs cover (already
  happened in May 2026 — panning conventions)
- Onboarding a second contributor / cross-checking with another agent
