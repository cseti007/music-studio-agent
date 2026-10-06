# Domain Knowledge - Audio Mixing and Mastering

Policy revision: 2026-09-09. This document separates measurements, delivery
requirements, engineering heuristics, and personal taste. Session recipes below
are historical examples unless explicitly labeled otherwise. They must not be
promoted to universal rules or used to override the user's listening decisions.

## Evidence and research policy

Use local knowledge first. Research when a delivery specification is stale, a
technique is unfamiliar, or sources conflict. Prefer standards bodies and the
platform's own delivery documentation. Record the source URL, revision, date
verified, scope, confidence, and whether the claim is a requirement,
recommendation, heuristic, preference, or session observation. Save the evidence
used with the session so future research does not silently change a recall.

| Topic | Classification | Verified source and scope |
|---|---|---|
| Integrated loudness and true peak | Measurement specification | [ITU-R BS.1770-5 (2023)](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en), verified 2026-09-09 |
| EBU R128 programme loudness | Broadcast recommendation / requirement when contracted | [EBU R128 (2023)](https://tech.ebu.ch/publications/r128), -23 LUFS; does not define musical quality, verified 2026-09-09 |
| Spotify playback and mastering | Platform recommendation | [Spotify](https://support.spotify.com/us/artists/article/loudness-normalization/): normal playback -14 LUFS; recommended maximum -1 dBTP, or below -2 dBTP for masters louder than -14 LUFS; verified 2026-09-09 |
| Apple stereo assets | Delivery requirements | [Apple asset guide](https://help.apple.com/itc/videoaudioassetguide/en.lproj/static.html): source resolution and format rules; -16 LUFS is not a universal stereo acceptance condition; verified 2026-09-09 |
| Reference matching | Engineering technique | [iZotope](https://www.izotope.com/en/learn/13-tips-for-using-references-while-mixing): level-match references for listening comparisons; verified 2026-09-09 |

The repo's polyphase true-peak meter is an estimate; passing synthetic regressions
is not a certification. Validate against the [EBU test set](https://tech.ebu.ch/publications/ebu_loudness_test_set)
before claiming standards conformance. Oversampling the waveform does not simulate
AAC, Vorbis, or any other codec. Actual codec audition remains a separate step.

## Listening evidence and review policy

Classification: project workflow policy, revised 2026-09-09. This is a safeguard
against unsupported conclusions, not an industry delivery specification.

Declare whether the agent can actually audition audio. Distinguish direct
listening, externally supplied human feedback, and numerical inference. A
playback widget, spectral plot, or successful decoder does not establish hearing.
Without direct audition, create matched excerpts and use focused human feedback;
continue objective work while leaving subjective judgments provisional.

First renders are drafts. Preserve approval scope and file identity: approving
vocal balance in one passage does not approve the full mix or a later master.
A criticism reopens the relevant decision. Store actual feedback, source, person,
scope, decision, and reviewed audio hash. Never manufacture listening records.
`tools/review_delivery.py` validates these records against the current export;
its readiness result covers only the supplied specification and human approval.
It cannot authenticate a reviewer or certify professional quality.

`tools/prepare_audition.py` matches excerpts with linear gain only. Both receive
any additional attenuation needed for peak headroom. Use the same section and
explicit source offsets, especially after trimming or changing the arrangement.
Do not compare a loud chorus with a quiet verse, or treat a louder option as an
improvement without checking at matched loudness.

For every consequential creative change, retain the intended improvement,
timestamp, hypothesis, parameters, measurements, comparison, human/direct
listening evidence, and decision. Review intro, quiet and dense sections, key
transitions, exposed vocals, solos, and ending. Preserve useful contrasts instead
of minimizing every metric difference. No amount of green scoring or passing
software tests establishes commercial success or artist equivalence.

## Knowledge maintenance

Use this curated repository guidance for stable concepts and repeatable review
procedures. Search primary sources for unfamiliar techniques, conflicting claims,
and current delivery rules. Reverify time-sensitive requirements for each new
delivery brief. Record URL, publication/revision, verification date, claim type,
scope, confidence, and exceptions. Vendor tutorials describe techniques and may
promote products; they do not establish universal targets.

Keep session observations and numerical recipes labeled as historical. An old
song's settings are not evidence that another performance needs those settings.
New research informs a proposed comparison; it must not silently change an
approved mix, a saved preference, or an earlier recall.

## Decision policy

- Preserve original edits, channel relationships, artistic processing and levels.
  Per-clip normalization and equal-LUFS bus auto-trim are opt-in choices.
- Propose a specific audible improvement; retain it only after measurement and
  level-matched audition. A treatment that raises a score may still sound worse.
- LUFS measures programme loudness. LRA measures longer-term loudness variation;
  low LRA does not prove overcompression and high LRA does not prove good punch.
- Stereo width and tonal targets depend on arrangement, references and taste.
  Spectral overlap is a masking hypothesis, not a mandate to cut EQ.
- A green report means only that the stated measured checks passed. Listening
  approval, required formats and any contractual delivery checks remain separate.
- Use floating-point intermediates. Quantize at export; read back the resulting
  WAV and measure its actual loudness and true peak. Peak safety takes precedence
  when a requested loudness is unattainable without unwanted processing.
- A -3 dBFS premaster peak is this project's default, not a universal standard.
  No fixed premaster LUFS or minimum LRA is required for good mastering.
- Continuous source reconstruction changes timing and can restore edited-out
  material. Treat it as an explicit editorial alternative; never auto-select it
  solely from clip counts or instrument names.

## Stem Assembly and Clip Gain

**Rule: clip gain and assembly are done together as the first step, before any further processing.**

`apply_gain --per-clip` reads the session.json clip layout and assembles the full-length
stem at original levels. Add `--normalize` only for intentional per-clip leveling.

The correct order is:
```
1. parse_session                      ->  session.json (clip layout)
2. apply_gain --per-clip session.json ->  assembled.wav (clip gain + assembly)
3. analyze                            ->  read the assembled stem
4. apply_gain --per-channel (optional)->  delivery normalization only
5. ...further processing
```

### Clip gain vs per-channel gain staging

Assembly defaults to original recording levels. `--normalize` opts into equal
per-clip LUFS to correct identified accidental gain differences. The existence
of a DAW clip-gain control does not imply a standard -18 LUFS normalization rule.
Per-channel gain applies one gain to an entire stem and preserves its internal
level relationships. Use it when needed for balance or processor operating level.

### Intentional dynamics and clip boundaries

Preserve clip levels unless their differences are unwanted. Instrument names and
clip counts do not establish whether takes share recording gain. Independently
normalizing related microphones can change the balance and stereo image.

Assembly applies no crossfade by default. `--crossfade-ms 5` opts into an
equal-power crossfade near adjacent clip boundaries. Audition the join: correlated
material can gain level during an equal-power overlap. This is a new edit and
does not reconstruct the DAW's original fades. Prefer consolidated stems when
those fades, automation, plugins, or channel mappings matter.

### Continuous source mode - editorial alternative

Continuous reconstruction can remove source jumps but also reverses timing
edits and restores material the editor removed. Clip count alone does not prove
an audible defect or justify this mode. Preserve the DAW clip layout by default.

If a seam sounds wrong, compare the source, original assembly, and a local repair
at the same timestamp. Try a small fade only where needed and listen for changes
to attacks and sustained tones. Crossfade artifacts depend on source phase,
length, and level; there is no universal warble rate derived from clip counts.

Use `--source-mode continuous` only when the editorial change is explicitly
wanted. It groups source placements and blends overlaps; it is not exact DAW
playback. Check timing and boundaries against the intended arrangement.

### Per-source normalize (`--normalize-per-source`)

Continuous mode preserves the natural recording dynamics of each take — including the engineer's potentially-different INPUT GAIN settings between session takes. If the engineer recorded section A at -18 dB input gain and section B at -12 dB, those level differences persist in the continuous-mode assembly. Result: in-mix sections have audibly different volumes even after bus-level autotrim averages them.

The `--normalize-per-source --source-target-lufs -18` flag combination normalizes EACH placement (source-cluster) to the target LUFS before crossfade-blending into the timeline. For a track with 5 source-files, that's 5 normalization points (vs 100+ in per-clip mode).

**This is a refined form of per-clip normalization that solves the same goal (inter-section level consistency) with far fewer normalization boundaries and without flattening intra-take performance dynamics.**

Comparison:

| Dimension | Per-clip `--per-clip --normalize` (opt-in) | Per-source `--source-mode continuous --normalize-per-source` |
|---|---|---|
| Normalization points (100 clips / 5 sources) | 100 | 5 |
| Inter-section consistency | Max (every clip level) | Good (every source level) |
| Intra-take dynamics | FLATTENED (kills natural performance) | PRESERVED |
| Boundary warble risk | yes (100+ × 5ms ≈ 500ms warble) | low (5-15 boundaries × 50ms ≈ 250-750ms blend) |
| Best for | unstable player amplitude needing brutal leveling | natural performance dynamics + engineer section-gain fixes |

**Field measurement — horgonyt 2026-05-25 (10 continuous-mode tracks: 2 bass + 8 guitars):**

| Metric | v3 (no per-source norm) | v4 (per-source norm) |
|---|---|---|
| 3s-window loudness range (mix) | -39.6 → -18.1 = **21.5 LU** | -34.0 → -18.1 = **15.9 LU** |
| Stddev across windows | 3.77 LU | 2.95 LU |
| Quiet breakdown (4:00-5:15) | -29 LUFS | **-27 LUFS** (+2.5 LU lift) |
| Loud chorus (2:45-3:45) | -19 LUFS | -19 LUFS (unchanged) |
| Master sum_in | +0.92 dBFS | +0.58 dBFS |

The per-source norm raised quiet sections without touching loud sections — natural compression without dynamic flattening within a take.

**Use only when this editorial change is explicitly wanted.** For drums (per-clip mode), the flag is ignored. For tracks with a single source-file referenced by many clips (e.g., bass with 100 clips / 1 source), the flag is a no-op (only 1 placement to normalize).

**Workflow command:**

```bash
apply_gain --per-clip session.json --track "BASS DI CLEAN" --track "BASS DI PEDAL" \
  --track "GTR 1 FENDER" --track "GTR 1 ORANGE" \
  --track "GTR B FENDER.01" --track "GTR B ORANGE.01" \
  --output-dir output/<session>/tracks \
  --source-mode continuous --crossfade-ms 50 \
  --interloper-head-ms 2000 --interloper-tail-ms 800 \
  --normalize-per-source --source-target-lufs -18
```

This is an optional reconstruction recipe, not a default.

### Polarity and timing in bass microphone/DI blends

Parallel paths can have different polarity, delay, frequency response, or
processing. Strong negative correlation can identify an inverted duplicate,
but does not prescribe which path to mute or shift. Inspect simultaneous active
sections, compare the combined tone in mono and stereo, and preserve the
intended performance timing.

`polarity_flip` in the track configuration inverts a path before summing.
`align_phase.py` estimates a time/polarity alternative. Audition either against
the original blend; do not flip or align from the instrument name or one global
correlation number. If only one path is retained, investigate whether another
path contributes useful tone before adding a substitute effect.

### Identifying clips that belong together

#### If the session was recorded in a DAW with a session file (e.g. Pro Tools .ptx):

Use `ptformat` / `ptftool` (open-source C library, https://github.com/zamaudio/ptformat) to
parse the session file and extract the exact clip layout.

Build from source (no pip package exists):
```bash
git clone https://github.com/zamaudio/ptformat.git /tmp/ptformat
cd /tmp/ptformat
CXX=g++ make all INCL="-I."
# produces: ptftool, ptunxor, ptgenmissing
```

Run on a .ptx session file:
```bash
/tmp/ptformat/ptftool "path/to/session.ptx" 2>&1
```

Output format:
```
`track_name` t(id) (source_wav_file.wav) @ TIMELINE_SAMPLE + OFFSET_IN_FILE, LENGTH
```

- TIMELINE_SAMPLE: position in the session timeline (samples at session sample rate)
- OFFSET_IN_FILE: read offset inside the source WAV (samples)
- LENGTH: how many samples to read from that offset

From this, for each logical track, sort clips by TIMELINE_SAMPLE and reconstruct a
continuous audio file with silence in the gaps.

**NOTE:** Multiple WAV files with similar names may exist on disk (e.g. `GTR 1 DI.dup2.09_24.wav`
and `GTR 1 DI.dup2.09_26.wav`). The PTX tells you exactly which file was actually used and where.
Do NOT assume all files on disk are used — many are discarded takes.

#### Naming convention in Pro Tools exports (observed in Terido session, 2022):

File format: `INSTRUMENT_NAME.XX_YY.wav`
- `XX` = internal clip/region start index in the session
- `YY` = internal clip/region end index or take number
- These are NOT bar or measure numbers
- Overlapping XX ranges (e.g. `08_10` and `09_11`) = alternative takes of the same section,
  NOT overlapping timeline content

The `dup1`, `dup2`, `dup3` suffix = double-tracked guitar layers (played twice for stereo width),
each intended as a separate parallel track in the mix, NOT alternative takes of the same track.
Exception: if two files share the same dupN prefix AND the same XX value (e.g. `dup2.09_24` and
`dup2.09_26`), they ARE alternative takes — check the PTX to see which one is on the timeline.

#### If no session file is available:

Options ranked by reliability:
1. Ask the engineer which takes were used (most reliable)
2. Sort files by the first number in the `XX_YY` suffix, exclude clear duplicates by comparing
   spectrograms (same pattern = likely same take, different = different section)
3. Use amplitude-based heuristic: prefer the clip with higher peak/RMS as the "intended" take

---

## DAW Session File Formats

When assembling channels from raw clips, a session file tells us exactly which clips go where
on the timeline. Format support varies by DAW:

| DAW | Extension | Format | How to parse |
|---|---|---|---|
| Pro Tools | `.ptx` / `.pts` / `.ptf` | binary, proprietary | `ptformat` / `ptftool` (C, build from source) |
| Ableton Live | `.als` | gzip-compressed XML | `gzip` + `xml.etree.ElementTree` (stdlib only) |
| Reaper | `.rpp` | plain text, XML-like | read directly, regex or custom parser |
| Logic Pro | `.logicx` | folder (package) containing XML | unzip, parse inner XML |
| Studio One | `.song` | zip + XML | `zipfile` + `xml.etree` |
| Bitwig Studio | `.bwproject` | zip + JSON | `zipfile` + `json` |
| Cubase | `.cpr` | binary, proprietary | no reliable open parser |
| FL Studio | `.flp` | binary | `pyflp` Python library (`pip install pyflp`) |

### Ableton .als parsing (simplest case)

```python
import gzip, xml.etree.ElementTree as ET

with gzip.open("session.als", "rb") as f:
    tree = ET.parse(f)
root = tree.getroot()
# clips live under AudioTrack > DeviceChain > MainSequencer > ClipTimeable > ArrangerAutomation > Events
```

Key XML paths in .als:
- Tracks: `//AudioTrack`
- Track name: `AudioTrack/Name/EffectiveName/@Value`
- Clips: `AudioTrack/DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events/AudioClip`
- Clip timeline position: `AudioClip/@Time` (in beats)
- Source file: `AudioClip/SampleRef/FileRef/Path/@Value`
- Clip start/end in file: `AudioClip/@StartRelative`, `AudioClip/@LoopEnd`

**NOTE:** Ableton stores clip positions in **beats**, not samples. Convert using session BPM and
sample rate: `sample_position = (beat_position / bpm * 60) * sample_rate`

### Reaper .rpp parsing

Reaper files are human-readable. Key tokens:
- `TRACK` block = one track
- `NAME "track name"` = track name
- `ITEM` block = one clip
- `POSITION x` = timeline position in seconds
- `LENGTH x` = clip length in seconds
- `SOFFS x` = source offset in seconds
- `FILE "path/to/file.wav"` = source file

### parse_session.py tool

A unified tool that auto-detects the session format and outputs a canonical JSON.
Supports: Pro Tools (.ptx .pts .ptf) and Ableton Live (.als).

```json
{
  "session_file": "...",
  "sample_rate": 48000,
  "tracks": [
    {
      "name": "BASS DI CLEAN",
      "clips": [
        {
          "source_file": "BASS DI CLEAN.08_10.wav",
          "timeline_start_sample": 123456,
          "source_offset_sample": 0,
          "length_samples": 317405
        }
      ]
    }
  ]
}
```

This canonical session.json is the input for `apply_gain --per-clip`.

---

## Gain Staging

### Targets and presets

`tools/master_mix.py` contains delivery presets. Streaming LUFS defaults and the
CD loudness value is a creative starting point, not an acceptance test (the
Red Book has no loudness specification). `vinyl_pre` has no LUFS target: it is
peak-normalized to -3 dBTP with no limiter, and sub-mono is opt-in
(`--vinyl-elliptical`), because the cutting engineer decides those. Broadcast
uses the selected EBU loudness requirement. Its -2 dBTP ceiling is a
conservative project choice within the EBU R128 -1 dBTP maximum. When a master
target is louder than -14 LUFS, the default ceiling drops to -2 dBTP
(Spotify/SoundCloud guidance for loud masters).
CD export is 44.1 kHz/16-bit; Apple stereo export preserves supported native
sample rates. Verify destination-specific requirements before delivery.

`apply_gain --per-clip` preserves original levels by default. `--normalize`
uses the configured clip target. `--per-channel` sets whole-stem gain. None of
these operations establishes mix readiness by itself.

### 2026 trend: smart / preventive gain staging

- Direction: automate gain staging BEFORE plugin chains, not after.
- Continuous LUFS + RMS tracking across the full signal chain.
- True peak and inter-sample peak detection — transient peaks that standard meters miss.
- "Prevention over correction" — stable gain structure from clip gain through final output.
- Sources: [mixingmonster.com/gain-staging](https://mixingmonster.com/gain-staging/),
  [DLK Music Pro — Smart Gain Staging](https://news.dlkmusicpro.com/the-rise-of-smart-gain-staging-in-modern-audio-production/)

### General mix stage targets

- Recording input: peak around -12 to -6 dBFS (never record loud in digital)
- During mixing: mix bus peaks at -6 to -3 dBFS before mastering
- Plugin unity gain: compensate output after each plugin so in ≈ out level-wise
- Individual track headroom before plugins: aim for -18 to -12 dBFS peak

### Optional bus auto-trim

`--generate-config --auto-trim` calibrates dry bus sums to -18 LUFS. This changes
relative musical balance and is only an optional starting point. Default config
generation preserves zero auto-trim. `volume_db` remains editable for taste.
Do not automatically recompute calibration after an approved balance change.
Bus processing changes loudness, so dry calibration is not a final-output invariant.

### Mix vs master separation — premaster handoff

**Invariant:** `render_mix` produces a clean **premaster**, NOT a finished
master: 32-bit float, peak-normalized (default -3 dBFS), no limiter. The master
phase (`master_mix.py`) owns LUFS normalization, true-peak limiting, clipper,
M/S processing, and dither.

**Why.** Stacking mastering moves at the mix stage and then running them
again at the master stage produces a **two-stage limiter cascade**: every
transient is flattened twice, which is the literal definition of "doubled
limiting distortion" that mastering forums consistently warn against.
Historical sessions exposed excessive cumulative peak reduction when both the
mix render and mastering stage limited the same transients. Compare gain reduction
across stages rather than treating every stage as an independent loudness task.

The default premaster render uses optional glue compression, EQ, and a scalar
peak adjustment to -3 dBFS. It bypasses delivery limiting and loudness matching.
This is a project workflow, not a rule forbidding artistic mix-bus processing.
Keep intentional processing that defines the sound and document it for mastering.
`master.premaster_mode: false` retains the older combined mix/master path.

`mix_report.json` identifies the render stage. Health checks distinguish peak
safety from advisory loudness, dynamics, width, and tonal measurements.

### Premaster handoff

Preserve the approved sound, edits, and channel layout. Check finite samples,
clipping, and agreed headroom. The default -3 dBFS peak is a convenience; there
is no mandatory -18 LUFS or LRA >= 6 LU gate. Lowering a loud premaster is a
transparent gain change, not intrinsically a loss of quality. Diagnose distortion
by comparing stages and listening; loudness/LRA alone cannot identify its cause.

### Diagnose clipping at the relevant stage

Floating-point buses can exceed 0 dBFS without numerical clipping. Their level
matters when feeding a nonlinear processor or an integer export. A bus peak,
master sum, or amount of peak attenuation alone cannot prove audible distortion.
Inspect processor input/output and compare a level-matched bypass before changing
the mix balance. Scaling an internal float bus down is different from recovering
an already clipped recording. Exported integer files must remain within range.

### Master spatial preset family — when to use which

The `master_mix.py` `MASTERING_PRESETS` dict has a family of `modern_rock_spatial*` variants. Each adds specific spatial moves on top of the base `modern_rock` preset:

| Preset | Sub-mono on side | Top side EQ | Stereo width | Exciter mix | Clipper | Use when |
|---|---|---|---|---|---|---|
| `modern_rock` | none | +1 dB shelf @ 8k | 1.0 | 0.10 | soft -2 dB | baseline modern rock master |
| `modern_rock_spatial` | HP @ 150 Hz | +2 dB shelf @ 8k | 1.0 | 0.10 | soft -2 dB | first spatial increment — adds side high-pass (sub-mono) |
| `modern_rock_spatial_v9` | HP @ 150 Hz | +2 dB shelf @ 8k | 1.0 | **0.12** | soft -2 dB | Leprous/Wheel crisp top direction |
| `modern_rock_spatial_v10` | HP @ 200 Hz | **+1 dB peak @ 2.5k + 3 dB shelf @ 8k** | **1.05** | 0.10 | soft -2 dB | full spatial: wider sub-mono, presence boost on side, stereo width bump |
| `modern_rock_spatial_dark` | HP @ 200 Hz | +1 dB shelf @ 8k only | 1.05 | **0.05** | soft -2 dB | spatial benefits BUT top dialed back — for ear-fatigue cases |
| `modern_rock_spatial_noclip` | HP @ 150 Hz | +2 dB shelf @ 8k | 1.0 | 0.10 | **NONE** | diagnostic — isolates clipper-induced distortion |

These are historical chain variants. Names and descriptions do not establish
fitness for Tool, Wheel, any genre, or a fatigue complaint. Read the actual
settings, choose the smallest relevant change, and compare against a minimally
processed baseline. Switching presets changes several variables and cannot
isolate a clipper fault unless all other settings are identical.

## Panning convention by band size

Rock-band stereo placement is conventional, not arbitrary. The default per-bus
pan values shipped in `tools/style_profiles/<name>.json` `default_bus_pan` are
applied automatically by `render_mix --generate-config --style NAME`. Below is
the rationale; deviate intentionally, not by accident.

**Common starting point:** center foundation buses, then assess intentional stereo sources and arrangement:
- `drums` parent: 0 (kick / snare / tom / hi-hat are individually placed via
  per-track pan, but the bus output is centered; the L/R stereo width comes
  from the OH and ROOM mic positions)
- `bass`: 0 (mono foundation — sub frequencies don't translate well off
  centre; a panned bass loses translation on mono playback systems)
- `vocal_lead`: 0 (the listener locates the singer in the centre of the
  stereo image; off-centre lead vocal is reserved for arrangement effects)

**Rhythm guitar buses are panned by number of guitarists.** Guitar tracks
named `GTR <player> <mic>` get one sub-bus per player (`gtr_<player>`); a `GTR`
name without a player token goes to the centred `gtr` sub-bus. The style
profile's `guitar_player_pans` list is assigned in sorted player order:

| Active rhythm guitar buses | Convention | Example pans |
|---|---|---|
| **1 guitarist** | Center (or slight ±0.2) | `[0]` |
| **2 guitarists** (classic 2-guitar wall) | Hard L/R | `[-0.85, +0.85]` |
| **3 guitarists** | Two wide + center | `[-0.85, +0.85, 0]` |
| **4+ guitarists** | Spread evenly | -0.7, -0.3, +0.3, +0.7 (consider whether overdubs are warranted) |

**Genre-specific spread for 2 guitarists (`guitar_player_pans`, project preferences):**

| Style | 1st player | 2nd player | Character |
|---|---|---|---|
| `punchy_modern_rock` | -1.0 | +1.0 | Very wide, aggressive |
| `modern_rock` | -0.85 | +0.85 | Standard wide rhythm wall |
| `tool_inspired` | -0.65 | +0.65 | Wide, dark image |
| `classic_rock` | -0.6 | +0.6 | Slightly narrower, band-feel |
| `pop` | -0.5 | +0.5 | Conservative, vocal-centric |
| `jazz_acoustic` | -0.3 | +0.3 | Narrow, intimate placement |
| `hip_hop` | 0 | 0 | Centered, drum-and-bass-led |

**When NOT to pan wide (even with multiple rhythm guitar buses):**

- The 2 guitar buses are the **same player in different sections** of the song
  (e.g. one session's `gtr_1` played the intro 11-30s and `gtr_2` the body 30-210s —
  they never sound simultaneously, so panning has nothing to separate, just
  produces "switching from left to right" between sections). Detection: scan
  the timeline activity per bus; if pairwise overlap < 10 % of either bus's
  total active time, treat as sequential and keep at center.
- The 2 buses are **doubling the same riff** with high correlation (> 0.8).
  Panning these wide just produces an L/R mono-ish image with comb filtering
  at the centre. Either commit to keeping them stacked (center) or true-double
  them with detuning / different takes before panning.
- A **single overdub for atmosphere/lead** (e.g. a third guitar bus with only 17 s of
  activity acting as a solo or accent) → center, not panned. Lead lines and
  solos traditionally sit center.

**Why this matters perceptually.** Two rhythm guitars panned 0/0 share the
centre with the bass and the kick. The centre channel becomes the dominant
"loudness column" in the mix, and the bass — being the most constant and
sub-heavy element — perceptually leads. Panning the guitars out moves their
mid-frequency content (200 Hz - 5 kHz) to the sides, clearing the centre for
bass/kick punch and adding stereo width. **The bass numerical level doesn't
change — but the perceived dominance drops** because the centre is no longer
crowded with mid-range guitar content masking the kick attack.

**Operational rule.** After `render_mix --render`, audit:
- `mix_config.json` `buses.<bus>.pan` for 2+ rhythm guitar buses
- If both are at 0, decide: are they sequential / doubling? Then leave at 0.
  Are they truly parallel different guitarists? Then pan per the style table.

### Drum-kit panning (audience perspective)

`_detect_pan` in render_mix recognises drum-kit pieces by name and assigns
audience-perspective pans automatically during `--generate-config`. The
convention is "what would a listener facing the kit on stage hear" — pitches
sweep across the stereo field, cymbals sit where they physically are on the
kit (drummer's right-hand cymbals end up on the audience's left side).

| Track name keyword | Pan | Reasoning |
|---|---|---|
| KICK / SN / SNARE / CRASH (generic) | 0 (center) | Foundation hits stay center |
| RACK TOM 1 | -0.4 | Highest-pitched tom, leftmost |
| RACK TOM 2 | -0.15 | Mid tom, slightly left |
| RACK TOM (generic) | -0.4 | Single rack tom defaults to left |
| FLOOR TOM | +0.5 | Lowest-pitched tom, right |
| HIHAT / HI-HAT | -0.2 | Drummer's right hand reaches audience-left |
| RIDE | +0.3 | Drummer's right-hand-far-reach cymbal |
| OH ... L / R | -0.7 / +0.7 | Overheads provide most of the kit's stereo image |
| ROOM ... L / R | -0.7 / +0.7 | Room mics widen ambience |

Sources: standard rock-mix references (Producer Society 2025, Sound on Sound
LCR articles, iZotope panning tips). Tweak via mix_config.json per-track
`pan` field if a particular session has unusual kit placement or the listener
prefers drummer-perspective (mirror image — high tom on right).

**Why the pitch sweep matters.** A kick + snare at center plus toms spread
L-to-R by pitch gives the listener a clear directional cue: the drummer's
fills move across the stereo image, not just amplitude-bouncing in the
center. Combined with OH L/R at hard ±0.7, the kit feels "set up in front
of you" rather than collapsed to a mono mid-band column.

**Critical: spread toms for the fill-sweep effect, not just pan-by-physical-position.**

If you have 3 toms (Rack 1, Rack 2, Floor), DO NOT pan Rack 1 and Rack 2
both to the same side just because they sit on the drummer's left
side physically. That makes a descending fill (R1 → R2 → Floor) go
LEFT → LEFT → RIGHT — a "jumped" placement, not a sweep.

For the **fill-sweep** effect, distribute toms across the full stereo
field:

| Tom | Audience-perspective pan | Drummer-perspective pan |
|---|---|---|
| RACK TOM 1 (highest) | **+0.65** (hard RIGHT) | -0.65 (hard LEFT) |
| RACK TOM 2 (mid) | **0.00** (CENTER) | 0.00 (CENTER) |
| FLOOR TOM (lowest) | **-0.65** (hard LEFT) | +0.65 (hard RIGHT) |

A descending fill (R1 → R2 → Floor) now smoothly sweeps RIGHT → center → LEFT.
A build-up fill (Floor → R2 → R1) sweeps LEFT → center → RIGHT.

Each tom occupies a distinct location in the stereo image, maximising the
"drum fill across the stereo field" perception. **Don't bunch toms on one
side** even if they physically sit together — the mix decision overrides
the physical position for spatial clarity.

**Audience vs drummer perspective — pick one and stay consistent across all drum mics.**

If you flip to audience perspective, the OH L/R close mic must ALSO flip:
- OH AEA L (originally captured drummer's LEFT side) → pan to +0.85 (audience right)
- OH AEA R (originally captured drummer's RIGHT side) → pan to -0.85 (audience left)

Otherwise the OH stereo image and the close-mic stereo image fight each
other (kick close panned 0, OH L panned right, kick component in OH R
panned left — net asymmetric kick image). Mixing audience-perspective
close mics with drummer-perspective OH = phase confusion + comb filtering.

**Modern prog metal default = audience perspective.** Tool, A Perfect
Circle, some classic prog use drummer perspective; Periphery, TesseracT,
Karnivool, Leprous, Wheel, Haken — audience perspective is the standard.

### Historical LCR pan preferences (2025)

Modern rock production frequently uses **hard pan (85-100%)** for double-
tracked rhythm guitars, often the LCR (Left-Center-Right) convention. The
shipped style profile values reflect this — `modern_rock` at ±0.85 and
`punchy_modern_rock` at ±1.0 (full LCR). Sources: Nail The Mix, Sound on
Sound, Producer Society, iZotope. Earlier internal numbers (±0.6) were
more conservative than that session's later preference and got bumped after a real-world
mix revealed the centre column was crowded by guitars + bass + drums.

---

## Frequency Bands (reference)

Used in analyze.py band RMS measurements:

| Band name | Range | Typical instruments |
|---|---|---|
| SUB | 20–60 Hz | kick sub, bass fundamental |
| BASS | 60–250 Hz | bass guitar body, kick punch, guitar low end |
| MID | 250–2000 Hz | vocals, guitar, snare body, most instrument fundamentals |
| HIGH | 2000–8000 Hz | presence, attack, string detail, cymbal body |
| AIR | 8000–20000 Hz | room, cymbal shimmer, high-freq artifacts |

**Interpreting band RMS in context:**
- Bass DI: energy in SUB + BASS, minimal above MID — normal
- Electric guitar amp: energy in BASS + MID + HIGH, little SUB
- Drum overhead: energy across MID + HIGH + AIR
- Kick in mic: SUB + BASS dominant, HIGH has the click attack
- Room mic: spread across all bands, lower overall RMS

---

## Noise Floor Reference

**IMPORTANT: the noise floor metric (5th percentile of frame RMS) is unreliable for distorted
instruments.** See "Distorted Instruments" section below before acting on any noise floor reading.

| Signal type | Typical noise floor | Notes |
|---|---|---|
| Clean DI recording | -80 to -90 dBFS | very clean |
| Good studio mic | -70 to -80 dBFS | acceptable |
| Live/location recording | -60 to -70 dBFS | some background noise |
| Problematic (clean instruments only) | above -60 dBFS | noise removal may help |

### Distorted instruments — noise floor is not a reliable metric

Distortion is a compression effect. It amplifies both the guitar signal AND everything else
in the chain (amp hum, pick noise, room reflections) by the same factor (2,000–3,000x for
high-gain). The result is that the 5th-percentile RMS of a distorted amp recording is
dominated by the amp's own sustain and character, not by unwanted noise.

**Typical values observed in Terido session (2026):**
- Clean DI guitar (no amp processing): noise floor -38 dBFS, DR 7–8 dB
- Distorted amp mic recording (SM57, ribbon): noise floor -18 to -19 dBFS, DR 2–3 dB

The -18 dBFS "noise floor" on the ribbon and SM57 tracks is NOT problematic noise —
it is the distorted guitar's sustained amp character between notes.

**Reliable way to distinguish actual noise from distorted guitar character:**

1. **Dynamic range (DR)**: DR 2–3 dB = heavy distortion (expected). DR 8+ dB = clean signal.
   Low DR alone does NOT indicate a noise problem — it indicates distortion.

2. **Frequency signature**: Electrical hum (ground loop) appears as a narrow-band peak at
   50 Hz or 60 Hz (and harmonics: 100, 150, 300 Hz) in a spectrogram — a steady horizontal
   line. Guitar distortion harmonics appear as an organized series at f, 2f, 3f, 4f... and
   are time-varying (they change with playing). Broadband noise appears as diffuse texture.

3. **DI vs mic comparison**: If the DI for the same instrument has a clean noise floor
   (-35 dBFS+) but the mic recording does not, the difference is the amp/mic chain, not
   actual noise in the recording environment.

### When NOT to apply noise removal to guitar/bass

- **Never apply broadband denoising to distorted guitar** — it alters harmonic content and
  destroys the organic distortion character. Digital artifacts are not acceptable substitutes.
- **Noise gates are safer**: they mute the signal during silence (between notes) without
  affecting the playing. They do NOT remove noise during playing — the hum/sustain remains
  when playing, but that is part of the sound.
- **High-pass filter at 80–100 Hz** is the safest fix for 50/60 Hz ground loop hum on
  guitar recordings — it cuts the hum frequency without affecting guitar tone.
- **Broadband denoising is appropriate for**: clean vocals, acoustic instruments, room mics,
  dialogue — signals where the noise is genuinely separate from the desired content.

---

## EQ — Filter Types

| Type | Shape | Use case |
|---|---|---|
| `highpass` | cuts below cutoff | remove rumble, bleed, sub-sonic content |
| `lowpass` | cuts above cutoff | roll off harsh highs, noise |
| `bandpass` | passes only the band | isolation, send routing |
| `notch` | deep narrow cut | electrical hum removal, narrow resonances |
| `peak` | bell boost or cut | tonal shaping, presence, mud reduction |
| `lowshelf` | boosts/cuts all below hz | body/warmth control, sub weight |
| `highshelf` | boosts/cuts all above hz | air/sparkle, high-end rolloff |

**Phase mode:** minimum phase (`sosfilt`) is the default. `--phase zero` runs the filter forward and backward with half the dB gain per pass, so the magnitude matches the requested filter with no phase shift. Zero-phase filtering is non-causal and therefore pre-rings ahead of transients; minimum-phase filtering does not pre-ring but shifts phase. Neither is universally better: compare on transient-rich material.

**Parameter notation:**
- `q` — bandwidth control. High Q (20-50) = narrow/surgical. Low Q (0.5-2) = broad/musical.
- `slope` — shelf steepness (1.0 = maximum, default).
- `order` — HP/LP filter order (2 = 12 dB/oct, 4 = 24 dB/oct).

---

## EQ — Instrument Starting Points

These are conservative starting points. Always analyze first, then apply preset, then re-analyze and adjust.
Presets live in `tools/presets/`. Apply with `apply_eq.py --preset NAME`.

### Bass guitar

| Source | HP | Problem cut | Character boost | High cut |
|---|---|---|---|---|
| DI | 40 Hz | 300-400 Hz (mud), Q 1.5, -3 dB | 100 Hz low shelf +1.5 dB (body) | — |
| Amp mic | 80 Hz | 350 Hz (boxiness), Q 1.5, -3 dB | — | 7 kHz LP (noise) |

- DI: full harmonic range, can boost carefully
- Amp mic: prefer cuts over boosts; boosting with mic increases feedback risk

### Electric guitar

| Source | HP | Problem cut | Character boost |
|---|---|---|---|
| Clean DI/amp | 150 Hz | 350 Hz (mud), Q 1.5, -2 dB | 3.5 kHz peak +2 dB (presence) |
| Distorted amp mic | 100 Hz | 400 Hz (cardboard), Q 1.5, -3 dB | 1 kHz subtle +1.5 dB (mid presence) |

- Distorted amp: noise floor metric unreliable (low DR expected from distortion compression)
- Do NOT apply broadband denoising to distorted guitar
- Roll off above 10 kHz on distorted amp mics (harsh sizzle)

### Kick drum

| Mic position | HP | Cut | Boost | Notes |
|---|---|---|---|---|
| Inside/beater | 30 Hz | 400 Hz (cardboard) -4 dB | 3 kHz (click) +3 dB | attack definition |
| Sub/outside | 20 Hz | 350 Hz (boxiness) -4 dB | 90 Hz low shelf +2 dB | body/weight |

### Snare drum

| Mic position | HP | Cut | Boost |
|---|---|---|---|
| Top | 80 Hz | 400 Hz (cardboard) -3 dB | 200 Hz +2 dB (body), 5 kHz +3 dB (crack) |
| Bottom | 600 Hz | — | 4 kHz +3 dB (wire rattle) |

- Bottom mic: aggressive HP at 600 Hz removes kick bleed; blend low in mix
- Both mics should use identical HP settings to avoid phase cancellation

### Drum overheads

| Mic type | HP | Cut | Boost/Roll-off |
|---|---|---|---|
| Condenser (U87, small diaphragm) | 120 Hz | 5 kHz -2 dB (harsh sizzle) | high shelf -3 dB above 12 kHz |
| Ribbon (AEA R84, Royer) | 80 Hz | — | high shelf +2 dB above 10 kHz (restores natural rolloff) |

- Condenser: roll off highs, do NOT boost air (harsh)
- Ribbon: safe to boost air shelf — ribbons don't distort the way condensers do

### Hi-hat and cymbals

| Instrument | HP | Cuts | Notes |
|---|---|---|---|
| Hi-hat | 200 Hz | 350 Hz -2 dB, 4 kHz -2 dB | metallic harshness in 2-8 kHz range |
| Crash/ride | 150 Hz | 350 Hz -2 dB, 5 kHz -1.5 dB | ride bell at 3-5 kHz — adjust per cymbal |

### Toms

| Mic | HP | Cut | Boost |
|---|---|---|---|
| Rack tom | 50 Hz | 500 Hz (boxiness) -3 dB | 150 Hz +2 dB (body), 4 kHz +2 dB (attack) |
| Floor tom | 50 Hz | 500 Hz -3 dB | 80 Hz +2 dB (deeper body), 4 kHz +2 dB |

### Room mics

- HP at 150 Hz (remove low-end mud — let close mics handle the low end)
- Cut 280 Hz -3 dB (undefined low-mid buildup)
- Compare EQ before and after compression: order changes what drives the detector and the resulting tone.

### Drum internal balance - kit relationships

Classification: engineering technique, not a required shell-to-cymbal level gap.
Balance close mics, overheads, and rooms by their contribution to the whole kit.
Overheads may carry essential shell body and space. Do not high-pass them at a
fixed frequency or mute recorded rooms merely because the genre is metal.

Compare snare/overheads, kick/overheads, kick/snare, then toms and rooms. Test
polarity and timing alternatives in context and in mono; maximum correlation
is not necessarily the best sound. Joe Barresi describes this contextual
approach in [his Tool interview](https://www.waves.com/in-the-studio-with-tool-and-evil-joe-barresi)
(2019-11-07; verified 2026-09-09; high confidence as a description of his method,
not a universal rule). Recording geometry and arrangement are exceptions to any
fixed alignment prescription.

Use matched passages to assess attack, body, sustain, cymbal decay, and room
contribution. There is no universal 10-12 dB shell/OH relationship. Integrated
track loudness also depends on how often an instrument plays; do not average it
into an automatic fader correction.

## True Peak vs Peak

Sample peak measures stored samples; reconstructed waveform peaks may be higher.
Use the agreed delivery ceiling and current platform recommendations from the
source table. This project's premaster peak setting is a preference. A single
-2 dBTP rule does not apply to every platform, format, and loudness choice.

Measure decoded output and, when applicable, actual codec round trips. An
oversampled WAV peak estimate does not test an encoder or prove audible clarity.

## Reverb in a Rock Mix

### Track-level vs bus-level reverb

**Bus-level reverb (correct for guitars):** All tracks in a bus share one reverb return. They sound like they're in the same room, there's no reverb accumulation, and the wet level is controlled from one place. Apply via `render_mix.py` `reverb_send` on the bus.

**Track-level reverb (correct for selective drums):** When only specific drum elements should have reverb (toms yes, kick no), bus reverb can't be used — it would reverb the kick too. Apply via `apply_reverb.py` insert mode on individual stem files.

**Never apply reverb per-track to a bus of many similar instruments** (e.g. 10+ guitar tracks). The reverb tails accumulate in the bus sum and create a washed-out, muddy result. Even with a reduced wet level (e.g. 0.10 instead of 0.18), the accumulation still muddies the sound.

### Reverb settings by instrument

- **Snare (gated plate):** room_size=0.70, pre_delay=15ms, hp=400Hz, gate hold=300ms, release=70ms. Long reverb + gate = punch + size simultaneously.
- **Toms (room):** room_size=0.28, damping=0.60, pre_delay=8ms, hp=300Hz. Small room for cohesion, high damping. HP critical — low-end in reverb kills punch.
- **Guitar bus (room):** room_size=0.45, damping=0.50, pre_delay=20ms, hp=150Hz, wet=0.15 on bus.
- **Kick, bass:** no reverb. Low-frequency reverb destroys punch and muddies the low end.

### Choosing between algorithmic and convolution

The algorithmic Freeverb engine (the default for `apply_reverb`) is fast,
parameter-driven, and gives a coloured, characterful tail that works well
on snares and plates. It can sound metallic on long halls though.

The convolution engine (`--ir` for a custom IR, or `--ir-preset` for the
built-in pack) is slower but more transparent. Use it when:

- You need a long hall tail without metallic ringing (`--ir-preset hall_concert`)
- The mix asks for a specific room character (`--ir-preset room_live`)
- A guitar wants spring reverb (`--ir-preset spring_guitar`)
- The artist hands you a custom IR they recorded

The shipped IR pack (`tools/irs/`) is synthetic — generated by
`tools/generate_irs.py` from noise + decay envelopes + spectral shaping.
Six IRs cover the common cases: plate_short, plate_long, room_tight,
room_live, hall_concert, spring_guitar. Re-run the generator if you ever
want different characteristics; nothing else depends on the file contents.

### BPM-synced pre-delay

Pre-delay rhythmically aligned to the song tempo locks the reverb into
the groove. `apply_reverb --bpm 184 --pre-delay-division sixteenth` sets
the pre-delay to one sixteenth note at 184 BPM = 81.5 ms. Useful values:

| Tempo | Eighth | Sixteenth | Triplet-eighth |
|---|---|---|---|
| 80 BPM | 375 ms | 187.5 ms | 250 ms |
| 120 BPM | 250 ms | 125 ms | 166.7 ms |
| 140 BPM | 214 ms | 107 ms | 143 ms |
| 184 BPM | 163 ms | 81.5 ms | 109 ms |

Sixteenth-note pre-delay on snare reverb is a common Nashville move —
makes the snare feel locked into the groove. Eighth-note pre-delay on
vocal reverb is the classic Phil Collins / Bowie sound.

### Sidechain reverb (pumping pattern)

`apply_reverb --sidechain kick.wav --sc-depth -12` ducks the reverb
tail every time the sidechain hits. The reverb breathes with the song
instead of washing over transients. Use cases:

- Snare/vocal plate ducked by kick: keeps the kick attack clean, lets
  the reverb fade in between hits
- Hall on a sparse stem ducked by the mix bus: makes the reverb sit
  behind the loudest moments

Optional `--sc-hp` and `--sc-lp` band-pass the sidechain trigger to
isolate the kick beater click (e.g. `--sc-hp 60 --sc-lp 200`) — this
prevents the bass from also triggering the ducking on kick-bass
overlapping pieces.

### Master bus reverb — subtle cohesion

A modern (2024-2026) mastering technique: apply a very subtle reverb to
the entire stereo mix at the mastering stage to add cohesion and a sense
of "one room". The principle: **feel** the reverb, don't **hear** it.

**Parameters that work:**
- **Mode**: INSERT (dry + small wet, NOT send-only) — `--dry 1.0 --wet 0.04..0.08`
- **Preset**: `hall_ambient` or `room_drums` (medium decay, not too long)
- **HP @ 300 Hz**: critical — keeps reverb tail out of the bass body, no mud
- **Pre-delay 60-80 ms**: pushes the verb tail behind the dry signal, preserves dry transient clarity
- **Wet level cap**: 0.08 is the perceptual ceiling — above this it becomes audible-as-effect, defeating the "cohesion" purpose

**Practical workflow:**
1. Render the mix as premaster (peak -3 dBFS)
2. Apply reverb: `apply_reverb premaster.wav --preset hall_ambient --pre-delay 60 --hp 300 --wet 0.07 --dry 1.0 -o <dir>`
3. Master the reverb-treated premaster with `master_mix.py` — the LUFS norm + limiter will compensate for any slight level shift from the wet content
4. Compare against the no-master-reverb master at the same -14 LUFS — the wet version should sound slightly more "glued" without obvious reverb tails

**When master reverb backfires (terido v11 lesson):**
If the mix has cumulative top-emphasis (e.g. side highshelf +3 dB @ 8k, guitar EQ +1.5 dB @ 3.5k, exciter mix 0.10), adding master reverb at wet 0.07 stacks 300 Hz+ content into the master clipper at +5-7 dB above threshold → audible "overdrive" perception. Fix: drop wet to 0.03-0.04 OR reduce the cumulative top emphasis upstream OR use the `modern_rock_spatial_dark` master preset (drops side highshelf +3 → +1, drops master EQ shelf +1 → +0.5, halves exciter mix).

**Sources:**
- [Music Guy Mixing — How to Use Reverb on the Master Bus](https://www.musicguymixing.com/reverb-on-master/)
- [Mastering.com — Reverb Layering Strategies](https://mastering.com/reverb-layering-strategies-how-to-combine-different-reverb-types-to-create-a-cohesive-3d-space/)

---

## Bass Amp Simulation

### Ampeg SVT 8x10 cabinet frequency response

Spec: -3dB at 58Hz and 5kHz. Characteristic hump at 100-125Hz.

EQ model for apply_amp.py:
- HP @40Hz — sub-rumble removal
- Low shelf +3dB @120Hz — cabinet body (the signature Ampeg "thump")
- Mid peak +2dB @800Hz Q=2.0 — midrange grind (optional, for SVT character)
- LP @5000Hz — speaker rolloff (-3dB point per Ampeg spec)

### Slap bass EQ

Slap bass needs a different EQ than fingerstyle:
- Low shelf +4dB @80Hz — thumb "thump" (lower frequency than SVT's 120Hz)
- Mid cut -3dB @700Hz Q=1.5 — hollow mid scoop (characteristic slap sound)
- LP @8000Hz (not 5kHz) — lets string "pop" and click through; critical for slap articulation

### Tube vs tape saturation

**Tube (asymmetric tanh):** Positive half clips harder. Generates predominantly even-order harmonics (2nd, 4th). Warm, colored. Good for bass, vocal, individual instruments.

**Tape (symmetric tanh):** Both halves clip equally. Generates odd-order harmonics (3rd, 5th) plus some even. Less colored than tube. Good for bus saturation (drums, guitars) — adds cohesion without tonal shift.

Both are RMS-normalized in this pipeline. Equal RMS does not guarantee equal perceived loudness after their spectral changes; level-match auditions.

---

## Mastering Workflow and Philosophy

Mastering is the final pass after the mix is bounced to a stereo file. It's
a different mindset than mixing — the mix engineer balances the parts; the
master engineer treats the song as one finished object and prepares it for
delivery.

This project treats mix and master as **two separate phases**:

| Phase | Input | Output | Tool |
|---|---|---|---|
| Mix | stems + mix_config.json | mix.wav | `render_mix.py` |
| Master | mix.wav | master_<format>.wav | `master_mix.py` |

In premaster mode (default) the `render_mix.py` master chain is only glue
comp + EQ + peak normalization: **the mix engineer's polish**, not the master
pass. The legacy chain (`premaster_mode: false`) adds guarded clipper, guarded
M/S, LUFS norm and a true-peak limiter; avoid it when the mix will be mastered. The actual mastering pass is `master_mix.py`, run
separately on the bounced stereo file with format-specific delivery
targets.

### Why two phases

1. **Independent iteration**: tweak master EQ without re-rendering 56 stems.
2. **Multi-format delivery**: one mix → many masters (Spotify, Apple, CD,
   vinyl pre, etc.) with format-specific LUFS / true peak targets.
3. **Compatibility with external mixes**: if someone hands you a mix.wav,
   you can master it without their session.
4. **Match real-world workflow**: mix engineers and master engineers are
   usually different people; the tools should reflect that boundary.

### Delivery presets and requirements

The source table at the top is authoritative for the claims verified in this
review. Values in `master_mix.FORMAT_PRESETS` are project defaults unless explicitly
classified as a requirement. Spotify's playback normalization level is not a
mandatory musical loudness target. Apple does not universally require -16 LUFS.
One approved streaming master often suffices; do not automatically make a different
artistic master for each platform's playback setting.

CD export resamples to 44.1 kHz and quantizes to 16 bits with dither. Its loudness
is an artistic choice. For vinyl, obtain the cutting engineer's specification;
a generic LUFS preset cannot establish suitability for a particular cut. The
broadcast preset targets -23 LUFS with a conservative -2 dBTP ceiling; confirm
the contracted delivery specification and permitted tolerance.

### Mastering chain presets

`master_mix.py` ships eleven chain templates (the *what to do* part, distinct
from the format target *how loud* part): the six below plus the
`modern_rock_spatial*` family described earlier.

| Preset | Chain | Use when |
|---|---|---|
| `gentle` | comp only (1.5:1, gentle) | Acoustic, jazz, classical-leaning rock. Preserves dynamics. |
| `modern_rock` | EQ + glue comp + exciter + M/S side highshelf + soft clip | Competitive rock loudness with audible LUFS lift and a wider top. |
| `modern_rock_mb` | EQ + 3-band multiband + exciter + M/S side highshelf + stereo width 1.05 + soft clip | Modern rock with tighter band-by-band dynamics. Replaces glue comp with multiband — better controlled low end. |
| `pop` | EQ (bright) + comp + exciter + M/S side highshelf + width 1.1 + soft clip | Bright top, present mids, slightly wider image. |
| `hip_hop` | EQ (sub boost) + comp + exciter + width 0.95 (slightly narrower) + hard clip | Sub weight, impact, mono-leaning width to keep the 808 centred. |
| `transparent` | LUFS norm + true-peak limiter only | When the mix doesn't need master tone. |

### Optional chain steps and when to use them

The chain presets above wire up these steps for you, but you can override
any of them via a custom preset JSON:

- **Multiband compressor** (`multiband` field): replaces or supplements the
  glue comp. Use when the bass needs tighter control than the mids/highs
  can tolerate. The `modern_rock_mb` preset is the typical example.
- **M/S processing** (`ms` field): independent EQ and gain for the mid
  (mono-summed) and side (stereo-difference) channels. Standard mastering
  tricks: +1-2 dB highshelf on the side for "shine", small mid gain cut
  to push the kick/bass to the sides ratio. Avoid side boost > +2 dB
  unless the mix is narrow to begin with.
- **Stereo width** (`stereo_width` field, scalar): scales the side signal.
  1.0 = no change. 1.05-1.15 = subtle widening. 0.95 = slightly narrower
  (good for sub-heavy genres). 0.0 = mono. Width > 1.3 risks
  mono-compatibility.
- **Vinyl elliptical EQ**: zero-phase side high-pass (sub-mono) below
  ~150 Hz. Opt-in with `--vinyl-elliptical [HZ]` on the `vinyl_pre` format;
  off by default because the cutting engineer normally decides it.

### Waveform true peak and codec audition

The health tool reports 4x and 8x oversampled waveform peaks. Neither is a codec
simulation. `codec_roundtrip.py` encodes with the local ffmpeg build, decodes,
and measures the decoded peaks and loudness; listening to its decoded files is
still a separate, human codec review. Overshoot confined to the first or last
milliseconds usually comes from an abrupt full-scale file start or end.

### Punch index and phase diagnostics

The punch index is a short/long envelope statistic. It cannot establish audible
punch, fatigue, or the processing history of a recording. The same value can
arise from different arrangements. Review changes on the same passage at matched
loudness, especially drum attacks against sustained guitars and bass.

Per-band correlation and M/S width locate possible mono-translation issues.
They do not require mono sub-bass or a fixed amount of high-frequency width for
every song. Listen for loss of important elements in mono; relate measurements
to source microphones and intentional stereo effects before changing them.

### Compression-history heuristic

Low crest together with high sample peaks may suggest heavy processing, but no
waveform summary proves the source's history. Low LRA alone is not evidence of
compression. Compare the original and processed audio and ask about intentional
upstream processing before adding another mastering pass.

### Reference deck

A single reference can be misleading — if the reference happens to be
extra-bright or has a specific vocal mix, the comparison drifts that way.
Mastering engineers use **multiple references** (a "deck") and look at the
average target spectrum. `master_health.py --reference ref1.wav ref2.wav
ref3.wav` averages all references' 1/3-octave PSDs and reports region-level
deltas against your master. 3-5 references is the typical deck size.

### Technical checks and listening decisions

| Check | Role | Interpretation |
|---|---|---|
| File format and peak ceiling | Technical check | Correct violations of the agreed delivery specification. |
| Integrated loudness | Requirement only when contracted; otherwise advisory | Playback normalization is not rejection of the master. |
| Codec behavior | Measured by `codec_roundtrip.py` on a local encoder build | Waveform oversampling cannot certify encoded playback; platform encoders differ from the local build. |
| Phase, width and punch | Listening prompts | Investigate audible mono cancellation or transient loss; no universal threshold proves quality. |
| Compression history | Hypothesis | Metrics cannot establish which processing happened upstream. |
| Reference spectrum | Tonal guide | Arrangement, vocals, instrumentation and era can explain differences. |

The aggregate health result covers the implemented technical checks. It does not
certify standards conformance, codec safety, or listening approval.

---

## Master Bus Chain Order

Processing order matters. This is the `render_mix.py` master chain in the
legacy `premaster_mode: false` mode (each step optional, guarded ones skip if
their relevance_check fails). Premaster mode stops after step 7 and
peak-normalizes instead of steps 8-9:

1. **Bus saturation** (per bus, before summing to master)
2. **Bus parallel saturation** (guarded, drum bus only — relevance_check: crest > 10 dB AND LRA > 4 LU)
3. **Master sum** (all top-level buses)
4. **Master glue compressor** (2:1, slow-ish attack, catches sustained program material)
5. **Master clipper** (guarded, soft cubic or hard — relevance_check: sample peak ≥ -10 dBFS AND LRA ≥ 4 LU)
6. **M/S processing** (guarded — independent mid/side EQ + gain — relevance_check: width ≥ 0.05)
7. **Master EQ** (zero-phase — HP@30Hz + gentle high shelf typical)
8. **LUFS normalization** (target -14 LUFS for streaming)
9. **True-peak brickwall limiter** (pedalboard `BrickwallLimiter`: stereo-linked, 5 ms lookahead, 4x true-peak detection, no makeup gain) followed by an 8x true-peak verification and static safety trim. pedalboard's older `Limiter` class is not used: it adds a fixed 4:1 stage above -10 dBFS, automatic makeup gain and a 0 dBFS hard clip.

For the **master_mix.py** pass on a finished stereo mix, the chain is
slightly different (more aggressive, format-aware) — see "Mastering
Workflow and Philosophy" above.

### Master glue compressor settings (pedalboard Compressor notes)

Pedalboard's Compressor uses peak detection per channel; the tools derive one stereo-linked gain curve from it (`_dsp.linked_gain`) so a one-sided hit does not shift the image. With slow attack (>20ms), short transients pass through and the measured peak GR appears 0. For program material compression, use:
- threshold: -10 dBFS (works with typical -12 to -9 LUFS pre-norm signals)
- attack: 10ms (catches sustained peaks while letting some transient through)
- ratio: 2:1
- release: 300ms
- Expected GR: -0.5 to -1.0 LUFS on a dense rock mix

---

## Creative processing - relevance and audition

Effects can change weight, density, width, texture, and space. Choose them for
an identified problem or creative intention, not a promise of a hit or an
assumption that more processing creates professionalism. Retain a minimally
processed baseline for comparisons.

A tool's `relevance_check` is a conservative project heuristic. It may skip a
process, but passing it does not establish benefit. Refer to the current tool
report for its actual thresholds. An override needs a documented reason and
matched listening review; existing user authorization applies.

Compare the same passage before and after each consequential intervention.
Measurements show what changed. Listening or actual human feedback establishes
whether that change served the intention. A pumping flag that appears after
processing only means a detector threshold was crossed. It cannot distinguish
an artifact from a musical pulse by itself.

Keep processing purposeful. No fixed effect count or minimum LRA guarantees a
good result. Do not alter upstream compression to make a downstream guard pass.
If a change produces no intended benefit, revert it; if listening is unavailable,
keep it provisional instead of inventing an audible verdict.

## Reading the New Analysis Metrics

`analyze.py` produces several metrics introduced for the make-it-hit and re-analyze workflows. They are not in the textbooks; here is how to read them.

### frequency_bands_crest_db

Peak-to-RMS within each 5-band region (sub/low/mid/high/air). Tells you which bands have headroom for dynamic processing and which are already squashed.

| Band crest | Meaning | Action |
|---|---|---|
| > 18 dB | Loose / transient-rich | Multiband or parallel sat on this band has room to work |
| 8-15 dB | Healthy | Normal range, no special action |
| < 6 dB | Squashed | Avoid multiband / parallel sat on this band — it'll just smear without controlling anything |

Use case: deciding the per-band ratios for a multiband chain. If the low-band crest is 22 dB but the high-band crest is 5 dB, you want a tight low-band comp and almost no high-band comp.

### pumping (pumping_detected, pump_rate_hz, modulation_depth_db, lf_excess_db, active_frame_ratio)

Detects 1-5 Hz envelope modulation. Two criteria both must trigger for `pumping_detected: true`:

1. `modulation_depth_db ≥ 5` — the envelope swings by 5+ dB peak-to-trough on active frames (RMS > -40 dBFS). The active-frame gate fixes a previous bug: silent gaps between hits dragged p5 to ~0 and falsely hid pumping on intermittent material like kick mics and guitar with verse rests.
2. `lf_excess_db ≥ 6` — the 1-5 Hz peak in the envelope's spectrum exceeds the 5-15 Hz reference by 6+ dB. Synthetic continuous pumping signals show excess > 20 dB; real-world musical content typically shows 5-15 dB.

When `pumping_detected: true`, disambiguate before reverting any upstream step:

1. **Did the flag appear AFTER a comp/multiband/clipper step?** Compare the analysis JSON from before and after. False to true means a detector threshold crossing; audition before attributing an audible artifact.
2. **Is `pump_rate_hz` close to song-tempo quarters/eighths?** At 120 BPM: quarter = 2.0 Hz, eighth = 4.0 Hz. At 82 BPM: quarter = 1.37 Hz. If pump_rate matches the groove pulse, it is likely **musical strumming/groove**, not comp artifact.
3. **What stem is it on?** Guitar (especially rhythm), bass, drum buses → typically musical pulse. Vocal, sustained pad, master mix → comp artifact more likely.
4. **Depth vs excess profile.** High depth + moderate excess (depth 18 dB, excess 5 dB) = musical pulse. High depth + high excess (depth 8 dB, excess 30 dB) = comp artifact.

If the conclusion is "musical pulse, not artifact": **say so explicitly and do NOT revert**. Note it in the session summary so the next analysis pass doesn't re-flag it as a problem.

### true_peak_dbfs vs sample_peak_dbfs

`analyze.py` reports both:
- `sample_peak_dbfs` — the raw maximum sample value, naive
- `true_peak_dbfs` — 4×-oversampled, ITU-R BS.1770-4 style

The difference is the inter-sample peak (ISP). For low-frequency signals they are nearly identical. For HF content (cymbals, distorted guitar, snare crack) the true peak can sit 0.5-3 dB above the sample peak. After codec encoding (Spotify Ogg/Vorbis, Apple AAC), the encoded signal's inter-sample peak can climb further, occasionally pushing samples above 0 dBFS.

**For stems: the difference rarely matters.** For master delivery: -1 dBTP is the common streaming recommendation, -2 dBTP for masters louder than -14 LUFS; actual codec playback still needs testing. The limiters in `master_mix.py` and the legacy `render_mix.py` chain verify the result at 8x oversampling and apply a static trim if the ceiling is exceeded.

### onsets_sec, tempo_bpm, estimated_key (rhythm & tonal context)

Three top-level fields in `analysis.json` that give time-domain and tonal context, useful when an EQ / comp / FX choice depends on rhythmic or harmonic content beyond the basic loudness numbers.

| Field | Type | What it is | When to use |
|---|---|---|---|
| `onsets_sec` | list of float seconds | Onset times from librosa onset detection. Same detector as `transient_density_per_sec`, exposed as a raw list. | Identify rhythmic structure; pair-wise stem alignment; precise "uneven playing" detection per onset; visual debugging. |
| `tempo_bpm` | float (or `null`) | librosa `beat_track` estimate. Returns `null` for clips shorter than ~4 s or when the estimate is unstable / out of range (30–300 BPM). | Pick BPM-synced division for `apply_reverb --pre-delay-division` or `apply_delay --bpm`. Sanity-check against the human-known tempo (drummer's clicktrack). |
| `estimated_key` | `{key, mode, confidence}` | Krumhansl-Schmuckler key estimation on `chroma_stft`. Confidence is the Pearson correlation (-1..1) with the best reference profile; white noise scores about 0.25. | Decide whether a tonal mid-EQ move should track the song's key (e.g. boosting 220 Hz on an A-minor track lines up with the root). Drums / overheads / noise give low values — `< 0.5` means "no reliable key", ignore. |

The cost of computing these is modest (~+10% on `analyze.py`). They are computed unconditionally on every analyze pass — no opt-in flag needed.

### envelopes (RMS / LUFS short-term / spectral flux per second)

Three time-series at 1-second resolution stored under `analysis.envelopes`:

| Subfield | Unit | What it is |
|---|---|---|
| `rms_db_per_second` | dBFS | RMS level of each 1-second slice. Quick "section loudness map" — quiet intro vs. loud chorus is visible at a glance. |
| `lufs_short_term` | LUFS | BS.1770 short-term loudness (3 s window, 1 s step). Standardised perceptual loudness curve. Slightly different shape than RMS because the K-weighting attenuates sub-bass and emphasises 2–4 kHz. |
| `spectral_flux_per_second` | arbitrary (librosa onset strength units, per-frame mean) | Energy change in the spectrum per second. Peaks at section boundaries (intro → verse → chorus) where the instrumentation changes substantially. |

**Use cases:**

- **Section detection**: scan `rms_db_per_second` or `lufs_short_term` for sustained shifts ≥ 3 dB. Each shift marks a verse / chorus / bridge boundary.
- **Spectral flux peaks** flag the same boundaries from a different angle — useful to confirm a section change vs. a level change inside the same section.
- **Mix consistency check**: if `lufs_short_term` ranges 8+ LU on a master that's supposed to be modern-rock-loud (LRA target 4–9), something is too dynamic.

The arrays are JSON-array-valued, which makes them safe to pretty-print but **noisy** in `analysis.json` — they account for ~10–20 KB per stem on a 400-second take. Worth it for the analytical value.

### Health reports and delivery readiness

Health tools expose technical checks separately from musical diagnostics and
always leave listening review pending. Missing format requirements are reported
as unassessed, not passed. Style scores measure preference similarity only.

- Correct a failed agreed technical requirement before approved delivery.
- Treat tonal, envelope, width, and masking warnings as investigation prompts.
- Do not apply inverse reference EQ or chase a green score automatically.
- Export drafts for review without representing them as final approvals.
- Run `review_delivery.py` on the actual export, with the agreed peak ceiling and
  applicable format/contractual loudness requirements. Read its `delivery_ready`
  field; narrow or stale feedback cannot satisfy full-song approval.

## Stem Analysis — Interpreting Metrics

`analyze.py` produces `analysis.json` and `spectrogram.txt`. The STATS SUMMARY block at the bottom of `spectrogram.txt` condenses the key metrics. Interpret them together — no single number tells the full story.

### Loudness Range (LRA)

EBU R128 Loudness Range: how much the loudness varies across the file, in LU.

| Context | Typical LRA | Notes |
|---|---|---|
| Rock mix (delivered) | 5–12 LU | healthy dynamics |
| Rock mix (pre-master) | 8–16 LU | more dynamics available before limiting |
| Pop/EDM mix | 3–7 LU | heavily compressed by design |
| Acoustic/jazz | 12–25 LU | wide natural dynamics |
| Brick-walled | < 2 LU | no dynamic variation left |

LRA < 3 LU can reflect a dense arrangement, intentional consistency, or processing. Compare stages and listen before changing the compressor.

### Crest Factor

Peak-to-RMS ratio in dB. Higher = more dynamic material (transient peaks well above sustained body).

| Range | Meaning |
|---|---|
| < 8 dB | Over-compressed / brick-walled |
| 8–12 dB | Typical loud modern rock — compressed but usable |
| 12–18 dB | Healthy dynamics — transients intact |
| > 18 dB | Very dynamic — minimal compression applied |

Crest factor and LRA should agree directionally. Low LRA + high crest factor = short transients survived the limiter but mid-term dynamics are gone. Low LRA + low crest factor = full brick-wall.

### Stereo Metrics (stereo files only)

**Balance (dB):** RMS difference between L and R channels. Positive = L > R.
- 0 dB: perfect balance
- ±0.5 dB: imperceptible in most contexts
- > ±1.5 dB: noticeable tilt — check panning choices
- > ±3 dB: strong imbalance — likely a deliberate hard pan or a missing channel

**LR Correlation:** Pearson correlation between left and right.
- +1.0: perfect mono
- +0.7 to +0.95: typical rock mix (bass and kick are mono, stereo elements present)
- +0.4 to +0.7: wide stereo (heavy panning, stereo reverb, chorus)
- 0 to +0.4: uncorrelated — unusual for a full mix; check for phase issues
- Negative: out-of-phase channels — will cancel in mono; always investigate

**M/S Width Ratio (Side / Mid energy):**
- 0–0.1: near-mono
- 0.1–0.3: moderate width — typical mixed rock with panned guitars
- 0.3–0.6: wide
- > 0.6: very wide — check mono compatibility

### Transient Density (onsets/sec)

Detected onset events per second of audio. Useful for characterizing playing style and evenness.

| Source | Typical range |
|---|---|
| Kick drum alone | 1–3 /s (one per beat at 60–120 BPM) |
| Full drum kit | 5–15 /s |
| Strummed guitar | 2–6 /s |
| Slap bass (even playing) | 2–4 /s |
| Full rock mix | 3–10 /s |

**Detecting uneven playing:** run analyze on short sections (e.g. intro vs verse) and compare densities. A section with 1.5 /s followed by 4.0 /s in the same phrase length signals the performer hit much harder in one section. Use this to calibrate compression — the bigger the gap, the more aggressive the compressor threshold needs to be.

### Spectral Centroid (Hz)

Energy-weighted center of frequency content. Single-number brightness indicator.

| Source | Typical range |
|---|---|
| Kick drum alone | 200–800 Hz |
| Bass guitar | 400–900 Hz |
| Full rock mix | 1500–3500 Hz |
| Guitar-forward rock mix | 2500–4500 Hz |
| Bright / cymbal-heavy mix | 4000–6000 Hz |

**Use for EQ verification:** track centroid before and after EQ. A high-shelf boost should raise the centroid; a mid cut should lower it. If centroid doesn't change after applying an EQ band, that band may have had no energy to affect.

### Transient Profile (percussive instruments only)

Two metrics measure whether transient shaping would help:

**Transient prominence (dB):** attack peak (first 5ms after onset) vs. sustain RMS (5–150ms).
Measures how much the initial hit stands out from the body of the sound.

| Range | Meaning | Action |
|---|---|---|
| > 8 dB | Strong attack — already punchy | No Attack+ needed |
| 4–8 dB | Moderate — attack present but not dominant | Consider Attack+ if mix context buries it |
| < 4 dB | Weak attack — sustain dominates | Attack+ transient shaping likely helps |

**Decay time (ms):** time from envelope peak to -20 dB below peak (5ms RMS-smoothed envelope).

| Instrument | Tight | Normal | Long (may need Sustain-) |
|---|---|---|---|
| Kick | < 50ms | 50–150ms | > 150ms |
| Snare | < 40ms | 40–100ms | > 100ms |
| Tom | < 80ms | 80–200ms | > 200ms |

**Instrument-specific interpretation:**

| Instrument | Use prominence | Use decay | Use prominence std |
|---|---|---|---|
| Kick, snare, toms | Yes — transient shaper decision | Yes — tightness check | Yes — playing consistency |
| Slap bass | No (naturally low, ~4 dB) | No | Yes — primary unevenness indicator |
| Fingerstyle bass, distorted guitar | No | No | No — always low/noisy, ignore |
| Acoustic guitar, clean DI | Somewhat | No | Yes — pick consistency |
| Overhead, room mic | No | No | No — multi-instrument sum |

**Std as an unevenness indicator (all instruments):**
If `transient_prominence_std_db` > half of `transient_prominence_db`, playing is uneven.
Example: BASS DI slap — mean 4.3 dB, std 8.8 dB → std > mean → highly uneven hits.
This complements transient_density: density tells you *how often* onsets occur, std tells you *how consistently* each hit lands.

**When NOT to use these metrics:**
- Heavily compressed stems: compression artificially lowers prominence — measure the pre-comp assembled.wav.
- Overhead/room mics: measure the room, not individual drums — not meaningful.

**Terido session reference (2026-05-16, raw assembled.wav):**
- KICK IN: prominence 10.9 dB, decay 31.7ms → strong attack, tight → no shaping needed
- SN TOP: prominence 12.1 dB, decay 46.4ms → strong crack, normal decay → no shaping needed
- BASS DI: prominence 4.3 dB, decay 72ms → expected for sustained instrument, ignore

### Giving recommendations from analysis

Read all metrics together and give a verdict — one of:

1. **Everything within normal range — no action needed.** State which metrics confirm this and why.
2. **One specific problem identified.** Name the metric, the observed value, the expected range, and the single most likely fix.
3. **Multiple issues — prioritize.** Address the most audible or most likely root cause first; one change at a time.

Never recommend a change just because a value is "not ideal". The ear is the final arbiter. Always note when a value could have multiple explanations (e.g. low LRA could be over-limiting OR could be correct for a dense rock arrangement).

---

## LUFS vs. Perceived Loudness — Measurement Equality ≠ Perceptual Equality

A common pitfall when balancing busses with `bus_balance.py`: two busses can
sit at the *exact same* integrated LUFS yet sound very different in loudness
to a listener. This is not a bug in the measurement — it's a known limitation
of the BS.1770 K-weighting model when applied to spectrally dissimilar
content (bass vs midrange-dominated guitar, for example). When the user says
"the bass sounds louder than the guitar" but the LUFS values are identical,
**trust the ear** — adjust the bus volume down by 1–3 dB and re-render.

### Why this happens

| Factor | Mechanism |
|---|---|
| **K-weighting low-frequency rolloff** | BS.1770 applies a 12 dB/octave high-pass at ~100 Hz. Sub/low-bass energy is *under-counted* by the meter — the bass track has more acoustic energy than its LUFS number suggests. |
| **K-weighting mid-shelf** | A ~4 dB high-shelf boost above 2 kHz over-emphasises the midrange-presence band where guitars dominate. The guitar's LUFS reading is *inflated* relative to its perceived contribution. |
| **Consumer headphone bias** | Most consumer/audiophile headphones (Harman target, Sennheiser/AKG/Sony tunings) boost 50–100 Hz by +3 to +6 dB vs flat. On headphones the bass perception is amplified beyond what speakers (or the meter) show. |
| **Tactile loudness** | Sub-bass is not only heard but felt (chest resonance, head/skull vibration). The meter doesn't measure tactile energy; the listener integrates it into perceived loudness. |
| **Sustained vs transient content** | Bass plays sustained notes (long average loudness); guitar plays punctuated transients (high peaks, quiet inter-onsets). The integrated LUFS averages both to the same number, but the listener's loudness perception tracks the *sustained* energy. |

### Practical rule of thumb

When the measured LUFS of two busses is equal but they don't *sound* equal:

| Spectral difference | Typical perception bias | Suggested compensation |
|---|---|---|
| Bass-heavy bus (sub + low dominant) vs midrange bus | bass perceived +1 to +3 dB louder | drop bass bus by 1–3 dB |
| Kick + bass bus vs vocal bus | kick/bass perceived louder than the meter shows | drop low-end stems by 1–2 dB |
| Hard-panned (±0.7+) bus vs centered bus | panned bus perceived slightly quieter than centered (constant-power pan is true at LUFS level but binaural integration favours centered content) | optional +0.5–1 dB on the panned bus *if mono compatibility matters* |

### What this means for `style_check.py` profiles

Profile targets are project preferences measured at the profile's loudness.
Matching them does not establish correct tone, audible balance, genre identity,
or reference equivalence. Inspect actual in-mix contributions and compare at
matched loudness. Keep user preferences when profile scores disagree.

### Field-test reference: terido v4 → v5

After v4 hit drum = bass = guitar = -19 LUFS exactly on `bus_balance`, the
listener reported the bass still felt 1–2 dB hotter on stereo headphones.
v5 dropped the bass bus by 2 dB (volume_db -1 → -3). Documenting this so
the next session-opener doesn't repeat the "but the LUFS numbers are
equal" debate.

### Sources

- iZotope, ["What Are LUFS?"](https://www.izotope.com/en/learn/what-are-lufs) — K-weighting filter shape
- Mastering.to, ["What is LUFS"](https://mastering.to/blog/what-is-lufs) — explicit "12 dB/octave HP at 100 Hz" filter description
- Yurii Arefyev, ["LUFS Is Not Loudness: What Artists Still Don't Understand"](https://medium.com/@arefyevstudio/lufs-is-not-loudness-what-artists-still-dont-understand-d539b1bffdf6) (March 2026) — perceptual-vs-measured loudness gap
- Harman target curve — published research on consumer headphone bass bias

---

## Vocal Mixing

Preserve the approved lead take, doubles, and intended expression. Review phrase
levels, breaths, consonants, sibilance, pitch/timing intent, and effect tails in
quiet and dense sections. Do not substitute whole-file loudness for this review.

### Chain choices

Use gain rides, EQ, compression, and de-essing only where they serve a stated
purpose. Compression can change the prominence of sibilance, but does not
universally amplify it. De-essing before compression can prevent sibilants from
driving the detector; de-essing afterward can control what the chain emphasizes.
Choose placement by comparing the actual problem at matched loudness.

EQ placement also affects compressor detection. Neither subtractive-before nor
additive-after is mandatory. Check whether a proposed EQ change is fixing the
vocal or compensating for another instrument masking it. Preserve the approved
balance while evaluating tone and consistency separately.

### Genre-specific aesthetics

| Genre | Comp ratio | EQ shape | Reverb | Saturation |
|---|---|---|---|---|
| **Pop** | 6:1 (heavy) | bright + heavy presence | plate (short) + slap delay | optional, light |
| **Rock** | 4:1 (moderate) | present, controlled top | room + small plate | tape, subtle |
| **Ballad** | 3:1 (gentle) | natural, intimate | lush hall, longer | none — preserve intimacy |
| **Hip-hop** | 6:1 + serial 4:1 (very heavy) | pitched-up + bright | chamber + delay throws | tube, audible |
| **Jazz** | 3:1 (gentle) | natural, full body | small chamber or room | none |

### Reading the vocal analyse fields

`analyze.py` writes a `vocal` block into `analysis.json` for any stem
(though the metrics only carry meaning for monophonic tonal material —
i.e. vocals, solo instruments). The agent uses these to decide chain
parameters automatically.

| Field | Meaning | Action |
|---|---|---|
| `sibilance.peak_db` | 5-8 kHz transient peak in dBFS | > -25 dBFS: listen for harsh esses; `apply_deesser` after the comp step is a candidate. -25 to -35: borderline, `deesser_smooth` if anything. < -35: relevance_check will skip (nothing to de-ess). |
| `sibilance.density_per_sec` | sibilant events per second | > 4/sec is dense — use `deesser_aggressive`. Sparse takes are fine with `deesser_smooth`. |
| `plosive.events_per_minute` | sub-100 Hz bursts (20 ms envelope) within 12 dB of the stem's loud level, 150 ms refractory; `events_count` is the total | > 10 per minute: audition a tighter HP (100-120 Hz) or clip-gain on the worst bursts; BG vocals tolerate a higher HP. |
| `pitch.mean_hz` | average fundamental | < 200 Hz typically male (use `deesser_male_lead` detection band 4-7 kHz); > 250 Hz typically female (use `deesser_female_lead` 6-9 kHz). |
| `pitch.cents_std` | RMS cents from per-note semitone targets after removing the global tuning offset (`tuning_offset_cents`); notes split at > 80-cent jumps and smoothed over 200 ms; random notes give about 29 | < 15: well intoned. 15-25: noticeable — listen to the phrases with high `fraction_over_25_cents`. > 25: surface to the user; the take may have intentional bends. Correction is the artist's choice. |
| `vibrato.rate_hz` / `vibrato.extent_cents` | 4-7 Hz pitch modulation per sustained note (>= 0.5 s); extent is the semi-extent (± cents) | Extent under ~10 cents: no meaningful vibrato. Rate and extent describe the performance; whether they suit the song is a listening judgment. |
| `breath.silence_ratio` | fraction of frames below -45 dBFS | > 0.4: a lot of breaths/silence between phrases — consider gating between phrases, or accept it as part of the intimate character. |

### Reverb-bus architecture (shared sends vs. insert)

Shared send reverb is a useful default for coordinating vocal space. Insert reverb can also be intentional. To use shared space, declare one
or two reverb buses at the top of `mix_config.json` and let each vocal
track send to those buses at appropriate levels:

```json
"reverb_buses": {
  "vocal_plate": {"preset": "vocal_plate", "wet": 1.0, "return_volume_db": -8},
  "vocal_hall":  {"preset": "vocal_hall_wide", "wet": 1.0, "return_volume_db": -12}
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

Why this matters:
- **Multiple sources, one reverb space.** Lead + BG + double-tracks all
  hit the same plate, which is what makes them sound like they're in the
  same room. With insert reverb each track gets its own room — they
  never glue together.
- **Wet/dry balance separate from level.** The reverb bus' wet level
  stays at 1.0 (pure wet — the dry is the original track in the master
  sum). Track-level sends control how much reverb each vocal gets, the
  bus' `return_volume_db` controls how loud the reverb tail is overall.
- **One reverb instance per bus, regardless of source count.** A plate
  bus with 5 vocals sending into it costs the same CPU as one vocal —
  five inserts would be five separate reverb computations.

### Pitch correction philosophy

Treat pitch correction as an aesthetic decision. Follow existing user intent
and authorization; when the intended correction is unclear, clarify it before
changing the performance. Keep the original and compare specific phrases.
A cents-deviation statistic can reflect bends, vibrato, tuning, or tracking
errors; it cannot prove an out-of-tune performance by itself.

Strength values are tool settings, not universal subtle/pop standards. The pitch
tracker and scale quantization can make wrong decisions. Review note transitions,
consonants, vibrato, and artifacts, and keep unreviewed correction provisional.

### Why the de-esser is a `relevance_check`-guarded tool

`apply_deesser.py` skips processing when the sibilance band peak is below
-25 dBFS. Reason: running a de-esser on a stem that has no sibilance
content just adds artefacts (the band-pass filter ringing through the
detector during quiet moments). The skip is the right answer for non-vocal
sources accidentally routed through the de-esser, or for vocals where the
sibilance is already controlled at the recording stage.

### Sources

Researched against the 2026 vocal-mixing consensus:

- [Music Guy Mixing — Vocal Chain Order](https://www.musicguymixing.com/vocal-chain/)
- [iZotope — Crafting a basic vocal chain](https://www.izotope.com/en/learn/crafting-a-basic-vocal-chain)
- [Sonarworks — Chaining vocal effects plugins](https://www.sonarworks.com/blog/learn/how-to-chain-multiple-vocal-effects-plugins-effectively)
- [Universal Audio — Top vocal chains](https://www.uaudio.com/blogs/ua/top-uad-vocal-chains)
- [Patrik Skoog — How to Mix Vocals](https://www.patrikskoogmusic.com/guides/how-to-mix-vocals-eq-compression-saturation)
- [PSOLA algorithm explanation (TCNJ Autotuner project)](https://engprojects.tcnj.edu/autotuner16/2016/04/11/the-psola-algorithm/)
- [JanWilczek/python-auto-tune — PYIN + PSOLA reference implementation](https://github.com/JanWilczek/python-auto-tune)

---

## Style Profiles - Project Preference Similarity

`tools/style_check.py mix.wav --style NAME` measures similarity of a finished master to one of seven project profiles in `tools/style_profiles/`: `classic_rock`, `hip_hop`, `jazz_acoustic`, `modern_rock`, `pop`, `punchy_modern_rock`, `tool_inspired`. On premaster input (`--input-kind premaster`, or auto-detected when worst-channel TP <= -2.5 dBTP) loudness, LRA and crest checks are N/A. The profile fixes loudness, dynamics, and 5-band tonal-balance targets — when no reference audio is supplied, the profile remains a numerical preference and cannot replace listening references.

### What's in a profile

Every profile JSON has the same shape:

| Section | Fields | Meaning |
|---|---|---|
| `lufs` | `integrated_target`, `tolerance_lu` | Project loudness preference. Symmetric tolerance — outside `target ± tolerance` is yellow, beyond 1.5× is red. |
| `lra` | `target_lu`, `range_lu` | Loudness Range. Range-based grading: inside [min, max] = green. |
| `crest_factor` | `target_db`, `range_db` | Sample peak vs RMS, range-graded. |
| `tonal_balance_dbfs` | `{sub_60hz, low_60_250hz, mid_250_2khz, high_2_8khz, air_8khz_plus}` each with `target` + `tolerance` | **Wideband band-RMS measured at the profile's LUFS target**, not iZotope-TBC PSD-curve numbers. Project targets; no independently validated reference corpus is supplied. |
| `default_bus_volume_db` | `{drums, bass, guitar}` each with a dB number | Starting `volume_db` for each top-level bus when running `render_mix --generate-config --style NAME`. The neutral-0-dB default rarely matches modern conventions — these per-genre starting points encode the legacy hand-tuned bus starting point (drums+bass as foundation in rock, vocal-forward in pop, sub-dominated in hip-hop, etc.). |

### Genre-typical bus starting points (current profile values)

| Profile | drums | bass | guitar | Logic |
|---|---|---|---|---|
| **modern_rock** | 0 | 0 | -3 | drums + bass as foundation, guitar 3 dB below |
| **classic_rock** | 0 | -1 | -1 | midrange-prominent, less sub bass than modern |
| **pop** | -1 | -1 | -3 | vocal-forward genre (guitars give the vocal space) |
| **hip_hop** | -1 | +1 | -4 | 808 bass drives the genre; guitar (if any) heavily backed off |
| **jazz_acoustic** | -3 | -1 | 0 | gentle drums for dynamics; guitar/piano on top |

The agent and user iterate from these starting points; they're a reference,
not a final answer. After `--style` generation, listen to the first render
and adjust on a per-session basis (e.g. a particularly bass-heavy bass DI
might need an extra -2 dB, as the v5 terido iteration documented).

### How the grading works

1. Measure integrated LUFS, LRA, crest factor on the input mix.
2. Apply a single linear gain so the mix sits at the profile's `integrated_target` LUFS.
3. Measure 5-band wideband RMS on the LUFS-normalised mix.
4. Grade each check: GREEN if within tolerance / range, YELLOW just outside, RED significantly outside.
5. Overall score: GREEN-check = 100, YELLOW = 50, RED = 0. Average → 0..100. Verdict thresholds: ≥85 green, ≥60 yellow, below 60 red.
6. There is no loudness/LRA hard-fail override. The score describes profile similarity; the CLI returns success when measurement completes, including a red similarity result.

### When to use which profile

| Profile | Match for | LUFS target | LRA target | Sub presence |
|---|---|---|---|---|
| `modern_rock` | Alt-rock, indie rock, post-rock streaming masters | -10 | 4–9 LU | Moderate (-26 dB) |
| `classic_rock` | Vintage / 60s–80s analogue aesthetic, dynamic | -13 | 8–14 LU | Less sub (-28 dB), more mid |
| `pop` | Top-40 streaming pop, vocal-forward, bright | -9 | 3.5–7 LU | Tight low end, more air |
| `hip_hop` | Trap / modern hip-hop with 808s | -8 | 2.5–5.5 LU | Massive sub (-22 dB), scooped mids |
| `jazz_acoustic` | Jazz, folk, singer-songwriter | -18 | 10–18 LU | Gentle low (-30 dB) |

### Calibration note (important)

The tonal-balance targets are **wideband band-RMS on a LUFS-normalised mix**, NOT the iZotope Tonal Balance Control PSD-curve values. The two measurement systems give different numbers for the same audio — TBC integrates over 1/3-octave PSD with proprietary smoothing; here we filter the mono mix into 5 wide bands and take a plain RMS. The two are not interchangeable.

Implication: do not paste these numbers into TBC and expect them to line up with TBC's curve display. They line up with `analyze.py` / `mix_health.py` measurements (same band edges and same RMS metric).

### Tuning a profile to a specific user's taste

The profiles contain historical project preferences; the repo does not supply a reproducible, independently validated calibration corpus. To explore a user-specific preference:

1. Measure 3–5 reference mixes the user considers "right" using `analyze.py` (LUFS-normalise each to the profile's target first).
2. Average the per-band RMS values.
3. Replace the profile's `tonal_balance_dbfs.<band>.target` with the averages. Increase the `tolerance` if the spread across the references is wider than the default ±2.5 dB.
4. Bump the `version` (e.g. 1.1 → 1.2) and add a one-line note under `calibration_note`.

The profiles are versioned and well-commented intentionally so that user-specific overrides remain readable.

---

## Progressive metal - section-based working method

Classification: project starting hypotheses, not an artist preset or universal
recipe. The user's arrangement and references determine the desired result.

1. Preserve takes and edits. Map quiet passages, dense riffs, exposed vocals,
   fills, transitions, solos, and the ending. Identify intended contrast.
2. Build the drum kit from useful microphone relationships. Compare overhead
   and room alternatives before gating or replacing their contribution. Test
   shell attack, body, sustain, and cymbal decay in the full arrangement.
3. Establish kick and bass roles by section. Compare clean and driven bass
   components for articulation, weight, and mono behavior. Neither instrument
   automatically owns an entire frequency range. Do not synthesize sub-bass
   solely because a spectrum looks sparse.
4. Evaluate guitar microphone blends, note definition, and stereo placement
   against the rhythm section. Avoid automatic mid scoops and top-end boosts.
5. Preserve the selected vocal take and double restraint. Review phrase levels,
   consonants, breaths, sibilance, effect tails, and intelligibility in quiet and
   dense sections. Do not infer pitch/timing edits from a genre label.
6. Use automation to support transitions and emotional contrast. Test one
   coherent change at a time with matched excerpts; keep the earlier version.
7. Compare a minimally processed master baseline with any more processed
   candidate. Inspect actual dynamics changes before seeking more loudness.
8. Deliver the requested formats after checking the exported files and recording
   scoped full-song approval. Do not generate every streaming preset by default.

Creative options to audition include changing room contribution across a build,
restrained delay throws in vocal gaps, or a bass/guitar texture change around a
transition. These are optional ideas; measurements alone cannot select them.

Barresi's [Tool interview](https://www.waves.com/in-the-studio-with-tool-and-evil-joe-barresi)
describes section-dependent instrument sounds and vocal treatments. That supports
contextual experimentation, not copying numerical EQ or stereo settings.

### Mastering review diagnostics

`master_mix.py` reports whole-file crest change, gain before limiting,
post-limiter peak attenuation, and true-peak/sample-peak difference. Current
review triggers are a crest reduction over 4 dB, normalization gain over 6 dB,
post-limiter attenuation over 3 dB, or peak difference over 1 dB. These are
conservative project heuristics, not evidence of audible damage or standards.
A green peak result does not clear these listening questions.

The brickwall limiter applies no makeup gain, so post-limiter attenuation is the
limiter's gain reduction plus any 8x safety trim; it is not all intersample
overshoot. Compressor input/output
LUFS difference is a loudness change, not measured instantaneous gain reduction.
Whole-file crest is sensitive to arrangement; use matched section comparisons
before choosing a corrective action.

## Sources

### Finishing techniques and their limits

These techniques are candidates for a section-based review. Numeric settings
belong to individual session records, not universal genre requirements.

| Technique | Intended purpose | Required listening question |
|---|---|---|
| Bounded phrase rides | Improve lead consistency while retaining expression | Do important words stay clear without leveling away the performance? |
| Parallel shell compression | Add drum body beneath original attacks | Does the kit gain useful weight without cymbal spill, pumping, or softened attacks? |
| Sample reinforcement | Support identified weak/inconsistent hits | Does the original kit identity, velocity, timing, and phase relationship survive? |
| External-sidechain dynamic EQ | Reduce a specific overlap only during competing activity | Is separation improved without audible holes or pumping? |
| Wet-return ducking | Keep ambience out of active vocal phrases | Do tails emerge naturally without making the voice unnaturally dry? |
| Selected delay throws or swells | Reinforce a musical event | Does the detail support the phrase and transition, rather than distract? |
| Section automation | Develop attention, space, and intensity through the song | Do transitions and contrasts feel intentional at matched loudness? |

`apply_dynamic_eq.py` uses a linked RMS detector to interpolate between dry audio
and a fixed minimum-phase bell cut. Its depth is bounded, and a silent detector
is dry bypass. This is a specific implementation, not an emulation of a named
commercial equalizer. Sidechain length and sample rate must match. Check actual
activity and center-cut statistics; they describe processing, not its benefit.

Room-depth guidance, reviewed 2026-09-10:

For more assertive drum production, Mark Mynett documents parallel snare
distortion, selective sample reinforcement, and separate short plate/room sends
in [Making Modern Metal, Part 3](https://www.soundonsound.com/techniques/making-modern-metal-part-3).
His particular settings and opinions are not universal requirements. Compare
instrument character before adding more room; retain original performances when
no concrete need for replacement or timing correction has been established.

Stronger nonlinear processing can generate aliases. The saturation tool supports
2x, 4x, and 8x oversampling, with overlapping blocks and a final linked RMS match.
Its default remains 1x for compatibility; input_gain_db provides additional drive
without intentional wet-path level gain. The modes are simple waveshapers, not
physical tape or tube models. Symmetric tanh has odd symmetry; asymmetric shaping
can generate even harmonics and DC. See the official
[FabFilter oversampling documentation](https://www.fabfilter.com/help/saturn/using/inputoutput)
for the general rationale, not equivalence to that product.

- Natural room microphones contain acoustic travel time and reflections. Preserve
  their timing unless a listening comparison supports a change. Check the full
  kit rather than optimizing a waveform correlation score. Barresi describes
  combining recorded rooms and additional reverbs, with listening-led microphone
  decisions in the [Tool interview](https://www.waves.com/in-the-studio-with-tool-and-evil-joe-barresi).
- Early reflections, predelay, decay, and wet balance affect different aspects of
  depth. A short shared room is a candidate for cohesion; more width or a longer
  tail alone does not establish realism. See the official
  [FabFilter reverb controls](https://www.fabfilter.com/help/pro-r/using/maincontrols).
- Shape instrument sends and room returns where needed. Filtered returns and
  predelay can help manage drum ambience, but numeric settings require a session
  comparison. See [Waves drum-reverb techniques](https://www.waves.com/tips-for-using-reverb-on-drums-in-your-mix).
- The repository's Freeverb presets are tonal starting points. Names such as
  plate and hall are not distinct physical models, and their approximate decay
  notes are not measurements. For a timed-room experiment, report a measured
  impulse-response decay estimate, including the fit interval and filters.

Delivery and premaster conventions, verified 2026-10-06 (repository review):

- Spotify ([loudness normalization](https://support.spotify.com/us/artists/article/loudness-normalization/)):
  playback at -14 LUFS (Normal), -11 (Loud), -19 (Quiet); keep true peak below
  -1 dBTP, or -2 dBTP when the master is louder than -14 LUFS. Official
  platform guidance; high confidence.
- SoundCloud ([help center](https://help.soundcloud.com/hc/en-us/articles/360053660014))
  now normalizes to -14 LUFS with the same -1 / -2 dBTP advice. Older
  "no normalization" claims are outdated.
- AES TD1008 (2021): -16 LUFS music distribution loudness with album
  normalization; deliberately not a mastering target. ITU-R BS.1770-5 (2023)
  is current; ATSC A/85 is -24 LKFS / -2 dBTP.
- Premaster handoff: Abbey Road (2021) asks for 2-3 dB of peak headroom with
  limiting removed; the Metropolis preparation guide (2026-05) asks for 32-bit
  float at the native sample rate and no clipping, and de-emphasizes headroom.
  "Premaster -18..-20 LUFS / LRA >= 6 LU / crest 14-18 dB" targets appear only in
  blogs and are folklore, not standards; mix_health treats LRA as advisory.
- pyloudnorm 0.2.0 (2026-01) provides `loudness_range()`; it has no true-peak
  meter, so true peak is measured in `tools/_dsp.py` by polyphase oversampling
  (4x/8x estimates, not certified against the BS.1770 conformance set).
- pedalboard 0.9.25 adds `BrickwallLimiter` (lookahead, true-peak option, no
  makeup gain). Its documentation does not guarantee the reconstructed ceiling,
  hence the 8x verification step.

Source register, verified 2026-09-09:

- [Joe Barresi on Tool](https://www.waves.com/in-the-studio-with-tool-and-evil-joe-barresi),
  published 2019-11-07. First-person engineering account; high confidence for
  his described section-dependent tones, vocal treatments, and microphone work.
  It does not prescribe universal Tool/Wheel settings.
- [Joe Barresi on vocal depth and details](https://www.waves.com/mixing-vocals-depth-excitement-ear-candy).
  Engineer tutorial page; supports selective vocal effects as a creative option.
  The page description was reviewed; no claim of auditioning its demonstration.
- [FabFilter external sidechain documentation](https://www.fabfilter.com/help/pro-q/support/externalsidechaining).
  Official implementation documentation; high confidence for triggering EQ from
  another signal. It establishes capability, not a song-specific need or setting.
- [Ian Shepherd on mastering metal](https://www.izotope.com/community/blog/how-to-master-metal),
  published 2019-09-04. First-person mastering guidance; supports considering
  separation, tonal balance, contrast, and appropriate dynamics processing.
  Historical streaming details require separate current verification.
- [Sound On Sound: Masters Of The Art Of Mixing](https://www.soundonsound.com/techniques/masters-art-mixing).
  Indexed excerpts describe different engineers' acoustic, parallel-processing,
  and sample-reinforcement approaches. Full-page access was blocked; confidence
  is limited to the retrieved excerpts, not a complete review of the article.

- [Gain Staging Explained 2026 — Mixing Monster](https://mixingmonster.com/gain-staging/)
- [Understanding LUFS 2026 — Mixing Monster](https://mixingmonster.com/understanding-lufs/)
- [The Rise of Smart Gain Staging — DLK Music Pro (Feb 2026)](https://news.dlkmusicpro.com/the-rise-of-smart-gain-staging-in-modern-audio-production/)
- [Mastering for Streaming: LUFS Targets 2026 — Genesis Mix Lab](https://genesismixlab.com/guides/mastering-delivery/)
- [Mastering for Streaming Platforms — iZotope](https://www.izotope.com/en/learn/mastering-for-streaming-platforms)
- [Mix Tip: Gain Staging using Clip Gain in Pro Tools — Danny Anthony](https://medium.com/@dannyanthony/mix-tip-gain-staging-using-clip-gain-pro-tools-718140441970)
- [Clip Effects and Clip Gain in Pro Tools — Audeobox](https://www.audeobox.com/learn/pro-tools/clip-effects-and-clip-gain/)
- [Audio Normalization: Should You Normalize Your Tracks? — LANDR](https://blog.landr.com/audio-normalization/)
- [How To Clean Up Your Guitar Sound After Recording — iZotope](https://www.izotope.com/en/learn/how-to-clean-up-your-guitar-sound-after-recording.html)
- [Solving Guitar Noise, Buzz and Hum — Sweetwater](https://www.sweetwater.com/sweetcare/articles/solving-guitar-noise-buzz-and-hum/)
- [Ground Loops Explained — Sound on Sound](https://www.soundonsound.com/techniques/ground-loops-explained)
- [Balancing Distorted and Clean Guitars In A Mix — Joey Sturgis Tones](https://joeysturgistones.com/blogs/learn/balancing-distorted-and-clean-guitars-in-a-mix)
- [Bass Guitar EQ Guide — Music Guy Mixing](https://www.musicguymixing.com/bass-guitar-eq/)
- [Electric Guitar EQ Guide — Music Guy Mixing](https://www.musicguymixing.com/electric-guitar-eq/)
- [Electric Guitar EQ Guide — Neural DSP](https://neuraldsp.com/articles/electric-guitar-eq-guide)
- [Kick Drum EQ 101 — Gear4music](https://www.gear4music.com/blog/kick-drum-eq/)
- [Complete Snare EQ Guide — Music Guy Mixing](https://www.musicguymixing.com/snare-eq/)
- [How To EQ Drum Overheads — SoundShockAudio](https://soundshockaudio.com/how-to-eq-overheads/)
- [Hi-Hat EQ Settings — Music Guy Mixing](https://www.musicguymixing.com/hi-hat-eq/)
- [How to EQ Tom Drums — Music Guy Mixing](https://www.musicguymixing.com/eq-tom/)
- [How To EQ Room Mics — SoundShockAudio](https://soundshockaudio.com/how-to-eq-room-mics/)
- [FabFilter Pro-Q 4 EQ Match feature](https://www.fabfilter.com/help/pro-q/using/eqmatch)
- [Advanced EQ Techniques — Mixing Monster](https://mixingmonster.com/advanced-eq-techniques/)
