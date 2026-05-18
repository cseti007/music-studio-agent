# Domain Knowledge — Audio Mixing & Gain Staging

Reference material for tool decisions and recommendations. Update when new research is done.

---

## Stem Assembly and Clip Gain

**Rule: clip gain and assembly are done together as the first step, before any further processing.**

`apply_gain --per-clip` reads the session.json clip layout, normalizes each clip to a consistent
LUFS level (clip gain), and assembles the full-length stem in a single pass.

The correct order is:
```
1. parse_session                      ->  session.json (clip layout)
2. apply_gain --per-clip session.json ->  assembled.wav (clip gain + assembly)
3. analyze                            ->  read the assembled stem
4. apply_gain --per-channel (optional)->  delivery normalization only
5. ...further processing
```

### Clip gain vs per-channel gain staging

**Clip gain** (`--per-clip`) normalizes each recorded clip to a consistent LUFS level before
assembly. This is the standard DAW practice (Pro Tools, Logic, etc. all have a dedicated
"Clip Gain" control). It operates before any plugins and fixes level differences caused by
different recording gain settings across sessions.

**Per-channel gain staging** (`--per-channel`) applies a single gain to the whole assembled stem
to reach a delivery target (Spotify, Apple Music, etc.). This is optional after a correct
per-clip pass — the stem is already at mix-ready levels.

### When NOT to use per-clip normalization

Per-clip normalizes each clip independently to the same LUFS target. This removes intentional
dynamic contrast between clips — a soft verse and a loud chorus recorded in the same session
at the same gain setting would be brought to identical levels.

Use `--per-clip` when: clips were recorded in different sessions with different interface/preamp
gain settings (accidental level inconsistencies).

Do NOT use `--per-clip` when: the level differences between clips are intentional performance
dynamics (the bassist plays softer in the verse and louder in the chorus by design).

**Rule: drums are NEVER per-clip normalized.** A drummer records the whole song in one
continuous take. Any editorial clip cuts in the session are the editor's work (repairs, region
splits), not different gain-staged recordings. Per-clip normalization on drums treats these cuts
as independent recordings and creates audible level jumps at edit boundaries — the result sounds
like a stereo-to-mono collapse at the cut point because the overhead mics (which provide stereo
width) get normalized to different levels on each side of the cut.

For drums: run `apply_gain --per-clip` only to assemble the stem (it handles region placement
and silence), then immediately run `apply_gain --per-channel` on the assembled result to apply
a single uniform gain. This preserves the natural dynamics of the performance across the whole
take and avoids artificial level steps at edit boundaries.

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

### LUFS targets by use case

| Context | Integrated LUFS | True Peak | Notes |
|---|---|---|---|
| Stem in mix | -18 LUFS | -3 dBTP | headroom for bus processing |
| Pre-master mix bus | -18 to -14 LUFS | -3 to -6 dBFS peak | leave room for mastering chain |
| Spotify delivery | -14 LUFS | -2 dBTP | platform default (Normal mode) |
| Apple Music delivery | -16 LUFS | -2 dBTP | Sound Check normalization |
| YouTube delivery | -13 / -14 LUFS | -2 dBTP | |
| Amazon Music delivery | -14 LUFS | -2 dBTP | Amazon requires -2 dBTP explicitly |
| Broadcast (EBU R128) | -23 LUFS | -2 dBTP | TV/radio standard, too quiet for music |
| Dolby Atmos (Apple Music) | -18 LUFS | -1 dBTP | ADM BWF 48kHz/24-bit + stereo fallback required |

**Key rule:** -23 LUFS is broadcast standard, NOT a music mixing target. Use -18 for stems.

### apply_gain.py modes and presets

**Two modes:**

`--per-clip session.json` — clip gain + assembly. Normalizes each clip to
`per_clip_target_lufs` (default -18 LUFS, set in config.toml). This is the PRIMARY
gain-staging step. Research confirms this is standard DAW practice: clip gain runs
before any plugins so compressors/EQs receive consistent input levels.

`--per-channel file.wav` — single gain applied to assembled stem. Use for delivery
normalization or when receiving pre-assembled stems. Presets:

| Preset | Target LUFS | True peak limit | Use when |
|---|---|---|---|
| `stem` | -18 LUFS | -1 dBTP | individual stem going into a mix |
| `premix` | -18 LUFS | -3 dBTP | alias for stem, more conservative peak |
| `spotify` | -14 LUFS | -2 dBTP | final delivery to Spotify |
| `apple` | -16 LUFS | -2 dBTP | final delivery to Apple Music |
| `amazon` | -14 LUFS | -2 dBTP | final delivery to Amazon Music |
| `broadcast` | -23 LUFS | -2 dBTP | TV/radio/podcast delivery |

**Key insight from research (2026):** per-clip normalization + per-channel gain staging
is redundant if the per-clip target matches the stem target. After a correct `--per-clip`
pass at -18 LUFS, the assembled stem is already at mix-ready levels — `--per-channel`
is only needed for final delivery to a specific platform.

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

**Phase mode:** `sosfiltfilt` (zero-phase, forward+backward) — no phase shift. Correct for offline mixing. Linear phase avoids pre-ringing artifacts on transients vs zero-latency minimum phase modes — for offline processing they are equivalent.

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
- EQ before compression — never compress first (creates dark, muddy character)

---

## True Peak vs Peak

- **Sample peak (dBFS)**: highest sample value — what DAW meters typically show
- **True peak (dBTP)**: inter-sample peak — can exceed sample peak after D/A conversion
- Always limit true peak to -2 dBTP for streaming delivery — Amazon Music requires this explicitly; lossy
  codec inter-sample peaks (AAC, Ogg Vorbis) can exceed the sample ceiling, so -2 dBTP gives necessary margin
- During mixing: -3 dBTP gives safe headroom

---

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

Both are RMS-normalized in this pipeline: they change the spectral content (add harmonics) without changing the perceived level.

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

The `render_mix.py` master chain (glue comp + guarded clipper + guarded
M/S + EQ + LUFS norm + ISP-aware limiter) is **the mix engineer's
polish**, not the master pass. It runs inside the render to give the mix
a coherent shape. The actual mastering pass is `master_mix.py`, run
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

### Format-specific delivery targets (2026)

| Platform | LUFS | True peak | Bit depth | Notes |
|---|---|---|---|---|
| Spotify | -14 | -1 dBTP | 24 | Ogg Vorbis encoding — codec ISP overshoots ~0.5-1.0 dB |
| Apple Music | -16 | -1 dBTP | 24 | AAC encoding — slightly quieter target than Spotify |
| YouTube | -14 | -1 dBTP | 24 | Same as Spotify; YouTube normalises loudly |
| Tidal | -14 | -1 dBTP | 24 | HiFi tier; lossless playback |
| CD | -9 | -1 dBTP | 16 | Louder; 16-bit dithered for Red Book |
| Vinyl pre-master | -12 | -1 dBTP | 24 | Gentle — the cutter adds its own limiting |
| Broadcast (EBU R128) | -23 | -2 dBTP | 24 | TV / radio |

The targets converge enough that one "streaming master" (-14 LUFS, -1 dBTP,
24-bit) is acceptable for Spotify, YouTube, Tidal. Apple gets a separate
target; CD and vinyl are their own paths.

### Mastering chain presets

`master_mix.py` ships six chain templates (the *what to do* part, distinct
from the format target *how loud* part):

| Preset | Chain | Use when |
|---|---|---|
| `gentle` | comp only (1.5:1, gentle) | Acoustic, jazz, classical-leaning rock. Preserves dynamics. |
| `modern_rock` | EQ + glue comp + exciter + M/S side highshelf + soft clip | Competitive rock loudness with audible LUFS lift and a wider top. |
| `modern_rock_mb` | EQ + 3-band multiband + exciter + M/S side highshelf + stereo width 1.05 + soft clip | Modern rock with tighter band-by-band dynamics. Replaces glue comp with multiband — better controlled low end. |
| `pop` | EQ (bright) + comp + exciter + M/S side highshelf + width 1.1 + soft clip | Bright top, present mids, slightly wider image. |
| `hip_hop` | EQ (sub boost) + comp + exciter + width 0.95 (slightly narrower) + hard clip | Sub weight, impact, mono-leaning width to keep the 808 centred. |
| `transparent` | LUFS norm + limit only | When the mix doesn't need master tone. |

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
- **Vinyl elliptical EQ**: sub-mono filter below ~150 Hz. Automatic on
  the `vinyl_pre` format (cuts side energy below 150 Hz so the vinyl
  cutter head doesn't leave the groove on wide bass). Not configurable
  per chain preset — driven by the format's `vinyl_elliptical_hz` field.

### Codec-ISP vs. true peak

Sample peak and 4× true peak miss what the codec encoder does to the signal.
Ogg Vorbis and AAC re-quantize transients and routinely overshoot the
4×-oversampled true peak by 0.5-1.5 dB. `master_health.py` includes an
**8×-oversampled codec-ISP estimate** as a conservative proxy: if your master
sits at -1.0 dBTP measured at 4× but the 8× value is -0.3 dBTP, expect the
encoded version to occasionally clip.

### Punch index

Mastering must preserve transient punch. The punch index in
`master_health.py` is `percentile_90(short-window RMS) / mean(long-window
RMS)` in dB — how far the transient peaks pop above the sustained bed:

| Punch index | Material character |
|---|---|
| < 2 dB | Squashed — transients buried in the bed. Over-limited. |
| 2-4 dB | Low punch — borderline, may sound fatiguing. |
| 4-7 dB | Healthy — modern rock master with intact transients. |
| > 8 dB | Very dynamic — dynamic jazz, live recording, classical. |

If a master's punch index drops below 4 dB after the chain, soften the
clipper / limiter or move to a gentler chain preset.

### Per-band phase coherence

Two stereo rules that matter at the master level:

1. **Sub should be near-mono.** L/R correlation below ~100 Hz should be
   > 0.85 (ideally > 0.95). Otherwise the bass collapses on mono playback
   (phone speakers, club PA mono fold-down).
2. **Top can be wide, but not anti-correlated.** L/R correlation in the
   8 kHz+ band should be > 0.2. Below that, the highs are decorrelated
   to the point of phasiness.

`master_health.py` checks both per band; failing the sub check is RED.

### Compression history detection

If the input mix.wav already shows signs of mastering (LRA < 4 LU, crest
< 10 dB, or sample peak > -0.5 dBFS), `master_health.py` flags
`likely_already_mastered: true` and warns. Applying more mastering on top
of an already-mastered track flattens it further with no benefit.

When this triggers: pull back the master_preset to `gentle` or
`transparent`, or work from the pre-master / pre-limit version of the mix.

### Reference deck

A single reference can be misleading — if the reference happens to be
extra-bright or has a specific vocal mix, the comparison drifts that way.
Mastering engineers use **multiple references** (a "deck") and look at the
average target spectrum. `master_health.py --reference ref1.wav ref2.wav
ref3.wav` averages all references' 1/3-octave PSDs and reports region-level
deltas against your master. 3-5 references is the typical deck size.

### Reference deck is a tonal GUIDE, not a hard delivery gate

When `master_health` returns a red on the reference-deck section, do **not**
treat it the same as a red on format conformance. The two are different
classes of check:

| Check | Class | What red means | What to do |
|---|---|---|---|
| Format conformance (LUFS, true peak, codec ISP) | **Hard gate** | The delivery target is missed; the master will trigger platform normalisation / clip | Re-master, do not ship |
| Phase coherence per band (sub mono, top wide) | **Hard gate** | The master collapses on mono speakers / has out-of-phase highs | Fix the M/S processing, re-master |
| Punch index | **Hard gate** | The master is over-limited and will sound fatiguing | Soften the chain, re-master |
| Compression history (already mastered?) | Yellow advisory | The input was already a master — extra master may flatten | Inform; if intentional double pass, proceed |
| **Reference deck spectral delta** | **Tonal guide** | The master is spectrally different from the chosen reference(s) — could be the master is off, OR the reference is just different (different vocal mix, different genre tilt, different era) | Use as direction-finding; ship if hard gates are green |

If the four hard gates (LUFS, true peak, phase, punch) are green, the master
is technically deliverable even with a red reference-deck verdict. Treat
the reference deck like a second opinion: if the deltas surprise you,
investigate; if they're explained by known content differences (e.g. you
mixed instrumental, the reference is vocal-driven; or you mastered for
modern rock loudness, the reference is a 90s mix), document it and ship.

---

## Master Bus Chain Order

Processing order matters. This is the current `render_mix.py` master chain
(each step optional, guarded ones skip if their relevance_check fails):

1. **Bus saturation** (per bus, before summing to master)
2. **Bus parallel saturation** (guarded, drum bus only — relevance_check: crest > 10 dB AND LRA > 4 LU)
3. **Master sum** (all top-level buses)
4. **Master glue compressor** (2:1, slow-ish attack, catches sustained program material)
5. **Master clipper** (guarded, soft cubic or hard — relevance_check: sample peak ≥ -10 dBFS AND LRA ≥ 4 LU)
6. **M/S processing** (guarded — independent mid/side EQ + gain — relevance_check: width ≥ 0.05)
7. **Master EQ** (zero-phase — HP@30Hz + gentle high shelf typical)
8. **LUFS normalization** (target -14 LUFS for streaming)
9. **ISP-aware true peak limiter** (-2 dBTP — pedalboard.Limiter + second-pass 4x-oversampled ISP scale-down)

For the **master_mix.py** pass on a finished stereo mix, the chain is
slightly different (more aggressive, format-aware) — see "Mastering
Workflow and Philosophy" above.

### Master glue compressor settings (pedalboard Compressor notes)

Pedalboard's Compressor uses peak detection. With slow attack (>20ms), short transients pass through and the measured peak GR appears 0. For program material compression, use:
- threshold: -10 dBFS (works with typical -12 to -9 LUFS pre-norm signals)
- attack: 10ms (catches sustained peaks while letting some transient through)
- ratio: 2:1
- release: 300ms
- Expected GR: -0.5 to -1.0 LUFS on a dense rock mix

---

## Make-it-hit Philosophy — Loudness, Weight, Width Without Fatigue

A modern rock/pop mix has to **"hit"** — sound dense, weighty, and impactful — but **without sounding squashed or tiring**. These are in tension. Too little processing = thin and dynamic but inconsequential. Too much = "loud but ugly", listener fatigue, sales drop on smartphone speakers.

The tools that exist to push toward "hit" — clipper, multiband compressor, sub-bass synth, exciter, parallel saturation, M/S width — are powerful and they all share a property: **they degrade the source signal in exchange for a perceived improvement.** Saturation adds harmonic distortion. Clipping flattens transients. Sub-synth adds harmonic content that wasn't there. M/S widening reduces mono compatibility.

The agent's job is to apply them **only when the data justifies the trade-off**. This is why every make-it-hit tool in this project ships with a `relevance_check` — the tool measures its input and refuses to write audio when the conditions for benefit are not met.

### The relevance_check pattern

Every make-it-hit tool runs its check first and returns a verdict like:

```json
"relevance_check": {
  "tool": "subharm",
  "sub_band_rms_dbfs": -25.0,
  "target_over_fundamental_db": 3.8,
  "recommend_skip": true,
  "issues": [
    "target band (80-200 Hz) is 3.8 dB louder than fundamental — new harmonics will be drowned"
  ]
}
```

When `recommend_skip: true`, the tool exits without writing the output WAV. The agent reads the report, understands why, and moves on. Only `--force` overrides — and that should require the user explicitly asking for it.

### Required-evidence thresholds (per tool)

These are the empirical thresholds derived from the field test on the terido session:

| Tool | Required evidence |
|---|---|
| Master clipper | Sample peak > -10 dBFS AND LRA > 4 LU. Below either, the clipper has no headroom to recover or just adds fatigue to an already-flat mix. |
| Sub-bass synth | sub_60hz_rms_db ≥ -35 (something to extract harmonics from) AND sub_60hz_crest_db ≥ 8 (sub isn't squashed) AND target band 80-200 Hz must NOT exceed the fundamental by more than 3 dB (else new harmonics are drowned by existing content). |
| Drum bus parallel sat | Bus crest > 10 dB (transient life left) AND LRA > 4 LU AND bus is drums. |
| Exciter | spectral_centroid_hz < 4000 (stem is dark) AND air_8khz_plus_rms_db < -40 (genuine air-band emptiness). |
| Multiband comp | At least 2 of 3 bands with crest ≥ 6 dB (real per-band dynamics to control). |
| M/S width | ms_width_ratio < 0.2 if side-boosting; do NOT side-boost if width > 0.5 (mono-compat risk). |
| Haas | ms_width_ratio < 0.3 AND NOT on bass / low-centroid stems. |

### Process budget

A single stem chain should not exceed **4 processing steps**. Typical: gain → EQ → comp → one fx. More than that compounds phase shift, transient smearing, and harmonics that didn't ask permission to be there. If you find yourself about to add a 5th step, stop and reconsider whether the earlier steps actually solved the problem — or whether the problem was something other than what the chain has been treating.

### Re-analyze loop

After every make-it-hit step, re-run `analyze.py` on the output. Two checks:

1. **The targeted metric moved the right way.** Sub-synth on a bass stem → sub_60hz_rms_db should rise 0.5-1.5 dB. Clipper → integrated_lufs rises 1-3 dB without LRA collapsing. Multiband → per-band crest tightens in the targeted band. Exciter → spectral_centroid_hz rises (note: with mix ≤ 0.15 this can be subtle, < 50 Hz delta on the centroid is normal).
2. **`pumping.pumping_detected` did not flip true.** If it was false before the step and true after, the step caused it. Revert or soften. (Pre-existing pumping flags often indicate musical pulse — see the pumping disambiguation section.)

If the metric didn't move, **revert**. The tool either didn't help or just shifted the problem.

### Field-test lesson: rock-band tracking and the subharm tool

On the terido test session (a real 56-stem rock recording) every stem failed the subharm relevance check — the target band 80-200 Hz was always 3+ dB louder than the 40-80 Hz fundamental. This is **not a bug**: rock recording captures harmonics naturally, so the target band is always full. Subharm is genuinely a synth-bass / 808 / sample-based-low-end tool. Honour the skip.

### Field-test lesson: pumping is often a musical pulse, not artifact

The pumping detector flags 1-5 Hz envelope modulation. On rhythm guitar tracks the strumming itself produces this modulation at song-tempo quarters or eighths (1.37 Hz at 82 BPM, 2.34 Hz at 140 BPM). The detector cannot tell musical pulse from comp pumping from envelope statistics alone. So `pumping_detected: true` is a **suspicion, not a verdict** — see "Reading the New Analysis Metrics" below.

---

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

1. **Did the flag appear AFTER a comp/multiband/clipper step?** Compare the analysis JSON from before and after. False → True after the step = the step caused it.
2. **Is `pump_rate_hz` close to song-tempo quarters/eighths?** At 120 BPM: quarter = 2.0 Hz, eighth = 4.0 Hz. At 82 BPM: quarter = 1.37 Hz. If pump_rate matches the groove pulse, it is likely **musical strumming/groove**, not comp artifact.
3. **What stem is it on?** Guitar (especially rhythm), bass, drum buses → typically musical pulse. Vocal, sustained pad, master mix → comp artifact more likely.
4. **Depth vs excess profile.** High depth + moderate excess (depth 18 dB, excess 5 dB) = musical pulse. High depth + high excess (depth 8 dB, excess 30 dB) = comp artifact.

If the conclusion is "musical pulse, not artifact": **say so explicitly and do NOT revert**. Note it in the session summary so the next analysis pass doesn't re-flag it as a problem.

### true_peak_dbfs vs sample_peak_dbfs

`analyze.py` reports both:
- `sample_peak_dbfs` — the raw maximum sample value, naive
- `true_peak_dbfs` — 4×-oversampled, ITU-R BS.1770-4 style

The difference is the inter-sample peak (ISP). For low-frequency signals they are nearly identical. For HF content (cymbals, distorted guitar, snare crack) the true peak can sit 0.5-3 dB above the sample peak. After codec encoding (Spotify Ogg/Vorbis, Apple AAC), the encoded signal's inter-sample peak can climb further, occasionally pushing samples above 0 dBFS.

**For stems: the difference rarely matters.** For master delivery: target `-2 dBTP` (the true peak, not the sample peak) to survive streaming codec encoding without clipping. `render_mix.py` does a second-pass true peak measurement after its limiter and scales the master down if the oversampled value exceeds the ceiling.

### onsets_sec, tempo_bpm, estimated_key (rhythm & tonal context)

Three top-level fields in `analysis.json` that give time-domain and tonal context, useful when an EQ / comp / FX choice depends on rhythmic or harmonic content beyond the basic loudness numbers.

| Field | Type | What it is | When to use |
|---|---|---|---|
| `onsets_sec` | list of float seconds | Onset times from librosa onset detection. Same detector as `transient_density_per_sec`, exposed as a raw list. | Identify rhythmic structure; pair-wise stem alignment; precise "uneven playing" detection per onset; visual debugging. |
| `tempo_bpm` | float (or `null`) | librosa `beat_track` estimate. Returns `null` for clips shorter than ~4 s or when the estimate is unstable / out of range (30–300 BPM). | Pick BPM-synced division for `apply_reverb --pre-delay-division` or `apply_delay --bpm`. Sanity-check against the human-known tempo (drummer's clicktrack). |
| `estimated_key` | `{key, mode, confidence}` | Krumhansl-Schmuckler key estimation on `chroma_stft`. Confidence is 0..1 (cosine sim against the reference profile). | Decide whether a tonal mid-EQ move should track the song's key (e.g. boosting 220 Hz on an A-minor track lines up with the root). Drums / overheads / noise will give a low-confidence answer — `< 0.5` means "no reliable key", ignore. |

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

### mix_health verdicts (green / yellow / red)

`mix_health.py` scores the final mix on 7 dimensions: LUFS-vs-target, true peak, LRA, M/S width, low-freq mono compatibility, tonal balance vs reference (if supplied), masking pair counts (if detect_masking was run), and stem pumping.

| Verdict | Meaning |
|---|---|
| All green | Mix is delivery-ready. |
| 1 yellow, rest green | Mix is close. Address the yellow item if it's worth the time; otherwise ship. |
| 2+ yellow OR any red | Address the issues, re-render, re-run mix_health. |
| Red on tonal balance | The bottom/mids/top region differs from the reference by 2-4 dB. Use compare_reference --apply to bake in inverse-delta EQ correction. |
| Yellow on stem pumping | One or more bus stems show 1-5 Hz envelope modulation. Verify per the pumping disambiguation checklist — usually it's musical pulse and can be left alone. |

---

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

LRA < 3 LU on a rock mix: the master compressor or limiter is working too hard. Consider raising the compressor threshold or loosening the attack.

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

## Style Profiles — Reference-Free Genre Grading

`tools/style_check.py mix.wav --style NAME` grades a finished mix against one of five built-in profiles in `tools/style_profiles/`: `modern_rock`, `classic_rock`, `pop`, `hip_hop`, `jazz_acoustic`. The profile fixes loudness, dynamics, and 5-band tonal-balance targets — when no reference track is supplied, the profile **is** the reference.

### What's in a profile

Every profile JSON has the same shape:

| Section | Fields | Meaning |
|---|---|---|
| `lufs` | `integrated_target`, `tolerance_lu` | Genre-typical streaming loudness. Symmetric tolerance — outside `target ± tolerance` is yellow, beyond 1.5× is red. |
| `lra` | `target_lu`, `range_lu` | Loudness Range. Range-based grading: inside [min, max] = green. |
| `crest_factor` | `target_db`, `range_db` | Sample peak vs RMS, range-graded. |
| `tonal_balance_dbfs` | `{sub_60hz, low_60_250hz, mid_250_2khz, high_2_8khz, air_8khz_plus}` each with `target` + `tolerance` | **Wideband band-RMS measured at the profile's LUFS target**, not iZotope-TBC PSD-curve numbers. Calibrated against real-world streaming masters. |

### How the grading works

1. Measure integrated LUFS, LRA, crest factor on the input mix.
2. Apply a single linear gain so the mix sits at the profile's `integrated_target` LUFS.
3. Measure 5-band wideband RMS on the LUFS-normalised mix.
4. Grade each check: GREEN if within tolerance / range, YELLOW just outside, RED significantly outside.
5. Overall score: GREEN-check = 100, YELLOW = 50, RED = 0. Average → 0..100. Verdict thresholds: ≥85 green, ≥60 yellow, below 60 red.
6. **Hard-fail rule**: a RED on `integrated_lufs` or `lra_lu` caps the overall verdict at RED (max 55 score) regardless of the band results. Wrong loudness or wrong dynamics aren't fixable with EQ alone.

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

The shipped profiles are calibrated against published streaming-mastering references and one real-world rock session (terido). To bias them toward a user's preference:

1. Measure 3–5 reference mixes the user considers "right" using `analyze.py` (LUFS-normalise each to the profile's target first).
2. Average the per-band RMS values.
3. Replace the profile's `tonal_balance_dbfs.<band>.target` with the averages. Increase the `tolerance` if the spread across the references is wider than the default ±2.5 dB.
4. Bump the `version` (e.g. 1.1 → 1.2) and add a one-line note under `calibration_note`.

The profiles are versioned and well-commented intentionally so that user-specific overrides remain readable.

---

## Sources

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
