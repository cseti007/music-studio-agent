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

## Master Bus Chain Order

Processing order matters:
1. **Bus saturation** (per bus, before summing to master)
2. **Master sum** (all top-level buses)
3. **Master glue compressor** (2:1, slow-ish attack, catches sustained program material)
4. **Master EQ** (zero-phase — no phase coloration; HP@30Hz + gentle high shelf)
5. **LUFS normalization** (target -14 LUFS for streaming)
6. **True peak limiter** (-2 dBTP)

### Master glue compressor settings (pedalboard Compressor notes)

Pedalboard's Compressor uses peak detection. With slow attack (>20ms), short transients pass through and the measured peak GR appears 0. For program material compression, use:
- threshold: -10 dBFS (works with typical -12 to -9 LUFS pre-norm signals)
- attack: 10ms (catches sustained peaks while letting some transient through)
- ratio: 2:1
- release: 300ms
- Expected GR: -0.5 to -1.0 LUFS on a dense rock mix

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
