# Domain Knowledge — Audio Mixing & Gain Staging

Reference material for tool decisions and recommendations. Update when new research is done.

---

## Stem Assembly — before any processing

**Rule: always assemble a full channel track before gain staging or any other processing.**

Individual recording clips are not the same as a full channel stem. When a musician records in
multiple takes or sections, the DAW holds many small clips on the timeline. Gain staging on
individual clips distorts the natural dynamic balance between sections (a softly-played section
would get boosted to the same level as a loud one). The EBU R128 gating handles silence, but
clip-level normalization still destroys intra-performance dynamics.

The correct order is:
```
1. assemble_channel  ->  full-length WAV per logical track
2. analyze           ->  read the assembled channel
3. apply_gain        ->  gain stage the assembled channel
4. ...further processing
```

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

### parse_session.py tool (planned)

A unified tool that auto-detects the session format and outputs a canonical JSON:
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

This canonical format is the input for `assemble_channel.py`.

---

## Gain Staging

### LUFS targets by use case

| Context | Integrated LUFS | True Peak | Notes |
|---|---|---|---|
| Stem in mix | -18 LUFS | -3 dBTP | headroom for bus processing |
| Pre-master mix bus | -18 to -14 LUFS | -3 to -6 dBFS peak | leave room for mastering chain |
| Spotify delivery | -14 LUFS | -1 dBTP | platform default (Normal mode) |
| Apple Music delivery | -16 LUFS | -1 dBTP | Sound Check normalization |
| YouTube delivery | -13 / -14 LUFS | -1 dBTP | |
| Amazon Music delivery | -14 LUFS | -1 dBTP | |
| Broadcast (EBU R128) | -23 LUFS | -1 dBTP | TV/radio standard, too quiet for music |

**Key rule:** -23 LUFS is broadcast standard, NOT a music mixing target. Use -18 for stems.

### apply_gain.py presets (derived from above)

| Preset | Target LUFS | True peak limit | Use when |
|---|---|---|---|
| `stem` | -18 LUFS | -1 dBTP | individual stem going into a mix |
| `premix` | -18 LUFS | -3 dBTP | alias for stem, more conservative peak |
| `spotify` | -14 LUFS | -1 dBTP | final delivery to Spotify |
| `apple` | -16 LUFS | -1 dBTP | final delivery to Apple Music |
| `broadcast` | -23 LUFS | -1 dBTP | TV/radio/podcast delivery |

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

| Signal type | Typical noise floor | Notes |
|---|---|---|
| Clean DI recording | -80 to -90 dBFS | very clean |
| Good studio mic | -70 to -80 dBFS | acceptable |
| Live/location recording | -60 to -70 dBFS | some background noise |
| Problematic | above -60 dBFS | noise removal recommended |

---

## True Peak vs Peak

- **Sample peak (dBFS)**: highest sample value — what DAW meters typically show
- **True peak (dBTP)**: inter-sample peak — can exceed sample peak after D/A conversion
- Always limit true peak to -1 dBTP for streaming delivery to avoid distortion after conversion
- During mixing: -3 dBTP gives safe headroom

---

## Sources

- [Gain Staging Explained 2026 — Mixing Monster](https://mixingmonster.com/gain-staging/)
- [Understanding LUFS 2026 — Mixing Monster](https://mixingmonster.com/understanding-lufs/)
- [The Rise of Smart Gain Staging — DLK Music Pro (Feb 2026)](https://news.dlkmusicpro.com/the-rise-of-smart-gain-staging-in-modern-audio-production/)
- [Mastering for Streaming: LUFS Targets 2026 — Genesis Mix Lab](https://genesismixlab.com/guides/mastering-delivery/)
- [Mastering for Streaming Platforms — iZotope](https://www.izotope.com/en/learn/mastering-for-streaming-platforms)
