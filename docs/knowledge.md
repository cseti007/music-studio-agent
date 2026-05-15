# Domain Knowledge — Audio Mixing & Gain Staging

Reference material for tool decisions and recommendations. Update when new research is done.

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
