# music-mix-agent — Claude session instructions

## What this project is

AI-assisted multi-track mixing pipeline for recorded stems (rock band, orchestral, etc.).
Claude acts as the agent: analyzes stems, reads the output data, proposes and applies processing.
All processing happens via Python CLI tools in `tools/`. Claude orchestrates them via Bash.

## Session start checklist

1. Read `docs/knowledge.md` — domain knowledge base (LUFS targets, per-instrument guidelines, trends).
2. Ask the user what session/folder they are working with today.
3. Check `output/` for any previous analysis runs on that session.
4. Ask what the goal is before running anything.

## Python environment

Always use the dedicated conda env — never base conda, never system Python.

```bash
/media/cseti/datassd/conda/miniconda3/envs/music-mix-agent/bin/python tools/<script>.py
```

## Available tools

| Tool | What it does | Key args |
|---|---|---|
| `tools/analyze.py` | Analyze a stem: LUFS, peak, freq bands, noise floor, text+PNG spectrogram | `<file> --output-dir output/<session>` |
| `tools/apply_gain.py` | Apply gain to a stem | `<file> --output-dir output/<session> --from-analysis <analysis.json>` or `--preset stem\|spotify\|apple\|broadcast` or `--gain-db <n>` |

## Output structure

Every tool writes into `output/<session>/<stem_name>/`:

```
output/
└── <session>/
    └── <stem_name>/
        ├── analysis.json
        ├── spectrogram.png
        ├── spectrogram.txt     <- read this to interpret frequency content
        ├── gain_report.json
        └── <stem_name>_gained.wav
```

## Workflow

```
analyze -> read spectrogram.txt + analysis.json -> propose processing -> apply -> verify
```

- Never apply processing without reading the analysis first.
- Always read `spectrogram.txt` from output — it is the primary way to understand what is in a stem.
- After applying any processing, re-analyze the output file to verify the result.
- Ask before processing multiple stems in bulk — do one first and confirm it is correct.

## Ground rules

- One stem at a time until the user confirms the result is correct.
- State what you observe from the analysis before proposing any action.
- If a result looks wrong (clipping, unexpected LUFS), stop and diagnose before continuing.
- Keep `docs/knowledge.md` updated when new domain knowledge is found.
