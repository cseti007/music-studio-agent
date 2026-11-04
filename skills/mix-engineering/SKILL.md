---
name: mix-engineering
description: Mixing and mastering method for multitrack music - how to plan a mix, interpret audio analysis (LUFS, true peak, LRA, crest factor, spectrum, masking, mix_health, reference comparison), choose and justify EQ / compression / gating / reverb / delay / saturation moves, and handle listening evidence and approval. Use it whenever an agent mixes, balances, processes or masters recorded tracks, either offline on stems with the music-mix-agent tools or live in a DAW (for example Ardour through the ardour-agent MCP tools), and whenever it has to decide what an analysis result means or whether a mix is ready.
---

# Mix engineering

DAW-agnostic mixing method from the music-mix-agent project. This file holds
the core rules; the details live in two references (read the sections you
need, not the whole files):

- `references/knowledge.md` - domain knowledge: gain staging, panning, EQ
  starting points per instrument, reverb, mastering, metric meanings.
- `references/offline-pipeline.md` - the music-mix-agent agent instructions
  (offline stem pipeline). Besides tool usage it contains the analysis
  interpretation table and the review rules referenced below.

User and project instructions take precedence over this skill. Existing user
approvals and authorizations carry forward; do not ask again for authorized work.

## Pick the backend

| Situation | How processing happens | What to follow |
|---|---|---|
| Offline stems (music-mix-agent repo) | Python tools render processed WAV files | `references/offline-pipeline.md` in full |
| Live DAW (e.g. Ardour via ardour-agent MCP) | Plugin parameters, faders, pans, sends on tracks and busses - non-destructive | This file + the reference sections listed below |

Live DAW rules:

- Process inside the DAW. Do not render files with the offline `apply_*`
  tools and import them back unless the user asks for that.
- Analyze exported or bounced audio with the MCP analysis tools
  (`analyze_file`, `detect_masking`, `compare_reference`, `mix_health`,
  `find_clicks`, `prepare_audition`). `mix_health` and `find_clicks` expect a
  folder with `mixes/mix.wav` (and optionally `stems/*.wav`).
- Presets (`list_presets`, `get_preset`) are starting-point hypotheses written
  for the offline tools. When translating one to a DAW plugin, state the
  mapping and every approximation (different filter shapes, missing
  parameters, different detector behavior).
- Read back what you set (gain, plugin parameters) and report the actual
  value, not the requested one.
- If the DAW has its own MCP server (Ardour 9.7+: "MCP HTTP Server
  (Experimental)"), use it for DAW operations on live state (tracks, faders,
  plugins, regions, undo) and the project's server for what it lacks (import,
  export, analysis, rendering, blind auditions). The project's own
  instructions (e.g. ardour-agent/CLAUDE.md) say which tool covers what.
- Deliver audition clips where the listener is (e.g. lossless FLAC to a phone),
  loudness matched, and reveal a blind key only after the answer.

## Session start

1. Declare listening capability: direct audition, external human feedback, or
   measurements only. An audio file on disk or a playback command does not
   mean anything was heard.
2. Establish source material, artistic intent, takes, reference tracks and
   requested deliverables.
3. Preserve original levels, channel relationships and edits. Normalization,
   trimming and editorial changes need a stated purpose.
4. Map sections with real timestamps (intro, quiet and dense passages,
   transitions, solos, ending). Mark inferred section labels as provisional.

## Evidence and approval

- Keep three evidence types apart in every decision: measured observation,
  inferred cause, heard or user-reported result. Never present an inference as
  something heard.
- Metrics, presets, style profiles and green health scores generate hypotheses.
  They never certify musical quality, genre authenticity or readiness.
- Compare before/after at matched loudness on representative sections. Change
  one coherent group of settings per comparison.
- For each consequential change record: problem, timestamp, hypothesis,
  intervention, measurements, comparison, feedback, keep/revert decision.
- Feedback is scoped: "the balance feels right" approves that balance in the
  heard excerpt, not timbre, dynamics, the whole song or a later master. Renew
  approval when the audio changes. Criticism is a revision request.
- First renders and exports are drafts ("rendered for review", "technical
  checks passed; listening pending"). Never fabricate a quote, reviewer or
  approval. Never claim professional quality or artist equivalence.
- Without direct audition, keep working on authorized tasks and offer
  reviewable alternatives, but keep subjective conclusions provisional.

## Workflow

1. Validate inputs; keep the timeline and original levels.
2. Set balance and pan first. EQ, dynamics, alignment and effects are optional
   interventions, each with a specific hypothesis - not mandatory stages.
3. Judge the drum kit as one instrument (close mics with overheads and rooms),
   then bass interaction, guitar mic blends, vocal phrase consistency, depth
   and transitions, in context.
4. Keep chains as simple as the job permits; review earlier choices before
   stacking another effect. Do not add processing to satisfy a score or a
   genre recipe.
5. Check the mix: resolve technical failures (clipping, nonfinite samples,
   broken routing); treat LUFS, LRA, width, masking and tonal targets as
   advisory. Collect listening feedback before calling it approved.
6. Master to the requested delivery profile only. Avoid accidental duplicate
   limiting. Verify format, peaks and any contractual loudness on the exported
   file.

## Reference snare matching

When the user wants their snare to sound like the snare on reference songs:

1. `snare_profile.py` on each reference song. Check `status`, the number of hits,
   and listen to `private_ref/` hits with the user if the profile looks odd.
2. `snare_match.py` with all profiles and licensed libraries. If `song_spread`
   warns, ask whether to use one target per song. `no_acceptable_match` is an
   answer, not a reason to force a sample.
3. Export the kit (all drum tracks, trigger plugins off) from the same start as
   the close snare track; `render_snare_layer.py` with `--kit`, triggered from
   the snare top. When the snare has top and bottom microphones, use the
   candidate's `_top` file as the sample and its `_bottom` file as
   `--companion`, and place each layer next to its own track: the bottom
   microphone carries most of the wire sound.
4. Place the layers at the snare tracks' position. Find the level with a blind
   level sweep in the full mix (e.g. the user's blend level, +6, +12 dB, and
   replacing both snare tracks); a reference character may only appear once
   the layer carries most of the snare.
5. Measure the full mix before and after with `snare_profile.py --compare-to`,
   then a blind loudness-matched A/B for the user. D is a hypothesis; the
   user's ear decides.

Never place hits cut from a reference (`private_ref/`, `--private`) in a mix
that will be released. Credit the sample library as its license requires.

## Interpreting analysis

Always end an analysis with a verdict, which may be "no action needed". Do not
invent problems and do not recommend processing without a specific reason in
the data. Normal ranges and how to act on each field are in
`references/offline-pipeline.md`:

- "Analysis interpretation — what to say after reading analysis.json" (field table)
- "Interpreting pumping_detected"
- "Analysis tool decision tree - when to run what"
- "Reference comparison workflow" (masking and reference results are hypotheses;
  never apply inverse spectral matching automatically)
- "Creative processing decision rules" and "Production finishing review"
  (when the user asks for a more modern, finished or impactful record)

## Where to look in references/knowledge.md

| Topic | Section |
|---|---|
| Evidence and listening policy | "Evidence and research policy", "Listening evidence and review policy", "Decision policy" |
| Gain staging, stage targets, premaster handoff | "Gain Staging" |
| Panning (band size, drum kit perspective) | "Panning convention by band size" |
| Frequency bands, noise floor | "Frequency Bands (reference)", "Noise Floor Reference" |
| EQ filter types and per-instrument starting points | "EQ — Filter Types", "EQ — Instrument Starting Points" |
| Kit balance | "Drum internal balance - kit relationships" |
| Reverb choice and settings | "Reverb in a Rock Mix" |
| Mastering philosophy, chain order, delivery | "Mastering Workflow and Philosophy", "Master Bus Chain Order" |
| Metric meanings (band crest, pumping, true peak, envelopes) | "Reading the New Analysis Metrics" |
| DAW session files (Pro Tools, Ableton) | "DAW Session File Formats" |

## Communication

- Announce long operations and report each result in one line.
- For plans with 3 or more steps, show a checklist (`[ ]` pending, `[>]` in
  progress, `[x]` done) and reprint it as steps complete.
- At the end report the state (draft / technically checked, listening pending /
  revision requested / approved for the stated delivery), the decisions made,
  the exact files or session state, and what remains unresolved.
