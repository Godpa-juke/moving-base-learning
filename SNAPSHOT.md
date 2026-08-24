# Snapshot provenance

## Source

- Canonical source: private experiment tree
- Source branch: `uvms-free-floating`
- Source commit: `21ef52136bcb453e1225dd889ee9b5dabb6493b6`
- Snapshot purpose: public code and qualitative visualization for the accompanying manuscript
- Git history policy: this public snapshot uses an independent history; the private canonical history is not copied

## Export allowlist

Only the following source paths were exported from the exact source commit:

- `.gitignore`
- `pyproject.toml`
- `src/`
- `scripts/`
- `tests/`

The public README, this provenance file, media, and one test-contract correction were added in the isolated snapshot tree.

## Public test-contract correction

The source commit's target-conditioned test still expected an earlier radial-random command API (`center_distance_range` and `center_azimuth_range`). The runtime implementation had intentionally moved to the later visible-line contract: the full bar is placed in world `x ∈ [-0.40, -0.30] m`, oriented along world `+Y`, with episode-random Y/Z center and length. The public snapshot updates only that stale test and its AppLauncher wrapper; runtime source is unchanged.

## Excluded

- `outputs/`
- `logs/`
- checkpoints and optimizer state
- TensorBoard event files
- raw evaluation summaries and trajectory arrays
- internal research notes under `docs/`
- manuscript working trees under `paper/`
- `.claude/` and other local assistant state
- the private repository `.git/` history
- Isaac Sim and Isaac Lab installations
- third-party robot assets and collected papers

## Demonstration media

- Public file: `media/moving_base_line_tracking.mp4`
- Poster: `media/demo_poster.png`
- Source rollout: fresh residual-policy rollout, task `Marine-UR3-ResidualIkDelay-Play-v0`, seed `971`, selected checkpoint iteration `2350`
- Camera preset: `topdown-45` (45-degree downward oblique view)
- Encoding: H.264, yuv420p, 1280×720, 30 fps
- Duration: 7.97 seconds
- Decoded frames: 239
- Rollout receipt: captured; cross-track RMSE `8.39 mm`; post-capture cross-track p95 `12.32 mm`

The media is a qualitative visualization. It is not used as population evidence for the reported multi-seed comparisons.
