# Snapshot provenance

## Source

- Canonical source: private experiment tree
- Source branch: `uvms-free-floating`
- Source commit: `fa2a6231e18b96d5e6b5a957f58d7e8b7f023113`
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
- Source rollout: `xneg30to40cm_finalvisible_seed47-step-0.mp4`
- Encoding: H.264, yuv420p, 1280×720, 30 fps
- Duration: 10.97 seconds
- Frames: 329

The media is a qualitative visualization. It is not used as population evidence for the reported multi-seed comparisons.
