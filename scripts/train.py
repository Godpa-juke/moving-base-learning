#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Marine-UR3-Random6DoFBase-WorldLine-Play-v0")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--max-iterations", type=int, default=1)
parser.add_argument("--seed", type=int, default=818)
parser.add_argument("--run-name", default="fresh_smoke")
parser.add_argument("--resume-checkpoint", default=None)
parser.add_argument(
    "--init-noise-std",
    type=float,
    default=None,
    help="override the policy's initial action standard deviation. Zeroing a residual "
    "actor's output layer is not enough under PPO: the policy still *samples* around "
    "that zero mean, so with the default 0.5 it perturbs the base controller by ~2.5 mm "
    "regardless. Silver et al. used DDPG, where exploration is separate from the actor.",
)
parser.add_argument(
    "--zero-init-residual",
    action="store_true",
    help="zero the actor's output layer at initialisation, so a residual policy starts "
    "as exactly the base controller. Canonical RPL (Silver et al. IV-A); without it the "
    "agent opens by perturbing a working controller and has to climb back.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner
    from marine_manipulator import compat

    cudnn_note = compat.disable_cudnn_rnn_if_unsupported()

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device or "cuda:0"
    agent_cfg.max_iterations = args.max_iterations
    if args.init_noise_std is not None:
        # After the deprecation handler, which migrates the legacy `policy.init_noise_std`
        # into `actor.distribution_cfg.init_std` and clears `policy`. Writing to `policy`
        # here would set an attribute nothing reads.
        actor_cfg = getattr(agent_cfg, "actor", None)
        distribution = getattr(actor_cfg, "distribution_cfg", None) if actor_cfg else None
        if distribution is not None and hasattr(distribution, "init_std"):
            distribution.init_std = args.init_noise_std
        elif getattr(agent_cfg, "policy", None) is not None:
            agent_cfg.policy.init_noise_std = args.init_noise_std
        else:
            raise SystemExit("cannot locate the policy's initial noise std to override")
    log_dir = project / "outputs" / "training" / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if args.resume_checkpoint:
        runner.load(args.resume_checkpoint)
    zero_init_note = None
    if args.zero_init_residual:
        # After construction and after any resume, so it is the state training starts from.
        zero_init_note = compat.zero_initialise_actor_output(runner)
        print(f"[marine] {zero_init_note}", flush=True)
    started = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - started
    env.close()
    models = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    from marine_manipulator import provenance

    result = {
        "task": args.task,
        "provenance": provenance.record(project, log_dir),
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "run_name": args.run_name,
        "cudnn_note": cudnn_note,
        "zero_init_note": zero_init_note,
        "init_noise_std": args.init_noise_std,
        "resume_checkpoint": args.resume_checkpoint,
        "elapsed_sec": elapsed,
        "latest_model": str(models[-1]) if models else None,
        "status": "fresh_train_ok" if models else "fresh_train_missing_checkpoint",
    }
    (log_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("fresh_train_summary", json.dumps(result, sort_keys=True), flush=True)
    if not models:
        raise RuntimeError("Training completed without a checkpoint")
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()
