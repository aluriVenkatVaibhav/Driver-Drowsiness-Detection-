# Pacman RL Project Progress README

This file documents the current state of the Pacman reinforcement learning project,
the fixes made so far, and the direction for the remaining agents. It is separate
from the original `README.md` so the project overview stays intact while experiment
notes can evolve.

## Project Summary

This project extends the Berkeley Pacman environment into a reinforcement learning
experimentation framework. It supports flat RL agents, deep RL agents, hierarchical
Pacman agents, RL ghost agents, shaped rewards, training logs, saved models, and
plots.

The main training entry point is:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config <config_file> --agent <agent_name> --episodes <num_episodes>
```

Available flat agents include:

- `q_learning`
- `sarsa`
- `approx_q`
- `dqn`
- `reinforce`
- `ppo`

The main code areas are:

- `src/train.py`: common training loop for flat agents
- `src/evaluate.py`: evaluation runner for saved models
- `src/agents/`: agent implementations
- `src/models/`: neural network models
- `src/utils/`: state parsing, reward shaping, replay buffer, logging
- `src/results/`: saved models, metrics, and plots
- `src/layouts/`: Pacman map layouts

## Important Config Files

### `src/config.yaml`

General project config. It currently uses `mediumClassic` with two ghosts. This is
harder and slower than the small DQN/PPO experiment setup.

### `src/dqn_config.yaml`

DQN-focused config for the improved baseline. It uses:

- `smallClassic`
- one random ghost
- 500 max steps per episode
- slower/stable DQN learning rate
- DQN epsilon decay by training steps
- reward scaling/clipping
- no `Stop` action when movement is available

Recommended DQN command:

```powershell
python train.py --config dqn_config.yaml --agent dqn --episodes 2000
```

### `src/dqn_directional_config.yaml`

Hard-condition DQN config. It trains DQN against `DirectionalGhost`
(`multiagent.mode: baseline`) and writes to a separate results folder:

- model path: `src/results/dqn_directional/dqn_best.pt`
- logs/plots under: `src/results/dqn_directional/`

This protects the strong random-ghost DQN baseline from being overwritten.

Recommended hard-DQN command:

```powershell
python train.py --config dqn_directional_config.yaml --agent dqn --episodes 3000
```

### `src/ppo_config.yaml`

PPO-focused config created to avoid confusion with `dqn_config.yaml`. It keeps the
same `smallClassic` environment for fair comparison against DQN, but names the file
for PPO experiments. PPO now writes to its own isolated output folder:

- model path: `src/results/ppo/ppo_best.pt`
- logs/plots under: `src/results/ppo/`

Recommended PPO command:

```powershell
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

### `src/ppo_stable_config.yaml`

Stability-focused PPO config. It writes to a separate output folder and uses more
conservative PPO updates:

- model path: `src/results/ppo_stable/ppo_best.pt`
- logs/plots under: `src/results/ppo_stable/`
- lower PPO learning rate
- larger rollout batches
- fewer PPO epochs per rollout
- smaller PPO clip range
- `ppo_target_kl` early stopping
- slower entropy decay with a higher entropy floor

Recommended stable PPO command:

```powershell
python train.py --config ppo_stable_config.yaml --agent ppo --episodes 2000
```

### `src/hierarchical_ghost_config.yaml`

Hierarchical ghost focused config. It uses `smallClassic`, one hierarchical ghost,
and writes to a dedicated folder:

- model path: `src/results/hierarchical_ghost/hierarchical_ghost_1.pkl`
- Pacman hierarchy artifacts under `src/results/hierarchical_ghost/`
- logs/plots under `src/results/hierarchical_ghost/`

Recommended smoke command:

```powershell
python hierarchical_train.py --config hierarchical_ghost_config.yaml --episodes 40
```

Recommended training command:

```powershell
python hierarchical_train.py --config hierarchical_ghost_config.yaml
```

### `src/q_config.yaml`

Q-learning focused config for the same `smallClassic` random-ghost environment.
It uses tabular-specific exploration and optimistic initialization.

The Q-table uses a compact state abstraction rather than exact Pacman coordinates.
This keeps the table small enough for 2000-episode training to generalize instead
of memorizing one-off board positions.

`q_config.yaml` also enables deterministic checkpoint evaluation. The saved
`q_learning_best.pt` is selected from no-learning evaluation games, not from
epsilon-noisy training episodes.

Recommended Q-learning command:

```powershell
python train.py --config q_config.yaml --agent q_learning --episodes 2000
```

Using `dqn_config.yaml` for PPO will not damage DQN results because model and log
outputs are prefixed by agent name, but `ppo_config.yaml` is cleaner.

## Output Safety

Training saves outputs using the selected agent name:

- DQN model: `src/results/dqn_final.pt`
- PPO model: `src/results/ppo_final.pt`
- DQN logs: `src/results/logs/dqn_metrics.*`
- PPO logs: `src/results/logs/ppo_metrics.*`
- DQN plots: `src/results/plots/dqn_*.png`
- PPO plots: `src/results/plots/ppo_*.png`

So running:

```powershell
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

does not overwrite the DQN model or DQN metrics.

## DQN Work Completed

The DQN agent was the first target because it was failing to learn reliably after
2000 episodes. The old behavior showed mostly negative rewards and weak/no win-rate
improvement.

Key changes made:

- Added Double DQN target selection/evaluation.
- Added soft target-network updates with Polyak averaging.
- Added warmup before optimization.
- Added step-based epsilon decay from config.
- Added gradient norm clipping.
- Replaced the DQN model with a dueling DQN architecture.
- Added legal-action masking to replay targets so DQN no longer bootstraps from
  impossible next actions.
- Added optional `Stop` avoidance when movement is available.
- Added reward scaling/clipping before DQN optimization.
- Added dense shaped reward for DQN Pacman transitions.
- Added terminal shaped reward consistency in the final update.
- Added max-steps-per-episode support to avoid stuck runs.

Important DQN files:

- `src/agents/dqn_agent.py`
- `src/models/dqn_net.py`
- `src/utils/replay_buffer.py`
- `src/utils/reward_shaper.py`
- `src/utils/state_parser.py`
- `src/train.py`
- `src/dqn_config.yaml`

## DQN Result Inference

The latest DQN run showed strong learning:

- Early training: rewards were negative and win rate was zero.
- Around episode 230-320: rewards started becoming positive.
- Around episode 330 onward: wins began appearing.
- Around episode 900-1600: trailing win rate reached roughly 0.8 to 0.93.
- Final episode 2000: trailing win rate was around 0.86 with strongly positive reward.

Conclusion: DQN is now a working baseline. It should be protected while improving
other agents.

Future DQN improvements, if needed:

- Save the best checkpoint by trailing win rate, not only the final checkpoint.
- Add deterministic evaluation checkpoints during training.
- Log epsilon, loss, score, win rate, and runtime into CSV together.

## Directional-Ghost DQN

The random-ghost DQN baseline evaluates well, around 75-80% win rate in 20-game
checks. Against `DirectionalGhost`, it is much weaker because the training
environment was different.

To make this comparison clean, `dqn_directional_config.yaml` was added. It trains
and evaluates DQN under directional ghosts while writing to `results/dqn_directional`
so it does not overwrite `results/dqn_best.pt`.

## Hierarchical Ghost Work Started

Hierarchical ghost is now isolated from DQN, Q-learning, and PPO through
`hierarchical_ghost_config.yaml` and `results/hierarchical_ghost`.

Key changes:

- Replaced the previous minimal ghost with a richer hierarchical ghost controller.
- Added high-level ghost intents: `chase_pacman`, `ambush`, `guard_capsule`, and
  `flee_scared`.
- Added compact tabular state features for ghost learning: goal, Pacman distance
  bucket, target direction, Pacman direction, scared-timer bucket, capsule
  distance bucket, and local legal-action count.
- Added a heuristic prior so an untrained ghost already chases, ambushes, guards
  capsules, and flees while scared in a sensible way.
- Added real epsilon exploration for ghost learning. The first implementation was
  too deterministic during learning because exploration still picked the
  heuristic-best action.
- Added goal-specific ghost shaping: catch Pacman bonus, scared-ghost survival
  behavior, progress toward target, and step penalty.
- Fixed `hierarchical_train.py` so hierarchical ghost receives both
  `hyperparameters` and `hierarchical` config values. Previously ghost-specific
  hierarchical config values were not passed to `HierarchicalGhostAgent`.
- Added max-step caps to hierarchical episodes to avoid stuck runs.
- Added `--episodes` override for short hierarchical smoke tests.
- Added ghost goal stats to training output.
- Removed a non-ASCII completion checkmark from `hierarchical_train.py` because
  Windows PowerShell cp1252 output crashed after a completed run.

Smoke test completed:

```powershell
python hierarchical_train.py --config hierarchical_ghost_config.yaml --episodes 40
```

The run completed all four phases and saved artifacts to
`src/results/hierarchical_ghost`.

## Training Loop Improvements Completed

The common training loop now prints:

- run start time
- run end time
- total runtime
- epsilon/exploration value when the agent exposes it

Example output:

```text
Starting training for 2000 episodes with agent dqn
Run started at: 2026-04-21 02:53:38
Episode 10/2000 | Avg Reward (last 10): -412.7 | Win Rate: 0.00 | Eps: 0.99
...
Training Complete. Results saved.
Run ended at:   2026-04-21 03:42:10
Total runtime:  00:48:32.12 (HH:MM:SS)
```

## PPO Work Started

PPO was selected as the next agent because it is the next deep RL baseline and is
more likely than tabular agents to suffer from scale, variance, and policy-update
issues.

Changes made so far:

- Added PPO-specific reward scaling and clipping.
- Added PPO-specific `Stop` avoidance.
- Avoided bootstrapping terminal states with a learned critic value.
- Switched critic loss from MSE to Huber loss for more stable value learning.
- Reset PPO trajectory bookkeeping after each update.
- Added `src/ppo_config.yaml`.
- Added shaped Pacman rewards for PPO in the training loop.
- Added PPO-specific learning rate through `ppo_learning_rate`.
- Changed PPO action distributions to use masked logits directly.
- Added PPO loss and entropy reporting from the final episode update.
- Added entropy printing in the training trace as PPO's exploration diagnostic.
- Changed PPO to update from 5-episode rollout batches instead of one episode at
  a time.
- Increased PPO entropy pressure in `ppo_config.yaml` to reduce premature policy
  collapse.

Important PPO files:

- `src/agents/ppo_agent.py`
- `src/models/actor_critic_net.py`
- `src/ppo_config.yaml`

PPO smoke test completed:

```powershell
python train.py --config dqn_config.yaml --agent ppo --episodes 5
```

The run completed successfully and printed runtime information.

Initial full PPO run before the latest fixes did not learn:

- rewards stayed around -400 to -500
- win rate stayed at 0.00 through 2000 episodes

Diagnosis:

- PPO was receiving raw score deltas while DQN received dense shaped rewards.
- PPO inherited the conservative DQN learning rate unless separately configured.
- PPO had no exploration diagnostic in the console.

After the fixes, a 200-episode PPO check showed reward improvement from roughly
`-415` early to around `-230` near episode 200. This is not a solved PPO model yet,
but it is a clear improvement over the failed flat trace.

A later 2000-episode PPO run improved much more than the original failed run, but
still remained weak:

- rewards gradually improved from around `-400` toward occasional near-zero or
  positive windows
- first wins appeared very late, around episode 1570
- win rate stayed low and unstable
- entropy sometimes collapsed near zero, meaning the policy became too
  deterministic too early

Follow-up PPO changes were added after that run:

- PPO now updates every 5 episodes to reduce variance.
- PPO entropy coefficient in `ppo_config.yaml` is now `0.06`.
- Entropy output is reused between update episodes so exploration is visible in
  the console trace.
- PPO now has a faster entropy schedule: `0.05` down to `0.003` over 150 update
  batches.
- PPO uses `ppo_batch_size: 128` to make neural updates more GPU-friendly.
- The actor-critic network now uses a DQN-sized CNN backbone.
- Training now saves `<agent>_best.pt` separately from `<agent>_final.pt` when
  trailing win rate/reward improves.
- PPO now uses clipped value updates (`ppo_value_clip`) and a PPO-only learning
  rate decay schedule.
- Training output now prints PPO learning rate in addition to entropy and entropy
  coefficient.

Latest 200-episode check after batched updates:

- entropy stayed healthier, roughly `0.6-0.8`
- rewards improved in the second half, reaching around `-240` to `-290`
- this is more stable, though not solved yet

After the actor-critic and entropy-schedule update, a 300-episode probe showed:

- CUDA was active: `Training device: cuda`
- entropy coefficient decayed from `0.050` to about `0.032`
- rewards improved into the `-150` to `-190` range by the end of the probe
- wins still did not appear in 300 episodes, so full-run behavior must be checked

Latest follow-up changes after the weak 2000-episode PPO run:

- Added `ppo_learning_rate_end: 0.00008`
- Added `ppo_lr_decay_updates: 150`
- Added `ppo_value_clip: 0.2`

These are intended to reduce late-training oscillation without touching DQN.

Current PPO status as of 2026-05-11:

- PPO has a real on-policy implementation: masked action sampling, GAE,
  clipped policy loss, clipped value loss, entropy scheduling, learning-rate
  scheduling, gradient clipping, and multi-episode rollout updates.
- Deterministic evaluation of the existing `ppo_best.pt` is still poor:
  20-game check on `smallClassic` with one random ghost gave 0% wins and about
  `-441.6` average score.
- Stochastic evaluation of the same checkpoint is better but still unsolved:
  20-game check gave 0% wins and about `-178.75` average score.
- PPO now treats max-step truncation as non-terminal for its final rollout
  transition, so the critic can bootstrap from timeout states. DQN and
  Q-learning keep their previous terminal handling.
- `ppo_config.yaml` now enables deterministic checkpoint evaluation every 100
  episodes using 20 eval games, so future `ppo_best.pt` checkpoints are selected
  from no-learning policy evaluation rather than noisy training windows.
- `PpoAgent.save()` now writes checkpoints through a temporary file and atomically
  replaces the target path. If Windows refuses to replace an existing checkpoint,
  PPO falls back to a timestamped checkpoint path instead of crashing training.
- A 2000-episode PPO run on 2026-05-11 completed successfully in about 11 minutes.
  It improved training rewards substantially, with some 10-episode windows around
  `-50` to `-100`, but deterministic checkpoint evaluations still stayed at 0%
  win rate. The best checkpoint eval score in that run was around `-163.3` at
  episode 1100.
- PPO checkpoint selection now only updates `ppo_best.pt` on deterministic
  checkpoint-eval intervals when checkpoint evaluation is enabled. This prevents
  noisy training windows, such as the episode-250 `-84.8` window from the latest
  run, from overriding the deterministic best checkpoint.
- `ppo_config.yaml` now uses `results_dir: "./results/ppo"` and
  `evaluation.model_path: "./results/ppo/ppo_best.pt"` so PPO does not overwrite
  or compete with DQN/Q-learning outputs.
- PPO now has its own `ppo_avoid_deadly_actions` action filter. During PPO action
  selection, immediate losing moves are masked out when at least one non-losing
  move exists. This mirrors the safety benefit Q-learning gets from its
  exploration heuristic, but is implemented only inside `PpoAgent`.
- A 100-episode smoke run after the PPO-only safety mask showed much stronger
  early behavior: training windows became positive by episode 60-100 and the
  episode-100 deterministic checkpoint eval improved to about `-73.5` average
  score. It still had 0% wins, so the next required check is a full 2000-episode
  PPO run with the updated config.
- A later PPO run with the safety mask showed real learning but high policy
  volatility: deterministic checkpoint eval reached 10% win rate at episode 200
  and 25% win rate at episode 600, but surrounding eval checkpoints dropped back
  to 0-5%. Training win-rate windows also oscillated. This suggests PPO is finding
  good policies and then partially overwriting them.
- PPO now supports `ppo_target_kl` early stopping and prints the latest KL value
  when available. This is PPO-only and does not affect DQN/Q-learning.
- Added `ppo_stable_config.yaml` for more conservative PPO runs in
  `results/ppo_stable`. A 30-episode smoke test completed successfully and printed
  KL values below the configured `0.015` guard.
- PPO now has PPO-only anti-loop and ghost-danger shaping:
  `ppo_avoid_ghost_radius`, `ppo_loop_window`, `ppo_revisit_penalty`,
  `ppo_empty_step_penalty`, and `ppo_ghost_danger_penalty`. These penalize
  revisiting recently used empty cells, moving without food progress, and ending a
  transition near an active ghost. The action filter also avoids moves that place
  Pacman next to an active ghost when another legal move exists.
- A 30-episode stable PPO smoke test after the anti-loop/ghost-danger change ran
  successfully with KL around `0.0001-0.0003`. This only verifies code health; the
  real learning check should be a 2000-3000 episode stable PPO run.

Recommended next PPO test:

```powershell
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

## DQN Protection Rule

While improving other agents, avoid changing these unless explicitly needed:

- `src/agents/dqn_agent.py`
- `src/models/dqn_net.py`
- `src/dqn_config.yaml`
- DQN-specific reward handling in `src/train.py`

Shared utility changes should be backward-compatible and should not alter DQN
training semantics.

## Suggested Roadmap

1. Keep the random-ghost DQN as the main solved baseline.
2. Train hard-condition DQN with `dqn_directional_config.yaml`.
3. Run/retrain Q-learning with `q_config.yaml` after the latest state-key changes.
4. Treat PPO as a weak baseline unless revisited later.
5. Improve SARSA next, using the Q-learning fixes as a base.
6. Improve REINFORCE after SARSA.
7. Return to hierarchical agents after flat baselines are stable.

If PPO is revisited, tune PPO-specific values only:
   - `ppo_reward_scale`
   - `ppo_reward_clip`
   - `entropy_coef`
   - `ppo_epochs`
   - `gae_lambda`

## Current Recommended Commands

DQN:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config dqn_config.yaml --agent dqn --episodes 2000
```

Directional DQN:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config dqn_directional_config.yaml --agent dqn --episodes 3000
```

Q-learning:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config q_config.yaml --agent q_learning --episodes 2000
```

PPO:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

Evaluate DQN:

```powershell
python evaluate.py --config dqn_config.yaml --agent dqn --episodes 20 --headless
```

Evaluate PPO:

```powershell
python evaluate.py --config ppo_config.yaml --agent ppo --episodes 20 --headless
```

Evaluate Q-learning:

```powershell
python evaluate.py --config q_config.yaml --agent q_learning --episodes 20 --headless
```

Evaluate directional DQN:

```powershell
python evaluate.py --config dqn_directional_config.yaml --agent dqn --episodes 20 --headless
```

Evaluation now reads defaults from each agent config's `evaluation` section:

- `ghost_mode` controls the evaluation ghost behavior.
- `model_path` can point to the preferred checkpoint, usually `*_best.pt`.
- If `model_path` is absent, evaluation tries `*_best.pt` and then `*_final.pt`.
- `--ghost_mode` and `--model_path` can still override the config for stress tests.

Example DQN stress test against directional ghosts:

```powershell
python evaluate.py --config dqn_config.yaml --agent dqn --episodes 20 --headless --ghost_mode baseline
```

## Notes

The DQN improvement is significant and should be treated as the current baseline.
PPO is now prepared for a serious 2000-episode run, but it has not yet been proven
the way DQN has. Its results should be evaluated from logs and plots after the full
run.

For runtime, `plot_interval` was added to the config files. Models/logs still save
every `save_interval`, but plots are generated less often. This should not affect
learning quality because plotting is only reporting overhead.

PyTorch CUDA was verified in the project venv:

```text
torch 2.7.1+cu118
cuda_available True
device NVIDIA GeForce RTX 4060 Laptop GPU
```

The neural networks run on GPU, but the Berkeley Pacman environment remains mostly
CPU/Python-bound. Fully making training GPU-bound would require vectorizing or
rewriting the environment, not just moving the model to CUDA.
