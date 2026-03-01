=============================================================================
PROJECT: AUTONOMOUS AGENTS VIA LATENT WORLD MODELS & IMITATION LEARNING
=============================================================================

Overview
--------
This repository contains the complete pipeline for a state-of-the-art AI 
engineering capstone project. It explores the training of an autonomous 
agent to play a custom 2D platformer game. 

The core thesis of this project is to demonstrate that pre-training an 
unsupervised "World Model" (an internal physics engine) drastically 
improves an agent's sample efficiency, Out-Of-Distribution (OOD) robustness, 
and translation invariance compared to standard end-to-end learning from scratch.

Project Progression & Phases
----------------------------

PHASE 1: Continuous World Models (Bouncing Ball)
- Started with a simple 2D bouncing ball environment, no actions, world_model_rect.py
- Built a continuous Convolutional Tokenizer and a Spatio-Temporal Transformer.
- Proved that a Transformer can autoregressively predict video frames and 
  learn basic bounce physics.


PHASE 2: Discrete Latents & Gravity (Simple Platformer)
- Upgraded to a simple platformer with jumping physics.
- Replaced the continuous autoencoder with a Vector Quantized Variational 
  Autoencoder (VQ-VAE). 
- Demonstrated that compressing pixels into discrete "tokens" prevents 
  blurring and allows the Transformer to confidently predict exact states.

PHASE 3: The Colorful Platformer & Expert Data
- Upgraded to `ColorfulPlatformerEnv`, a highly structured visual environment 
  with specific target platforms.
- Wrote `generate_expert_dataset.py` to collect thousands of trajectories 
  of a hard-coded expert successfully navigating the level, generating 
  our baseline Behavioral Cloning dataset.

PHASE 4: World Model Pre-Training
- Trained `vqvae_supervised_highcap.pth`: The "eyes" of the agent, converting 
  raw Pygame pixels into a 16x16 grid of structural tokens.
- Trained `wm_supervised_highcap_big_best.pth`: The "physics engine". A Latent 
  Transformer trained to predict the next token state given the current state 
  and an action.

PHASE 5: Imitation Learning & DAgger
- Built an Actor MLP with a 1x1 Convolution spatial head to sit on top of the 
  frozen World Model. 
- Diagnosed and cured "Causal Confusion" (the agent ignoring the screen and 
  just repeating past actions) by explicitly blinding its short-term memory.
- Implemented Dataset Aggregation (DAgger) via `run_dagger.py` to collect 
  recovery data when the agent drifted off the golden path.
- Aggregated the datasets to train robust offline policies.

PHASE 6: Live Interactive Training & The Ablation Study
- Built `live_interactive_dagger.py` to train agents in real-time at 30 FPS.
- Conducted a definitive ablation study comparing three paradigms:
    * Model A: Pre-trained Frozen World Model (Won: Sample efficient, learned in 9 eps).
    * Model B: Unfrozen Scratch Transformer (Lost: Took 2.5x longer, moving target problem).
    * Model C: Random Frozen World Model (Baseline: Proved frozen features aid MLP training).

PHASE 7: Adversarial Testing & The "25-Pixel Shatter"
- Upgraded `visualize_agents.py` with Human-in-the-Loop WASD adversarial overrides.
- Proved Live DAgger training made the agents completely robust to being violently 
  pushed into OOD states.
- Implemented the "Troll Modification" (shifting the level geometry dynamically).
- Discovered the "25-Pixel Shatter" limit: Model A's strict positional embeddings 
  caused it to fail when the geometry crossed a rigid VQ-VAE grid boundary, 
  highlighting the exact trade-offs between rigid World Models and soft end-to-end networks.

=============================================================================
FILE MANIFEST
=============================================================================

Environments & Data:
-------------------
generate_expert_dataset.py  : Contains `ColorfulPlatformerEnv` and offline dataset generator.
expert_dataset/             : The offline "Golden Path" dataset.
dagger_dataset/             : The offline mistake-recovery dataset.

World Model Architecture & Training:
-----------------------------------
* Bouncing Ball
world_model_rect.py

* Jumping Action Ball
train-wm-vqvae.py

* Colorful Retro Platformer
test_env_record.py
train-vq-vae.py             : Script to train the visual tokenizer.
test_vqvae_world.py         : Visualizes the VQ-VAE reconstructions.
capstone_world_model_supervised.py : PyTorch class definitions for VQVAE and LatentWorldModel.

Agent Training & Imitation Learning:
-----------------------------------
generate_expert_dataset.py
live_interactive_dagger.py  : Real-time 30FPS online DAgger training with CSV logging. Features 
                              the Model A/B/C ablation toggles.
visualize_agents.py         : Loads the `.pth` files and runs visual evaluation. Contains the 
                              interactive WASD override and the environment coordinate shifter.

Evaluation & Rendering:
----------------------
training_log_*.csv          : Telemetry logs from the Live DAgger ablation study.
*.mp4                       : Various video outputs of World Model rollouts ("dreaming").