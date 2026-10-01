# Deep Reinforcement Learning: Pong DQN

A from-scratch Deep Q-Network (DQN) implementation for the Gymnasium Atari environment `ALE/Pong-v5`.

This repository isolates the Pong portion of the reinforcement-learning assignment and includes the training implementation, an environment smoke test, and the resulting learning curve.

## Contents

- `Pong-v5.py` - CNN-based DQN trainer with replay memory, epsilon-greedy exploration, a target network, checkpoint/resume support, and learning-curve generation.
- `Pong-v5_1(a).py` - Minimal environment check that prints the observation/action spaces and steps through one random episode.
- `dqn_pong_learning_curve.png` - Learning curve produced by the training script.
- `requirements.txt` - Python dependencies used by the scripts.

## Algorithm

The agent learns an action-value function $Q(s,a)$ with two convolutional networks:

- **Policy network:** updated by gradient descent from sampled replay batches.
- **Target network:** periodically synchronized with the policy network to stabilize temporal-difference targets.

For each transition, the target is computed as:

```text
r + gamma * max_a' Q_target(s', a') * (1 - done)
```

The trainer uses the Smooth L1 (Huber) loss, Adam optimization, replay memory, gradient clipping, and exponentially decaying epsilon-greedy exploration.

## Observation Processing

Raw Atari frames are converted to grayscale, resized to `84 x 84`, and stacked into four consecutive frames. The network receives tensors with shape `4 x 84 x 84`.

The implementation restricts the environment to three mapped actions:

| Internal action | Pong action |
| ---: | ---: |
| 0 | 0 |
| 1 | 2 |
| 2 | 3 |

## Main Training Configuration

The current script uses:

- Environment: `ALE/Pong-v5` with `frameskip=4`
- Batch size: `32`
- Discount factor: `0.99`
- Epsilon: `1.0` to `0.1`
- Epsilon decay: `1,000,000` steps
- Replay capacity: `50,000`
- Replay warm-up: `10,000` transitions
- Target update interval: `10,000` steps
- Learning rate: `0.0001`
- Maximum steps per episode: `10,000`
- Training episodes: `1,900`

Training runs on CPU by default. Atari training can take a substantial amount of time.

## Setup

Use Python 3.10 or newer and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Atari extras install the ROM support needed by `ALE/Pong-v5`:

```bash
python -m pip install "gymnasium[atari,accept-rom-license]"
```

## Run the Environment Check

```bash
python 'Pong-v5_1(a).py'
```

This performs a random-action episode and verifies that Gymnasium and the Atari environment are available.

## Train the DQN

```bash
python Pong-v5.py
```

The trainer writes `checkpoint_pong.pth` periodically, allowing an interrupted run to resume automatically. At the end it writes `dqn_pong_learning_curve.png`.

Checkpoint and generated output files are ignored by Git because they can be large and are machine-specific.

## Notes

- The code uses the Gymnasium API, where `step()` returns `observation, reward, terminated, truncated, info`.
- The saved learning curve is an existing result from the assignment and is included for reference.
- The source scripts retain the original assignment structure and naming.
