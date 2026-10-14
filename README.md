# EdgeAI Challenge: Looped Arm Policy

An SO-101 robot arm picks up an Arduino UNO Q box from a table and hands it to a person. An overhead iPhone camera finds the box, and a small learned policy turns the box pose and the arm's joint angles into the next second of joint targets. The policy is small enough to run on the UNO Q board itself.

## Main idea: looping

The policy is built on the idea of looped transformers ([Yang et al.](https://arxiv.org/abs/2311.12424), [Geiping et al.](https://arxiv.org/abs/2502.05171)). Instead of stacking many different layers, the model has one block that it applies K times. The input is added back in before each pass, and the same weights are reused every time.

This has two effects:

- **Fewer parameters.** Six stacked transformer blocks have about 1.2M parameters. One looped block applied six times has about 206k, with the same accuracy.
- **Compute is a dial.** K is picked at run time. Fewer loops are faster and more loops are more accurate, with no retraining. During training, K is sampled at random for each batch so every K gives a usable policy.

We first built a looped transformer, then compared it against a stacked transformer, a plain MLP, and a looped MLP. For this input (6 joint angles plus a 4-number box pose) the looped MLP did best. Its error also stays flat past the trained range of K, so it is safe to turn the dial up on the board. The deployed model is one residual MLP block looped K times.

## Built from scratch

Nothing here starts from a pretrained model or an existing dataset.

- The model is defined in plain PyTorch and trained from random weights.
- The training data was generated in our own Isaac Sim scene of the arm and box, plus demonstrations we recorded by hand on the real arm.
- Box detection is plain OpenCV (thresholding against the dark tablecloth), with no learned vision model.
- The calibration between camera, table, simulator and real joints was measured by hand.

## Metrics

| Metric | Value |
|---|---|
| Parameters | 156,852 |
| Model file (ONNX, FP32) | 666–697 KB for K = 4–8 |
| Time per prediction (Mac CPU, 1 thread) | 0.03–0.06 ms for K = 4–8 |
| Error on held-out sim episodes, K = 8 | 2.53° |
| Error on held-out real episodes, sim training only | 15.7° |
| Error on held-out real episodes, after fine-tuning on real demos | 6.2° |

- **Error** is the average difference between the predicted and recorded joint targets over a 30-step chunk, in degrees. The gripper is on a 0–100 scale.
- **Loops vs. error:** error falls from 3.14° at K = 1 to 2.53° at K = 7, then stays roughly flat up to K = 24.
- **Not measured yet:** on-board latency, RAM and energy on the UNO Q. INT8 quantization worked for the looped transformer (317 KB) but does not load for the looped MLP yet.

## How it was built

1. **Wrote a shared contract** ([CONTRACT.md](CONTRACT.md)). It fixes joint order and units, the table coordinate frame, the box features `[x, y, cos 2θ, sin 2θ]` and the 30 Hz timing, so real data, sim data and the runtime all match.
2. **Set up the arms.** A leader/follower pair of SO-101 arms runs through LeRobot for calibration and teleoperation ([scripts/](scripts/)).
3. **Calibrated the camera to the table.** Four taped crosses at known positions give a homography from iPhone pixels to meters in the robot's base frame ([table_calib.json](table_calib.json)). The box was measured from a rectified photo: 89 × 66 × 25 mm.
4. **Wrote box detection** ([policy/perception.py](policy/perception.py)). The same code labels recorded demos and runs live, so training and deployment see the same features.
5. **Built the simulator task** ([isaac_sim/](isaac_sim/)). Isaac Sim ran on a rented GPU. A search over 8,192 random grasps found what works: top-down, across the 66 mm side, with a 0.30 N·m gripper limit.
6. **Generated sim demos.** Scripted pick demos with noise added to the executed actions produced 13,929 successful episodes across two datasets.
7. **Mapped sim joints to real joints** ([sim_bridge/](sim_bridge/)). Holding the real arm in known poses gave a sign and offset for each joint, so sim episodes convert into real-arm units.
8. **Compared architectures** ([experiments/arch_sweep.py](experiments/arch_sweep.py)). Looped transformer, stacked transformer, MLP and looped MLP were trained with the same recipe and compared on error vs. size.
9. **Trained the looped MLP on sim data** ([experiments/train_main.py](experiments/train_main.py)) for 20,000 steps on the GPU.
10. **Recorded real demos** through the demo UI ([scripts/record_ui.py](scripts/record_ui.py)). Each demo is cut 0.5 s after the grasp is detected.
11. **Fine-tuned on real demos** ([policy/finetune.py](policy/finetune.py)). Each batch is 75% real and 25% sim, so the model keeps what it learned in sim about the workspace.
12. **Exported to ONNX** ([policy/export.py](policy/export.py)). There is one unrolled graph per K, with normalization built in and an output check against PyTorch. The files are pushed to the UNO Q over adb.
13. **Ran it on the real arm** ([scripts/run_policy.py](scripts/run_policy.py)). The policy does the pick. A stalled gripper signals the grasp, then a handover motion learned from the demos ([policy/handover_learned.py](policy/handover_learned.py)) brings the box to the person, and the gripper releases when they pull on it.

[build_policy.sh](experiments/build_policy.sh) runs steps 7–12 (convert, train, export, push) in one command.

## Layout

| Path | Contents |
|---|---|
| `policy/` | Model, training, fine-tuning, ONNX export, perception, handover |
| `experiments/` | Architecture sweep, GPU training, one-command build |
| `isaac_sim/` | Isaac Sim task and demo generator |
| `sim_bridge/` | Sim-to-real joint map, calibration, data conversion |
| `scripts/` | Teleop, recording, calibration, running the policy |
| `demo_ui/` | Live demo screen (Python bridge + React app) and scorecard |
| `runs/` | Trained checkpoints, metrics and exported ONNX models |

The recorded datasets (`datasets/`) and generated sim episodes (`sim_data/`) are about 1 GB and are not in the repo.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Running the demo UI without hardware

```
cd demo_ui/web && pnpm install && pnpm build && cd ../..
.venv/bin/python -m demo_ui.bridge --replay --on-demand
open http://localhost:8765/#demo
```

Replay mode needs the sim episodes in `sim_data/`.
