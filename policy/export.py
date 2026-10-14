"""Export a trained policy to ONNX (one unrolled graph per loop count) and INT8, with parity checks.

Normalization is baked into the graph: inputs are raw follower joints (LeRobot degrees, gripper
0-100) and raw env features; the output is a (1, chunk, 6) block of raw joint targets.

    python -m policy.export runs/looped --loops 2 4 6 8

INT8 is dynamic quantization of MatMul weights only (ORT's recommendation for transformers);
LayerNorm and Softmax stay FP32. A model is only accepted if INT8 stays within --tol of FP32.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from .data import Normalizer
from .train import LOOPED, MODELS


class Deployable(torch.nn.Module):
    def __init__(self, model, normalizer: Normalizer, loops: int):
        super().__init__()
        self.model, self.loops = model, loops
        for key in ("state", "box", "action"):
            for stat in ("mean", "std"):
                self.register_buffer(f"{key}_{stat}", torch.tensor(normalizer.stats[key][stat]))

    def forward(self, state, env):
        s = (state - self.state_mean) / self.state_std
        e = (env - self.box_mean) / self.box_std
        a = self.model(s, e, self.loops)[-1]
        return a * self.action_std + self.action_mean


def load(run: Path):
    ckpt = torch.load(run / "policy.pt", map_location="cpu")
    model = MODELS[ckpt["model"]](**ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    normalizer = Normalizer(json.loads((run / "normalizer.json").read_text()))
    return model.eval(), normalizer, ckpt


def session(path: Path, threads=1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bench(sess, inputs, n=300) -> float:
    for _ in range(20):
        sess.run(None, inputs)
    t = time.perf_counter()
    for _ in range(n):
        sess.run(None, inputs)
    return (time.perf_counter() - t) / n * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--loops", type=int, nargs="+", default=[6])
    p.add_argument("--tol", type=float, default=0.5, help="max |INT8 - FP32| joint target, LeRobot units")
    args = p.parse_args()

    model, normalizer, ckpt = load(args.run)
    out = args.run / "onnx"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    # probe inputs drawn from the training distribution
    probe = [
        {
            "state": (normalizer.stats["state"]["mean"] + rng.normal(0, 1, 6) * normalizer.stats["state"]["std"])[None].astype(np.float32),
            "env": (normalizer.stats["box"]["mean"] + rng.normal(0, 1, len(normalizer.stats["box"]["mean"])) * normalizer.stats["box"]["std"])[None].astype(np.float32),
        }
        for _ in range(64)
    ]
    loop_counts = args.loops if ckpt["model"] in LOOPED else [1]
    report = {}
    for k in loop_counts:
        dep = Deployable(model, normalizer, k).eval()
        fp32 = out / f"policy_k{k}.onnx"
        torch.onnx.export(
            dep, (torch.from_numpy(probe[0]["state"]), torch.from_numpy(probe[0]["env"])), str(fp32),
            input_names=["state", "env"], output_names=["actions"], opset_version=18, dynamo=True,
        )
        s32 = session(fp32)
        with torch.no_grad():
            ref = [dep(torch.from_numpy(x["state"]), torch.from_numpy(x["env"])).numpy() for x in probe]
        err32 = max(np.abs(s32.run(None, x)[0] - r).max() for x, r in zip(probe, ref))
        report[k] = {
            "fp32_vs_torch": float(err32),
            "fp32_kb": fp32.stat().st_size // 1024 + sum(f.stat().st_size for f in out.glob(f"policy_k{k}.onnx.data")) // 1024,
            "mac_ms_fp32": bench(s32, probe[0]),
        }
        # INT8 is optional: dynamic quantization can fail on dynamo-exported graphs; FP32 is always shipped
        pre, int8 = out / f"policy_k{k}_pre.onnx", out / f"policy_k{k}_int8.onnx"
        try:
            quant_pre_process(str(fp32), str(pre))
            quantize_dynamic(str(pre), str(int8), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
                             per_channel=True, extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
            s8 = session(int8)
            err8 = max(np.abs(s8.run(None, x)[0] - r).max() for x, r in zip(probe, ref))
            report[k].update(int8_vs_torch=float(err8), int8_ok=bool(err8 <= args.tol),
                             int8_kb=int8.stat().st_size // 1024, mac_ms_int8=bench(s8, probe[0]))
        except Exception as e:  # noqa: BLE001
            int8.unlink(missing_ok=True)
            report[k].update(int8_ok=False, int8_error=str(e).splitlines()[0][:120])
        finally:
            pre.unlink(missing_ok=True)
        print(f"K={k}: " + "  ".join(f"{n}={v:.3g}" if isinstance(v, float) else f"{n}={v}" for n, v in report[k].items()))
    (out / "export_report.json").write_text(json.dumps(report, indent=1))
    (out / "normalizer.json").write_text((args.run / "normalizer.json").read_text())


if __name__ == "__main__":
    main()
