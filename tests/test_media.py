"""张量侧的小测试：`python tests/test_media.py`（需要 torch；没有则跳过）。

覆盖两件不会报错、只会让成片看起来「差一点」的事：
  · 接缝亮度过渡的幅度与边界；
  · 上下文文件的哈希/末帧亮度能否原样读回（血缘校验依赖它）。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"SKIP  没有 torch（{exc}）")
    sys.exit(0)

from rs_chain import media  # noqa: E402
from rs_chain.state import context_path  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def solid(value: float, frames: int = 12):
    return torch.full((frames, 8, 8, 3), value, dtype=torch.float32)


# ---------------------------------------------------------------- 亮度过渡

flat = solid(0.20)
same, info = media.luma_transition(flat, 0.20)
check("亮度差小于阈值 → 逐像素不变", info["applied"] is False and torch.equal(same, flat), str(info))

raised, info = media.luma_transition(flat, 0.60)
check("上一段更亮 → 应用过渡", info["applied"] is True and info["max_delta"] > 0, str(info))
check("单帧幅度不超过 0.10", info["max_delta"] <= 0.10 + 1e-9, str(info["max_delta"]))
check("首帧被拉向上一段亮度",
      media.frame_luma(raised[0]) > media.frame_luma(flat[0]) + 0.05,
      f"{media.frame_luma(raised[0]):.4f}")
check("末帧残留误差 < 0.05（过渡不硬切）",
      abs(media.frame_luma(raised[11]) - media.frame_luma(flat[11])) < 0.05,
      f"{media.frame_luma(raised[11]):.4f}")
check("形状与 dtype 不变", raised.shape == flat.shape and raised.dtype == flat.dtype)

# 亮度过渡只改亮度：同一帧的通道差不应被改变
check("通道差保持不变（不改色相）",
      torch.allclose((raised[0] - raised[0].mean()).float(),
                     (flat[0] - flat[0].mean()).float(), atol=1e-5))

# ---------------------------------------------------------------- 上下文读写

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    video = torch.zeros((1, 24, 7, 8, 8), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 37), dtype=torch.float32)
    info_written = media.write_context(
        context_path(root, 1), video, audio,
        chain_id="mv", segment_index=0, context_frames=22,
        audio_overhang=1.0 / 3.0, source_frames=124, last_luma=0.4321,
    )
    back = media.read_context(
        context_path(root, 1), chain_id="mv", source_index=0, context_frames=22
    )
    check("写读哈希一致", back["sha256"] == info_written["context_sha256"])
    check("末帧亮度原样读回", abs((back["meta"]["last_luma"] or 0) - 0.4321) < 1e-6,
          str(back["meta"]["last_luma"]))
    check("overhang 原样读回", abs(back["meta"]["audio_overhang"] - 1.0 / 3.0) < 1e-6)
    check("只读摘要与完整读回一致",
          media.context_digest(context_path(root, 1)) == back["sha256"])

    # 用同样的 metadata 重存一份不同内容的张量：读回必须发现哈希不符
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file

        target = context_path(root, 1)
        with safe_open(str(target), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        forged = target.with_suffix(".forged")
        save_file(
            {"video_tail": torch.ones_like(video), "audio_tail": audio},
            str(forged), metadata,
        )
        forged.replace(target)
        try:
            media.read_context(target, chain_id="mv", source_index=0, context_frames=22)
            check("张量被替换但 metadata 未变 → 拒绝", False, "没有报错")
        except Exception:
            check("张量被替换但 metadata 未变 → 拒绝", True)
    except OSError as exc:
        print(f"SKIP  篡改用例（文件被占用：{exc}）")

print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)
