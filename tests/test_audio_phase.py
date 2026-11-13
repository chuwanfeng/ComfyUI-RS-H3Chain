"""音频相位的纯函数测试：`python tests/test_audio_phase.py`（不需要 torch / ComfyUI）。

为什么单独测：相位差算错不会报错，只表现为接缝处音画差 1/40 秒，
而 40Hz 的音频栅格让这个误差在成片里几乎听不出来、却会逐段累积。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_chain.state import (  # noqa: E402
    FRAME_RESCALE,
    audio_guide_index,
    pixel_frames_from_latent_t,
    signed_audio_overhang,
    temporal_shape,
    video_latent_t,
)

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


# ---------------------------------------------------------------- 网格换算

check("FRAME_RESCALE = 5/3", abs(FRAME_RESCALE - 5.0 / 3.0) < 1e-12, str(FRAME_RESCALE))
check("124 帧 = 37 token", video_latent_t(124) == 37)
check("124 帧的时间轴 = 124 帧 / 207 音频步", temporal_shape(124)[0::2] == (124, 207), str(temporal_shape(124)))

# ---------------------------------------------------------------- 带符号 overhang

check("render 124 → +1/3 步", abs(signed_audio_overhang(207, 124) - 1.0 / 3.0) < 1e-9,
      str(signed_audio_overhang(207, 124)))
check("render 141 → 0 步", abs(signed_audio_overhang(235, 141)) < 1e-9,
      str(signed_audio_overhang(235, 141)))
check("render 107 → −1/3 步", abs(signed_audio_overhang(178, 107) + 1.0 / 3.0) < 1e-9,
      str(signed_audio_overhang(178, 107)))
check("render 158 → −1/3 步", abs(signed_audio_overhang(263, 158) + 1.0 / 3.0) < 1e-9,
      str(signed_audio_overhang(263, 158)))

# 人工构造 40Hz 步数（render 124 → 207 步；22 帧尾 → 37 步）
check("22 帧 = 7 token", video_latent_t(22) == 7)
check("22 帧对应的像素帧数 = 22", pixel_frames_from_latent_t(7) == 22)

# ---------------------------------------------------------------- 结束对齐索引

CTX, TAIL = 22, 37  # 22 帧衔接窗 → 37 个音频步

check("overhang +1/3 → index 0.0",
      abs(audio_guide_index(CTX, TAIL, 1.0 / 3.0)) < 1e-9,
      str(audio_guide_index(CTX, TAIL, 1.0 / 3.0)))
check("overhang 0 → index 0.0（吸附到目标音频行）",
      abs(audio_guide_index(CTX, TAIL, 0.0)) < 1e-9,
      str(audio_guide_index(CTX, TAIL, 0.0)))
check("overhang −1/3 → index −0.6",
      abs(audio_guide_index(CTX, TAIL, -1.0 / 3.0) + 0.6) < 1e-9,
      str(audio_guide_index(CTX, TAIL, -1.0 / 3.0)))

# 索引为负是预期行为：它表示音频窗跨过段首、结束在接缝处
check("负索引确实跨过原点",
      audio_guide_index(CTX, TAIL, -1.0 / 3.0) < 0.0)

# 56 帧衔接窗 → 93 步；结束对齐后仍应落在音频行上
CTX56, TAIL56 = 56, 93
idx56 = audio_guide_index(CTX56, TAIL56, 0.0)
check("56 帧衔接窗的索引落在音频行边界",
      abs(FRAME_RESCALE * (idx56 + TAIL56 / FRAME_RESCALE)
          - round(FRAME_RESCALE * (idx56 + TAIL56 / FRAME_RESCALE))) < 1e-9,
      str(idx56))

# 末段换长度时，索引随 overhang 改变（这正是旧实现丢掉的那一半）
check("107 源段的负 overhang 与 124 源段不同",
      abs(audio_guide_index(CTX, TAIL, -1.0 / 3.0)
          - audio_guide_index(CTX, TAIL, 1.0 / 3.0)) > 0.5)

print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)
