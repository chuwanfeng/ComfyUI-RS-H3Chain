"""链状态：路径规划、段几何、meta 守卫。

这个模块**只用标准库**，所以可以在没装 torch / 没装 ComfyUI 的机器上直接跑测试
（`python tests/test_state.py`）。所有与张量相关的部分在 media.py，与 T8 耦合的部分在 t8.py。

为什么把"段几何"写成纯函数：
    段长/窗口/裁切三者一旦算错，成片表现为"接缝处跳一下"或"越跑越短"，
    而且不会报错。这类数学必须能被单元测试钉死，不能埋在一次 ComfyUI 执行里。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

# 状态 schema：改字段语义就 +1（旧 state 会被明确拒绝，而不是被误读）
SCHEMA = 1

FPS = 24
AUDIO_LATENT_FPS = 40

# H3 视频 latent 的 T 轴不均匀：token k 覆盖 FRAME_PER_TOKEN[k % 5] 帧，5 token = 17 帧。
# 段长必须是 17k+5、上下文窗口必须是 17m+5，两者相减才是 17 的倍数 ⇒ 尾窗起点落在 token 边界。
# 这就是"相位自动对齐"的全部来源（JZL AssetManagerInfinite 的注释里同样写明）。
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
CONTEXT_FRAME_STEPS = {5: 2, 22: 7, 39: 12}
MAX_CONTEXT_FRAMES = max(CONTEXT_FRAME_STEPS)

# 渲染窗口候选：全部是 17k+5
RENDER_WINDOW_CHOICES = (56, 90, 124, 158, 192, 226, 260, 294)
CONTEXT_FRAME_CHOICES = tuple(sorted(CONTEXT_FRAME_STEPS))

STATE_SUBDIR = "rs_h3_chain"
CONTEXT_FILE = "tail.context.safetensors"
META_FILE = "meta.json"


class ChainError(RuntimeError):
    """链状态相关的可读错误（消息直接给用户看，不要包成 traceback 丢出去）。"""


# ---------------------------------------------------------------- 路径


def sanitize_name(raw: str) -> str:
    """链名 → 安全的目录名。空值给默认名，其余非法字符折成下划线。"""
    text = str(raw or "").strip()
    text = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", text).strip("._-")
    return text or "chain"


def resolve_chain_root(chain_dir: str, output_root: str | os.PathLike[str]) -> Path:
    """把用户给的 `chain_dir` 解析成绝对目录。

    相对路径落在 `<ComfyUI output>/rs_h3_chain/<name>`；绝对路径原样使用（但要确保不是文件）。
    这里不做"禁止绝对路径"的限制——把链放到大盘上是合理需求，只要求它是个目录。
    """
    raw = str(chain_dir or "").strip()
    if not raw:
        raise ChainError("chain_dir 不能为空：给它一个名字（如 mv-01）或一个绝对目录")
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return Path(output_root) / STATE_SUBDIR / sanitize_name(raw)


def segment_dir(root: Path, index: int) -> Path:
    return Path(root) / f"seg_{int(index):05d}"


def context_path(root: Path, index: int) -> Path:
    return segment_dir(root, index) / CONTEXT_FILE


def meta_path(root: Path, index: int) -> Path:
    return segment_dir(root, index) / META_FILE


def list_segments(root: Path) -> list[int]:
    """已完成的段号（有 meta + context 才算完成，避免半截状态被当成可用）。"""
    out: list[int] = []
    if not Path(root).is_dir():
        return out
    for child in sorted(Path(root).iterdir()):
        m = re.fullmatch(r"seg_(\d{5})", child.name)
        if not m or not child.is_dir():
            continue
        idx = int(m.group(1))
        if (child / META_FILE).is_file() and (child / CONTEXT_FILE).is_file():
            out.append(idx)
    return out


# ---------------------------------------------------------------- 段几何


def align_frame_count(n: int) -> int:
    """吸附到 17k+5 网格（只增不减，与官方 EmptyMiniMaxH3LatentAV 同款）。"""
    value = max(5, int(n))
    while value % 17 != 5:
        value += 1
    return value


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int) -> tuple[int, int, int]:
    """帧数 → (实际帧数, 视频 latent T, 音频 latent T)。"""
    frame_count = align_frame_count(length)
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def pixel_frames_from_latent_t(latent_t: int) -> int:
    """latent 的 T 轴 → 像素帧数（token 覆盖帧数不均匀，不能乘 4）。"""
    t = int(latent_t)
    if t < 1:
        raise ChainError(f"latent T={t} 非法")
    return sum(FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(t))


def check_render_frames(render_frames: int) -> int:
    value = int(render_frames)
    if value < 22:
        raise ChainError(f"render_window_frames={value} 太小（至少 22 帧，且必须是 17k+5）")
    if value % 17 != 5:
        raise ChainError(
            f"render_window_frames={value} 不是 17k+5：H3 的 latent T 轴每 5 token = 17 帧，"
            f"段长不在网格上会让尾窗起点落进 token 内部（接缝周期性重影，且不报错）"
        )
    return value


def check_context_frames(context_frames: int, render_frames: int) -> int:
    value = int(context_frames)
    if value not in CONTEXT_FRAME_STEPS:
        raise ChainError(
            f"context_frames={value} 不是支持的窗口：{sorted(CONTEXT_FRAME_STEPS)}"
            f"（对应 {CONTEXT_FRAME_STEPS} 个 latent 步）"
        )
    if value >= int(render_frames):
        raise ChainError(
            f"context_frames={value} 必须小于 render_window_frames={render_frames}，否则本段没有新内容可生成"
        )
    return value


def plan_segment(render_frames: int, context_frames: int, index: int,
                 timeline_start_frames: int | None = None) -> dict:
    """单段的时间轴与裁切量。

    第 1 段：时间轴 0 起，不裁头，产出 render_frames 帧。
    第 k 段：段首 context 帧是被钉住的上一段尾巴，解码后必须裁掉，
             所以产出 render − context 帧。

    `timeline_start_frames` 是渲染窗口的起点（= 被钉住的衔接帧的起点）。
    不传就按「所有段同窗口」的公式算：`(k-1) × (render − context)`。
    末段为了对准目标时长可以短一点（124 → 141/107…），这时**必须**由调用方
    从上一段的 meta 推出真实起点 —— 用固定步长会把后面所有段的绝对时间、
    音频窗口、关键帧位置一起算错，而且没有任何报错。
    """
    render = check_render_frames(render_frames)
    context = check_context_frames(context_frames, render)
    idx = int(index)
    if idx < 1:
        raise ChainError(f"segment_index 从 1 开始，收到 {idx}")
    advance = render - context
    if idx == 1:
        start = 0
    elif timeline_start_frames is None:
        start = (idx - 1) * advance
    else:
        start = int(timeline_start_frames)
        if start < 0:
            raise ChainError(f"timeline_start_frames={start} 不能为负")
        if start % 17 != 0:
            # 窗口起点 = 前几段 (render − context) 之和，必然是 17 的倍数
            raise ChainError(
                f"timeline_start_frames={start} 不是 17 的倍数："
                f"上一段的 render/context 可能被改过，检查上一条 meta.json"
            )
    return {
        "index": idx,
        "render_frames": render,
        "context_frames": context,
        "advance_frames": advance,
        "timeline_start_frames": start,
        "timeline_start_seconds": start / FPS,
        "head_trim": context if idx > 1 else 0,
        "output_frames": render - (context if idx > 1 else 0),
        "is_continuation": idx > 1,
    }


def chain_timeline_start(root: Path, index: int, context_frames: int,
                         render_frames: int) -> int:
    """第 `index` 段的渲染窗口起点：上一段起点 + 上一段窗口 − 衔接帧。

    上一段的 meta 读不到时退化成「所有段同窗口」的固定步长公式
    （只有 `allow_no_prev` 那条无上下文的路会走到这里）。
    """
    idx = int(index)
    if idx <= 1:
        return 0
    prev_meta = meta_path(Path(root), idx - 1)
    if prev_meta.is_file():
        prev = read_json(prev_meta)
        prev_render = int(prev.get("render_frames") or 0)
        if prev_render > 0:
            prev_start = prev.get("timeline_start_frames")
            if prev_start is None:
                prev_context = int(prev.get("context_frames") or context_frames)
                prev_start = (idx - 2) * (prev_render - prev_context)
            return int(prev_start) + prev_render - int(context_frames)
    return (idx - 1) * (int(render_frames) - int(context_frames))


def keyframe_local_frame(time_seconds: float, plan: dict) -> int:
    """成片时间轴（秒）→ 本段渲染窗口内的帧号。

    用绝对秒而不是"本段第几帧"，是因为用户说的是"第 8 秒放这张图"；
    段的时间轴起点由段几何固定（(k-1)×(W-C)），不随内容变化，所以绝对秒是稳定的。
    返回值可能落在段首衔接窗口内（< head_trim）——那是调用方要拒绝的情况，不是这里修。
    """
    return round(float(time_seconds) * FPS) - int(plan["timeline_start_frames"])


def audio_window_start_seconds(plan: dict) -> float:
    """本段要喂给模型的音频起点 = **渲染窗口起点**。

    续接段的前 context 帧（上一段尾巴）已经落在渲染窗口里：窗口是
    `[timeline_start, timeline_start + render)`，那 22 帧钉在窗口最前面，
    解码后由 `head_trim` 裁掉。所以音频也从 `timeline_start` 取、同样裁这么多。

    曾经在这里又减了一个 context（"多取一个衔接窗口"）—— 结果是本段音频整体
    往前错了 22/24 = 0.917 秒：实测第 2 段交付音频的起点是 4.25s，而画面起点是
    5.17s，接缝处听着像"音乐跳回去一下"，末尾还少 0.917s。别再加回来。
    """
    return max(0.0, plan["timeline_start_frames"] / FPS)


# ---------------------------------------------------------------- meta


def prompt_sha(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def make_meta(*, chain_id: str, plan: dict, width: int, height: int,
              steps: int, sampler: str, scheduler: str,
              shift_video: float, shift_audio: float, task_type: str,
              audio_mode: str, prompt: str, seed: int) -> dict:
    return {
        "schema": SCHEMA,
        "chain_id": chain_id,
        "segment_index": plan["index"],
        "render_frames": plan["render_frames"],
        "context_frames": plan["context_frames"],
        "advance_frames": plan["advance_frames"],
        "timeline_start_frames": plan["timeline_start_frames"],
        "head_trim": plan["head_trim"],
        "output_frames": plan["output_frames"],
        "width": int(width),
        "height": int(height),
        "fps": FPS,
        "steps": int(steps),
        "sampler": str(sampler),
        "scheduler": str(scheduler),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "task_type": str(task_type),
        "audio_mode": str(audio_mode),
        "prompt_sha256": prompt_sha(prompt),
        "prompt_chars": len(str(prompt or "")),
        "seed": int(seed),
        "created_unix": int(time.time()),
    }


# 这些字段必须整条链一致：它们决定 latent 的坐标系，改一个旧上下文就"接不上"了。
_CHAIN_WIDE_FIELDS = (
    ("schema", "状态 schema"),
    ("context_frames", "衔接窗口帧数"),
    ("width", "宽度"),
    ("height", "高度"),
    ("fps", "帧率"),
)

# 注意 `render_frames` **不在这里**：段长都是 17k+5、衔接窗口都是 17m+5，
# 两者的差值必为 17 的倍数，所以「上一段的尾巴」在下一段的坐标系里仍然落在
# token 边界上 —— 末段短一点（对准目标时长）是安全的。
# 需要守的是时间轴连续：上一段起点 + 上一段窗口 − 衔接帧 = 本段起点。


def meta_issues(prev: dict, cur: dict) -> list[str]:
    """上一段与当前段的 meta 差异（空 = 可以接）。

    只比"坐标系"字段：提示词、种子、步数可以逐段变，画幅/窗口不能变——
    变了以后上一段的 latent 尾巴在本次的条件空间里没有意义，接上去只会静默错位。
    """
    issues: list[str] = []
    if not isinstance(prev, dict):
        return ["上一段的 meta.json 读不出来（不是 JSON 对象）"]
    for key, label in _CHAIN_WIDE_FIELDS:
        if prev.get(key) is None:
            continue
        if prev.get(key) != cur.get(key):
            issues.append(f"{label}（{key}）不一致：上一段 {prev.get(key)}，本段 {cur.get(key)}")
    if int(prev.get("segment_index", 0)) != int(cur.get("segment_index", 0)) - 1:
        issues.append(
            f"上一段号不是 {int(cur.get('segment_index', 0)) - 1}"
            f"（meta 里写的是 {prev.get('segment_index')}）——段目录与 meta 对不上，可能被手工改过"
        )
    # 时间轴必须首尾相接：上一段起点 + 上一段窗口 − 衔接帧 = 本段起点。
    # 这条比「窗口必须相同」更本质：末段允许短一点，但不许出现空档/重叠。
    prev_start = prev.get("timeline_start_frames")
    prev_render = prev.get("render_frames")
    if prev_start is not None and prev_render is not None:
        expect = int(prev_start) + int(prev_render) - int(cur.get("context_frames") or 0)
        if int(cur.get("timeline_start_frames") or 0) != expect:
            issues.append(
                f"时间轴不连续：上一段 {prev_start}+{prev_render}−{cur.get('context_frames')}"
                f" 应为 {expect}，本段却是 {cur.get('timeline_start_frames')}"
                "（接缝处会重复或丢帧）"
            )
    return issues


# ---------------------------------------------------------------- 读写


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ChainError(f"找不到文件：{path}")
    except json.JSONDecodeError as exc:
        raise ChainError(f"{path} 不是合法 JSON：{exc}") from exc


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def clear_segment(root: Path, index: int) -> bool:
    """删掉某一段的状态（重跑前用）。返回是否真的删了东西。"""
    d = segment_dir(root, index)
    if not d.is_dir():
        return False
    for child in d.iterdir():
        if child.is_file():
            child.unlink()
    try:
        d.rmdir()
    except OSError:
        pass
    return True
