"""张量侧的小工具：音频取窗、裁头、上下文文件读写。

只有这个模块 import torch / safetensors。它不依赖 T8，也不依赖 JZL——
段间上下文是我们自己的格式（`video_tail` + `audio_tail` + 校验和），
与官方 `minimax_keyframes` 的 `latent` / `audio_latent` 直接对接。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from .state import FPS, ChainError

# 上下文文件自己的 schema：改字段语义就 +1，旧文件会被明确拒绝
CONTEXT_SCHEMA = 1


# ---------------------------------------------------------------- 音频


def waveform_of(audio) -> tuple[torch.Tensor, int]:
    """ComfyUI 的 AUDIO 字典 → (waveform [B,C,T], sample_rate)。"""
    if audio is None:
        raise ChainError("audio 为空")
    if not isinstance(audio, dict):
        raise ChainError(f"audio 结构不认识：{type(audio).__name__}（期望 {{'waveform','sample_rate'}}）")
    wave = audio.get("waveform")
    rate = audio.get("sample_rate")
    if wave is None or rate is None:
        raise ChainError("audio 缺少 waveform / sample_rate")
    wave = wave if torch.is_tensor(wave) else torch.as_tensor(wave)
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    if wave.ndim != 3:
        raise ChainError(f"audio waveform 形状应为 [B,C,T]，实际 {tuple(wave.shape)}")
    return wave, int(rate)


def window_audio(audio, start_seconds: float, frames: int, fps: int = FPS):
    """取本段音频：从 `start_seconds` 起、长度 = frames/fps 秒。

    越界补零、超出截断，**长度恒定**——长度和视频帧数不匹配时，mux 出来的成片音画不同步，
    而且不会报错。
    """
    if audio is None:
        return None
    wave, rate = waveform_of(audio)
    render_samples = round(int(frames) / float(fps) * rate)
    start = round(float(start_seconds) * rate)
    source_start = max(0, start)
    target_start = max(0, -start)
    out = wave.new_zeros((wave.shape[0], wave.shape[1], render_samples))
    writable = min(
        max(0, int(wave.shape[-1]) - source_start),
        max(0, render_samples - target_start),
    )
    if writable:
        out[..., target_start:target_start + writable] = wave[..., source_start:source_start + writable]
    return {"waveform": out, "sample_rate": rate}


def trim_audio(audio, drop_frames: int, fps: int = FPS):
    """按视频实际保留的起始帧数裁音频（音画严格等长）。"""
    if audio is None or int(drop_frames) <= 0:
        return audio
    wave, rate = waveform_of(audio)
    drop = round(int(drop_frames) / float(fps) * rate)
    if drop <= 0:
        return audio
    if drop >= wave.shape[-1]:
        raise ChainError(
            f"音频裁切量 {drop} 样本 ≥ 总长 {wave.shape[-1]}：几乎把整段音频裁空，请检查帧数/帧率"
        )
    return {"waveform": wave[..., drop:], "sample_rate": rate}


def ensure_audio_rate(audio, target_rate: int = 48000):
    """把交付音轨统一到 AAC 友好的采样率。

    为什么需要：H3 模型音轨是 32kHz；部分 PyAV/FFmpeg 构建在
    `add_stream("aac", rate=32000)` 时 `avcodec_open2` 直接返回 EINVAL。
    44.1/48k 原样保留，其余线性重采样到 48k（不改音高/时长）。
    """
    if audio is None:
        return None
    wave, rate = waveform_of(audio)
    if int(rate) == int(target_rate) or int(rate) in (44100, 48000):
        return audio
    n = max(1, round(wave.shape[-1] * float(target_rate) / float(rate)))
    out = torch.nn.functional.interpolate(
        wave.float(), size=n, mode="linear", align_corners=False
    )
    return {"waveform": out, "sample_rate": int(target_rate)}


# ---------------------------------------------------------------- 视频


def trim_frames(frames: torch.Tensor, drop: int) -> torch.Tensor:
    if int(drop) <= 0:
        return frames
    if int(drop) >= int(frames.shape[0]):
        raise ChainError(
            f"裁头 {drop} 帧 ≥ 解码出的 {frames.shape[0]} 帧：段长不足以容纳衔接窗口，请调大窗口或调小上下文"
        )
    return frames[int(drop):]


def frame_luma(frame: torch.Tensor) -> float:
    """单帧亮度（Rec.709 加权，[H,W,C] 或 [1,H,W,C]）。"""
    rgb = frame[..., :3].float()
    if rgb.ndim == 4:
        rgb = rgb[0]
    return float(
        (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).mean().item()
    )


def luma_transition(frames: torch.Tensor, prev_luma: float,
                    frames_n: int = 12, max_delta: float = 0.10,
                    eps: float = 0.008):
    """接缝首帧的**全局加性亮度**过渡（不复制轮廓、不压局部对比）。

    依据：官方 Director 实测只保留这一种接缝亮度做法——每帧给 RGB 加同一个 delta，
    单帧限幅，逐帧从上一段末帧亮度过渡到本段自身亮度。
    只改亮度、不改色相；三项差值都小于阈值时逐像素直通。

    返回 `(frames, info)`，`info` 会原样进 report 与 meta。
    """
    n = min(int(frames_n), int(frames.shape[0]))
    first_luma = frame_luma(frames[0]) if int(frames.shape[0]) else 0.0
    prev_value = None if prev_luma is None else float(prev_luma)
    info = {
        "applied": False,
        "prev_luma": None if prev_value is None else round(prev_value, 6),
        "first_luma": round(first_luma, 6),
        "frames": n,
        "max_delta": 0.0,
    }
    if n < 1 or prev_value is None:
        return frames, info

    y0 = first_luma
    y1 = frame_luma(frames[1]) if n > 1 else y0
    y2 = frame_luma(frames[2]) if n > 2 else y1
    y_end = frame_luma(frames[n - 1])
    gap = abs(y0 - prev_value)
    spike = abs(y1 - y0) + abs(y2 - y1)
    if gap < eps and spike < eps and abs(y_end - prev_value) < eps:
        return frames, info

    out = frames.clone()
    applied_max = 0.0
    for i in range(n):
        t = float(i) / float(n)
        target = prev_value * (1.0 - t) + y_end * t
        current = frame_luma(out[i])
        delta = max(-float(max_delta), min(float(max_delta), target - current))
        if abs(delta) < 1e-5:
            continue
        out[i] = (out[i].float() + delta).clamp(0.0, 1.0).to(dtype=frames.dtype)
        applied_max = max(applied_max, abs(delta))

    info["applied"] = applied_max > 0.0
    info["max_delta"] = round(applied_max, 6)
    return out, info


def image_fingerprint(image: torch.Tensor) -> str:
    """参考图指纹（取第一帧，12 位十六进制）。

    用途：链维护时能看出"哪几段用了同一个人、第几段换了人物"。
    不做校验、不做比较——换人物是允许的，但必须看得见。
    """
    t = image
    if t.ndim == 4:
        t = t[:1]
    return tensor_sha256(t.contiguous())[:12]


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------- 上下文文件


def _combined_digest(video_sha: str, audio_sha: str) -> str:
    """上下文文件的组合哈希：视频尾 + 音频尾。"""
    return hashlib.sha256(f"{video_sha}:{audio_sha}".encode("ascii")).hexdigest()


def write_context(
    path: Path,
    video_tail: torch.Tensor,
    audio_tail: torch.Tensor,
    *,
    chain_id: str,
    segment_index: int,
    context_frames: int,
    audio_overhang: float,
    source_frames: int,
    last_luma: float | None = None,
) -> dict:
    """把本段的衔接窗口落盘（video_tail / audio_tail + 校验和）。

    `segment_index` 是 **0 起**（与官方 keyframe 的段号语义一致）；`meta.json` 里是 1 起。
    `last_luma` 是本段交付最后一帧的亮度，供下一段做接缝亮度过渡（旧文件可为空）。
    """
    if video_tail is None or audio_tail is None:
        raise ChainError("上下文尾巴为空：采样结果里没有 video/audio latent")
    video_sha = tensor_sha256(video_tail)
    audio_sha = tensor_sha256(audio_tail)
    digest = _combined_digest(video_sha, audio_sha)
    metadata = {
        "schema": str(CONTEXT_SCHEMA),
        "chain_id": str(chain_id),
        "source_segment_index": str(int(segment_index)),
        "context_frames": str(int(context_frames)),
        "fps": str(FPS),
        "source_frames": str(int(source_frames)),
        "audio_overhang": repr(float(audio_overhang)),
        "video_shape": json.dumps(list(video_tail.shape)),
        "audio_shape": json.dumps(list(audio_tail.shape)),
        "video_dtype": str(video_tail.dtype),
        "audio_dtype": str(audio_tail.dtype),
        "video_sha256": video_sha,
        "audio_sha256": audio_sha,
        "context_sha256": digest,
        "last_luma": "" if last_luma is None else repr(float(last_luma)),
    }

    from safetensors.torch import save_file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        save_file({"video_tail": video_tail, "audio_tail": audio_tail}, str(tmp), metadata)
        with tmp.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "video_shape": list(video_tail.shape),
        "audio_shape": list(audio_tail.shape),
        "context_sha256": digest,
        "last_luma": last_luma,
    }


def read_context(
    path: Path,
    *,
    chain_id: str,
    source_index: int,
    context_frames: int,
) -> dict:
    """读上一段的衔接窗口。

    校验项：schema / 链名 / 段号 / 上下文帧数 / 形状 / dtype / 校验和 / 有限值。
    少校验一项的后果不是报错，而是拿一段"形状对、内容错"的 latent 去接——画面上表现为人变了。
    """
    from safetensors import safe_open

    target = Path(path)
    if not target.is_file():
        raise ChainError(f"缺少衔接上下文：{target}")
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
        keys = set(handle.keys())
        if keys != {"video_tail", "audio_tail"}:
            raise ChainError(f"{target} 的张量键应为 video_tail/audio_tail，实际 {sorted(keys)}")
        video_tail = handle.get_tensor("video_tail")
        audio_tail = handle.get_tensor("audio_tail")

    if int(meta.get("schema", -1)) != CONTEXT_SCHEMA:
        raise ChainError(f"{target} 的 schema 不是 {CONTEXT_SCHEMA}（重跑该段或换新链）")
    if meta.get("chain_id") != str(chain_id):
        raise ChainError(f"{target} 属于链「{meta.get('chain_id')}」，不是「{chain_id}」")
    if int(meta.get("source_segment_index", -1)) != int(source_index):
        raise ChainError(f"{target} 的源段号是 {meta.get('source_segment_index')}，期望 {source_index}")
    if int(meta.get("context_frames", -1)) != int(context_frames):
        raise ChainError(
            f"{target} 存的是 {meta.get('context_frames')} 帧上下文，本段要 {context_frames} 帧："
            f"整条链的衔接窗口必须一致"
        )
    if json.loads(meta.get("video_shape", "[]")) != list(video_tail.shape):
        raise ChainError(f"{target} 的 video 形状与 metadata 不符")
    if json.loads(meta.get("audio_shape", "[]")) != list(audio_tail.shape):
        raise ChainError(f"{target} 的 audio 形状与 metadata 不符")
    video_sha = tensor_sha256(video_tail)
    audio_sha = tensor_sha256(audio_tail)
    if meta.get("video_sha256") and meta["video_sha256"] != video_sha:
        raise ChainError(f"{target} 的 video 校验和不符（文件被改过或写坏了）")
    if meta.get("audio_sha256") and meta["audio_sha256"] != audio_sha:
        raise ChainError(f"{target} 的 audio 校验和不符（文件被改过或写坏了）")
    if not torch.isfinite(video_tail).all() or not torch.isfinite(audio_tail).all():
        raise ChainError(f"{target} 含 NaN/Inf（采样发散），不能作为下一段的引导")

    last_luma = None
    raw_luma = meta.get("last_luma", "")
    if raw_luma not in ("", None):
        try:
            last_luma = float(raw_luma)
        except (TypeError, ValueError):
            last_luma = None

    return {
        "video_tail": video_tail,
        "audio_tail": audio_tail,
        "sha256": _combined_digest(video_sha, audio_sha),
        "meta": {
            "chain_id": str(chain_id),
            "source_segment_index": int(source_index),
            "context_frames": int(context_frames),
            "audio_overhang": float(meta.get("audio_overhang", 0.0) or 0.0),
            "source_frames": int(meta.get("source_frames", 0)),
            "last_luma": last_luma,
            "path": str(target),
        },
    }


def context_digest(path: Path) -> str:
    """只读上下文文件的 metadata，拿它的组合哈希（不加载张量）。

    用途：链路血缘校验只需要「上一段消费的是哪个版本的上上段」，
    不需要把上上段的张量读进内存。
    """
    from safetensors import safe_open

    target = Path(path)
    if not target.is_file():
        return ""
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        meta = dict(handle.metadata() or {})
    recorded = meta.get("context_sha256")
    if recorded:
        return str(recorded)
    video_sha, audio_sha = meta.get("video_sha256", ""), meta.get("audio_sha256", "")
    if video_sha and audio_sha:
        return _combined_digest(video_sha, audio_sha)
    return ""
