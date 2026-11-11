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
) -> dict:
    """把本段的衔接窗口落盘（video_tail / audio_tail + 校验和）。

    `segment_index` 是 **0 起**（与官方 keyframe 的段号语义一致）；`meta.json` 里是 1 起。
    """
    if video_tail is None or audio_tail is None:
        raise ChainError("上下文尾巴为空：采样结果里没有 video/audio latent")
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
        "video_sha256": tensor_sha256(video_tail),
        "audio_sha256": tensor_sha256(audio_tail),
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
    return {"video_shape": list(video_tail.shape), "audio_shape": list(audio_tail.shape)}


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
    if meta.get("video_sha256") and meta["video_sha256"] != tensor_sha256(video_tail):
        raise ChainError(f"{target} 的 video 校验和不符（文件被改过或写坏了）")
    if meta.get("audio_sha256") and meta["audio_sha256"] != tensor_sha256(audio_tail):
        raise ChainError(f"{target} 的 audio 校验和不符（文件被改过或写坏了）")
    if not torch.isfinite(video_tail).all() or not torch.isfinite(audio_tail).all():
        raise ChainError(f"{target} 含 NaN/Inf（采样发散），不能作为下一段的引导")

    return {
        "video_tail": video_tail,
        "audio_tail": audio_tail,
        "meta": {
            "chain_id": str(chain_id),
            "source_segment_index": int(source_index),
            "context_frames": int(context_frames),
            "audio_overhang": float(meta.get("audio_overhang", 0.0) or 0.0),
            "source_frames": int(meta.get("source_frames", 0)),
            "path": str(target),
        },
    }
