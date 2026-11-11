"""只用 ComfyUI 核心实现的 MiniMax H3 能力（不依赖 T8、不依赖 JZL）。

两个参考实现各有一半更好，这里各取所长：

| 维度 | T8 InNodeLoop | JZL 无限时长 | 本模块取 |
|---|---|---|---|
| 段间引导 | 自己 patch `extra_conds`，每 token 一个 keyframe | 官方 `minimax_keyframes`，一个多帧 keyframe | **JZL**（原生，不 patch，升级不易碎） |
| 视频/音频 shift | 自研双时钟采样 | 不设（用模型默认） | **T8 的语义 + 官方 `MiniMaxH3SigmaShift` 的实现** |
| 采样 | 自研循环 | `comfy.sample.sample` | 官方路径 |
| 裁剪 | 像素域 | 像素域 + settle | 像素域（settle 待做，见 README） |

关键源码事实（官方 `nodes_minimax_h3.py`，不是推测）：
  · 视频 latent 的 T 轴不均匀：token k 覆盖 FRAME_PER_TOKEN[k%5]=(1,4,4,4,4) 帧，5 token = 17 帧；
    所以段长必须是 17k+5、上下文窗口必须是 17m+5，两者相减才是 17 的倍数 ⇒ 尾窗起点落在 token 边界；
  · `MiniMaxH3AddGuide` 的多帧引导就是 `{"resolved_frame_index": 0, "latent": <N token 的 latent>}`，
    N token 对应 17m+5 帧（22 帧 → 7 token）；音频用 `audio_latent`，一步 = 1/40 秒；
  · keyframe 的 cond 行永不降噪、每步重注入 ⇒ 段首那 W 帧解码后会原样出现，必须裁掉（音频同量裁）。
"""

from __future__ import annotations

import math

import torch

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers

from .state import (
    CONTEXT_FRAME_STEPS,
    FPS,
    align_frame_count,
    pixel_frames_from_latent_t,
    temporal_shape,
    video_latent_t,
)

AUDIO_LATENT_FPS = 40
FRAME_RESCALE = 5.0 / 3.0
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048

DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "simple"


# ---------------------------------------------------------------- 帧数换算


def empty_av_latent(width: int, height: int, length: int):
    """返回 ({"samples": NestedTensor}, 实际帧数)。形状与官方 EmptyMiniMaxH3LatentAV 一致。"""
    frame_count, latent_t, audio_t = temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16], device=device)
    audio = torch.zeros([1, 32, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _nested_parts(samples) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(samples, dict):
        samples = samples.get("samples", samples)
    if getattr(samples, "is_nested", False):
        parts = list(samples.tensors)
    else:
        parts = [samples]
    video = parts[0]
    audio = parts[1] if len(parts) > 1 else None
    return video, audio


def latent_tails(samples, context_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """采样结果 → 本段要落盘的尾部窗口（video [1,24,N,h,w] + audio [1,32,2,T]）。

    取的是**采样出来的 latent 尾部**，不是解码后再编码的像素：
    与下一段目标 latent 同画布同空间，零往返损失、零累积漂移（JZL 注释里的同结论）。
    """
    steps = CONTEXT_FRAME_STEPS.get(int(context_frames))
    if steps is None:
        raise ValueError(f"context_frames={context_frames} 没有对应的 latent 步数")
    video, audio = _nested_parts(samples)
    if int(video.shape[2]) < steps:
        raise ValueError(f"latent T={int(video.shape[2])} 短于上下文窗口 {steps} token")
    if audio is None:
        raise ValueError("采样结果没有音频 latent：无法保存跨段音频续接")
    audio_steps = min(int(audio.shape[-1]), round(int(context_frames) / FPS * AUDIO_LATENT_FPS))
    if audio_steps < 1:
        raise ValueError(f"context_frames={context_frames} 对应的音频步数为 0")
    return (
        video[:1, :, -steps:].detach().cpu().contiguous(),
        audio[:1, :, :, -audio_steps:].detach().cpu().contiguous(),
    )


def audio_overhang(samples) -> float:
    """音频 latent 与视频时间轴的相位差（0 ≤ x < 1）。写错会让下一段音画漂 1/40 秒。"""
    video, audio = _nested_parts(samples)
    if audio is None:
        return 0.0
    frames = pixel_frames_from_latent_t(int(video.shape[2]))
    value = int(audio.shape[-1]) - FRAME_RESCALE * frames
    return value if 0.0 <= value < 1.0 else 0.0


# ---------------------------------------------------------------- 条件


def _resize_like(image: torch.Tensor, width: int, height: int, crop: str = "disabled") -> torch.Tensor:
    """[B,H,W,C] → 目标尺寸。

    `crop="disabled"` = 拉伸（官方 i2v 首帧/参考图的语义）；
    `crop="center"` = 等比铺满后居中裁切（官方 AddGuide 关键帧/尾帧的语义）。
    """
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _ref_scale_factor(w: int, h: int, width: int, height: int, ref_scale: float) -> float:
    """参考图缩放：官方 match 是「只缩不放」的面积适配；ref_scale > 1 才额外放大（面积倍率）。"""
    down = min(1.0, math.sqrt((width * height) / float(w * h)))
    scale = down * max(1.0, ref_scale)
    return min(scale, max(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h))))


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """参考视频的画布：768 短边、768×1344 面积上限，逐轴对齐 32（与官方 R2V 一致）。"""
    ratio = float(width) / float(max(1, height))
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def encode_conditioning(clip, vae, prompt: str, width: int, height: int, *,
                        audio_vae=None,
                        images: list[torch.Tensor] | None = None,
                        videos: list[tuple[torch.Tensor, object]] | None = None,
                        audios: list[object] | None = None,
                        length: int | None = None,
                        ref_scale: float = 1.0):
    """prompt（+ 参考图/视频/音频）→ positive conditioning。

    参考物顺序与官方 `MiniMaxH3ReferenceToVideo` 一致：图片 → 视频（自带音轨的
    `<Audio j>` 标签紧挨在 `<Video k>` 之前）→ 独立音频；序号按类型从 1 计数。
    `videos` 里每项是 `(IMAGE 批次, AUDIO 或 None)`。
    """
    ref_items = []
    ref_blocks = []

    for img in (images or []):
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        scale = _ref_scale_factor(w, h, width, height, ref_scale)
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize_like(img[:1], tw, th)
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    if videos:
        frame_count = temporal_shape(max(5, int(length or 124)))[0]
        for frames, soundtrack in videos:
            if frames is None:
                continue
            vh, vw = int(frames.shape[1]), int(frames.shape[2])
            cw, ch = adapt_canvas(vw, vh)
            # 比画布还小的参考视频不放大（只缩不放）
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize_like(frames, cw, ch)
            if int(frames.shape[0]) > frame_count:
                frames = frames[:frame_count]
            n = int(frames.shape[0])
            if n < 5:
                raise ValueError("参考视频至少需要 5 帧（24fps ≈ 0.2s）")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                if audio_vae is None:
                    raise ValueError("参考视频带音轨，但没有接 audio_vae")
                audio_latent, ref_audio_t = encode_ref_audio(audio_vae, soundtrack)
                # 视频音轨占一个 <Audio j> 编号，排在 <Video k> 之前
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, n, FPS // 2))
            ref_items.append({
                "type": "video",
                "data": frames[sample_idx],
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": int(z.shape[2]),
                "latent_h": ch // 16,
                "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t,
                "latent": z,
                "audio_latent": audio_latent,
            })

    for audio in (audios or []):
        if audio is None:
            continue
        if audio_vae is None:
            raise ValueError("接了参考音频，但没有接 audio_vae")
        audio_latent, ref_audio_t = encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    return cond


def encode_keyframe_clip(vae, image: torch.Tensor, width: int, height: int) -> tuple[torch.Tensor, int]:
    """关键帧图 → (latent, 实际用的帧数)。

    和官方 AddGuide 同规则：一批图 ≥5 帧就当成小片段，向下吸附到 17k+5（5/22/39…）；
    少于 5 帧只用第一张（单帧锚点）。返回实际帧数是给调用方做边界检查用的。
    """
    n = int(image.shape[0])
    if n < 5:
        n = 1
    else:
        while n % 17 != 5:
            n -= 1
    frames = _resize_like(image[:n], width, height, "center")
    return vae.encode(frames), n


def encode_ref_audio(audio_vae, audio) -> tuple[torch.Tensor, int]:
    """参考音频（独立音频 / 视频原声）→ (audio latent, token 数)。"""
    wave, rate = media_waveform(audio)
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    wave = _resample_linear(wave[:1].float(), rate, vae_rate)
    z = audio_vae.encode(wave.movedim(1, -1))
    return z, int(z.shape[-1])


def _resample_linear(wave: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
    """线性重采样（不引 torchaudio：少一个环境依赖，音质差异在这里听不出来）。"""
    if int(src_rate) == int(dst_rate):
        return wave
    n = max(1, round(wave.shape[-1] * float(dst_rate) / float(src_rate)))
    out = torch.nn.functional.interpolate(wave, size=n, mode="linear", align_corners=False)
    return out


def encode_audio_guide(audio_vae, audio, audio_t: int) -> torch.Tensor:
    """把本段要用的整段音频编码成 audio_latent，钉在时间轴起点。

    为什么必须钉：不钉的话模型自己编一条音频、视频跟着那条走，我们再把歌 mux 上去 →
    动作与鼓点毫无关系（不报错，但 MV 全废）。官方 AddGuide 的 audio 输入就是这个语义。
    """
    wave, rate = media_waveform(audio)
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    wave = _resample_linear(wave[:1].float(), rate, vae_rate)
    z = audio_vae.encode(wave.movedim(1, -1))
    if int(z.shape[-1]) < int(audio_t):
        z = torch.nn.functional.pad(z, (0, int(audio_t) - int(z.shape[-1])))
    return z[..., : int(audio_t)].contiguous()


def media_waveform(audio):
    """避免 h3 → media 的循环 import：这里只做最小解包。"""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("audio 结构不认识（期望 {'waveform','sample_rate'}）")
    wave = audio["waveform"]
    if not torch.is_tensor(wave):
        wave = torch.as_tensor(wave)
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    return wave, int(audio["sample_rate"])


def add_guide_keyframe(positive, video_tail: torch.Tensor | None, audio_tail: torch.Tensor | None,
                       frame_index: int = 0):
    """把锚点钉成本段时间轴起点（官方 MiniMaxH3AddGuide 的同一结构）。

    一个 keyframe 可以同时带 `latent`（视频片段）与 `audio_latent`（整段音频）——
    这正是官方"带音轨的引导片段"的形状：视频只钉前 N 帧，音频可以覆盖整段。
    不 patch 模型：官方 `minimax_keyframes` 本来就被 DiT 读取（cond 行永不降噪、每步重注入）。
    """
    if video_tail is None and audio_tail is None:
        return positive
    keyframes = list((positive[0][1] or {}).get("minimax_keyframes") or [])
    keyframe = {"resolved_frame_index": int(frame_index)}
    if video_tail is not None:
        keyframe["latent"] = video_tail
    if audio_tail is not None:
        keyframe["audio_latent"] = audio_tail
    keyframes.append(keyframe)
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})


# ---------------------------------------------------------------- 采样


def apply_sigma_shift(model, shift_video: float, shift_audio: float):
    """视频/音频各自的 flow shift（等价官方 MiniMaxH3SigmaShift，不依赖任何自定义节点）。

    为什么需要：H3 的视频与音频共用一条采样轨迹，但两者的 shift 不同（12 / 3）；
    只设一个会让音频要么过噪要么过平滑。官方节点做的就是这两件事：换 model_sampling、
    把两个值塞进 transformer_options 交给 DiT。
    """
    if shift_video <= 0 or shift_audio <= 0:
        return model
    patched = model.clone()

    class _ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = patched.get_model_object("model_sampling")
    sampling = _ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(shift=float(shift_video), audio_shift=float(shift_audio))
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    patched.add_object_patch("model_sampling", sampling)

    options = patched.model_options.get("transformer_options", {}).copy()
    options["minimax_h3_sigma_shift_video"] = float(shift_video)
    options["minimax_h3_sigma_shift_audio"] = float(shift_audio)
    patched.model_options["transformer_options"] = options
    return patched


def sample_av(model, positive, latent: dict, *, seed: int, steps: int, cfg: float,
              sampler_name: str, scheduler: str, sigmas=None):
    """跑一次采样。与官方 KSampler / SamplerCustomAdvanced 同一条路径。

    `sigmas` 非空时走自定义 Sigmas（video_pika_h3_flex_ref2va 必须用这条路），
    此时 `scheduler` / `steps` 不参与（步数由 sigmas 长度决定）。
    """
    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, int(seed), None)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if sigmas is not None:
        # sample_custom 的 sampler 参数是**采样器对象**（KSamplerSelect 的输出），
        # 不是名字字符串——传 "lcm" 会在 sampler.sample 处炸 str has no attribute sample
        return comfy.sample.sample_custom(
            model,
            noise,
            float(cfg),
            comfy.samplers.sampler_object(str(sampler_name)),
            sigmas,
            positive,
            [],
            latent_image,
            disable_pbar=disable_pbar,
            seed=int(seed),
        )
    return comfy.sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        str(sampler_name),
        str(scheduler),
        positive,
        [],
        latent_image,
        denoise=1.0,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )


def decode_av(vae, audio_vae, samples):
    """联合 AV latent → (IMAGE [T,H,W,C], AUDIO dict|None)。"""
    video_z, audio_z = _nested_parts(samples)
    image = vae.decode(video_z)
    if image.ndim == 5:
        image = image[0]

    audio = None
    if audio_z is not None and audio_vae is not None:
        waveform = audio_vae.first_stage_model.decode(audio_z)
        audio = {
            "waveform": waveform,
            "sample_rate": int(getattr(audio_vae, "audio_sample_rate", 32000)),
        }
    return image, audio
