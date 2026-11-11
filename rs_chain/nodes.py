"""RS - H3 分段续接（单段）。

一次执行只跑一段：
    读上一段落盘的 latent 尾巴 → 作为官方 `minimax_keyframes` 钉在本段时间轴起点
    → 采样 → 解码 → 像素域裁掉衔接帧 → 落盘本段尾巴。

不依赖任何第三方节点：条件、引导、双时钟 shift、采样、解码全部走 ComfyUI 核心
（`comfy.sample` / `comfy.nested_tensor` / `node_helpers` / `comfy.model_sampling`）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:  # 只在 ComfyUI 里存在；纯函数单测不会 import 本模块
    import folder_paths
except Exception:  # noqa: BLE001
    folder_paths = None

from . import h3, media
from .state import (
    CONTEXT_FRAME_CHOICES,
    FPS,
    keyframe_local_frame,
    ChainError,
    audio_window_start_seconds,
    chain_timeline_start,
    clear_segment,
    context_path,
    list_segments,
    make_meta,
    meta_issues,
    meta_path,
    plan_segment,
    read_json,
    resolve_chain_root,
    sanitize_name,
    segment_dir,
    write_json_atomic,
)


def _samplers() -> list[str]:
    import comfy.samplers

    return list(comfy.samplers.KSampler.SAMPLERS)


def _schedulers() -> list[str]:
    import comfy.samplers

    return list(comfy.samplers.KSampler.SCHEDULERS)


def _output_root() -> Path:
    if folder_paths is None:
        raise ChainError("本节点必须在 ComfyUI 内运行（缺少 folder_paths）")
    return Path(folder_paths.get_output_directory())


def _require_multiple_of_32(value: int, label: str) -> int:
    v = int(value)
    if v <= 0 or v % 32 != 0:
        raise ChainError(f"{label}={v} 必须是 32 的倍数（H3 的 latent 分块要求）")
    return v


def _chain_name(chain_dir: str) -> str:
    raw = str(chain_dir or "").strip()
    return sanitize_name(Path(raw).name if Path(raw).is_absolute() else raw)


class RS_H3ChainSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "chain_dir": ("STRING", {"default": "mv-01", "multiline": False}),
                "segment_index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "width": ("INT", {"default": 480, "min": 128, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 864, "min": 128, "max": 4096, "step": 32}),
                "render_window_frames": ("INT", {"default": 124, "min": 22, "max": 1000, "step": 1}),
                "context_frames": (list(CONTEXT_FRAME_CHOICES), {"default": 22}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 60, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_samplers(), {"default": h3.DEFAULT_SAMPLER}),
                "scheduler": (_schedulers(), {"default": h3.DEFAULT_SCHEDULER}),
                "ref_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "force_rerun": ("BOOLEAN", {"default": False}),
                "allow_no_prev": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "drive_audio": ("AUDIO",),
                "ref_image_0": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_image_5": ("IMAGE",),
                "ref_image_6": ("IMAGE",),
                "ref_image_7": ("IMAGE",),
                "ref_image_8": ("IMAGE",),
                "ref_video_0": ("IMAGE",),
                "ref_video_1": ("IMAGE",),
                "ref_video_2": ("IMAGE",),
                "ref_video_audio_0": ("AUDIO",),
                "ref_video_audio_1": ("AUDIO",),
                "ref_video_audio_2": ("AUDIO",),
                "ref_audio_0": ("AUDIO",),
                "ref_audio_1": ("AUDIO",),
                "ref_audio_2": ("AUDIO",),
                "keyframe_image_1": ("IMAGE",),
                "keyframe_time_1": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100000.0, "step": 0.1}),
                "keyframe_image_2": ("IMAGE",),
                "keyframe_time_2": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100000.0, "step": 0.1}),
                "keyframe_image_3": ("IMAGE",),
                "keyframe_time_3": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100000.0, "step": 0.1}),
                "sigmas": ("SIGMAS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "report")
    FUNCTION = "run"
    CATEGORY = "RS/H3"

    def run(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        chain_dir,
        segment_index,
        prompt,
        seed,
        width,
        height,
        render_window_frames,
        context_frames,
        steps,
        cfg,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
        ref_scale,
        force_rerun=False,
        allow_no_prev=False,
        drive_audio=None,
        ref_image_0=None,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        ref_image_5=None,
        ref_image_6=None,
        ref_image_7=None,
        ref_image_8=None,
        ref_video_0=None,
        ref_video_1=None,
        ref_video_2=None,
        ref_video_audio_0=None,
        ref_video_audio_1=None,
        ref_video_audio_2=None,
        ref_audio_0=None,
        ref_audio_1=None,
        ref_audio_2=None,
        keyframe_image_1=None,
        keyframe_time_1=-1.0,
        keyframe_image_2=None,
        keyframe_time_2=-1.0,
        keyframe_image_3=None,
        keyframe_time_3=-1.0,
        sigmas=None,
    ):
        if not str(prompt or "").strip():
            raise ChainError("prompt 为空：这一段没有任何动作描述，先给提示词再跑")

        width = _require_multiple_of_32(width, "width")
        height = _require_multiple_of_32(height, "height")
        root = resolve_chain_root(chain_dir, _output_root())
        chain = _chain_name(chain_dir)

        # 起点从上一段的 meta 推：末段可以为了对准目标时长短一点（124 → 141/107），
        # 用固定步长 (k−1)×(W−C) 会把本段的绝对时间、音频窗口、关键帧位置一起算错。
        plan = plan_segment(
            render_window_frames,
            context_frames,
            segment_index,
            chain_timeline_start(root, segment_index, context_frames, render_window_frames),
        )

        meta = make_meta(
            chain_id=chain,
            plan=plan,
            width=width,
            height=height,
            steps=steps,
            sampler=sampler_name,
            scheduler=scheduler,
            shift_video=shift_video,
            shift_audio=shift_audio,
            task_type="ref2va",
            audio_mode="lock_source" if drive_audio is not None else "native",
            prompt=prompt,
            seed=seed,
        )

        # ---- 本段是否已经跑过（默认拒绝覆盖：已验收的镜头不该被手滑重跑）
        done_meta, done_ctx = meta_path(root, plan["index"]), context_path(root, plan["index"])
        if done_meta.is_file() and done_ctx.is_file() and not force_rerun:
            raise ChainError(
                f"第 {plan['index']} 段的状态已存在（{segment_dir(root, plan['index'])}）。"
                f"要覆盖就打开 force_rerun；要重跑本段及其后，先用 RS_H3ChainManage 的 reset_from"
            )

        # ---- 上一段
        prev = None
        if plan["index"] > 1:
            prev_meta_path = meta_path(root, plan["index"] - 1)
            prev_ctx_path = context_path(root, plan["index"] - 1)
            if prev_meta_path.is_file() and prev_ctx_path.is_file():
                issues = meta_issues(read_json(prev_meta_path), meta)
                if issues:
                    raise ChainError(
                        "上一段与本段的坐标系不一致，接上去只会静默错位：\n  - "
                        + "\n  - ".join(issues)
                        + "\n（画幅/窗口/帧率必须整条链一致；要换规格请用新的 chain_dir）"
                    )
                prev = media.read_context(
                    prev_ctx_path,
                    chain_id=chain,
                    source_index=plan["index"] - 2,
                    context_frames=plan["context_frames"],
                )
            elif not allow_no_prev:
                raise ChainError(
                    f"第 {plan['index']} 段需要第 {plan['index'] - 1} 段的状态，但 {segment_dir(root, plan['index'] - 1)} 里没有。"
                    f"先跑上一段（同一 chain_dir）；确实要无上下文生成，打开 allow_no_prev（接缝会跳）"
                )

        # ---- 音频：按时间轴取本段窗口（续接段要多取一个衔接窗口，否则音画差 22/24 秒）
        win_audio = None
        if drive_audio is not None:
            win_audio = media.window_audio(
                drive_audio, audio_window_start_seconds(plan), plan["render_frames"]
            )

        # 顺序即编号：图片 → 视频 → 独立音频（与官方 MiniMaxH3ReferenceToVideo 一致）
        refs = [
            img
            for img in (
                ref_image_0, ref_image_1, ref_image_2, ref_image_3, ref_image_4,
                ref_image_5, ref_image_6, ref_image_7, ref_image_8,
            )
            if img is not None
        ]
        ref_videos = [
            (frames, soundtrack)
            for frames, soundtrack in (
                (ref_video_0, ref_video_audio_0),
                (ref_video_1, ref_video_audio_1),
                (ref_video_2, ref_video_audio_2),
            )
            if frames is not None
        ]
        ref_audios = [a for a in (ref_audio_0, ref_audio_1, ref_audio_2) if a is not None]

        t0 = time.time()
        positive = h3.encode_conditioning(
            clip,
            video_vae,
            prompt,
            width,
            height,
            audio_vae=audio_vae,
            images=refs,
            videos=ref_videos,
            audios=ref_audios,
            length=plan["render_frames"],
            ref_scale=ref_scale,
        )

        # 段首锚点：
        #   视频 —— 第 2 段起用上一段的 latent 尾巴；第 1 段的 0 秒锚点由关键帧槽
        #           （keyframe_time_1=0）表达，不再有单独的 first_frame；
        #   音频 —— 接了 drive_audio 就钉**整段源音频**（否则模型自编一条音轨、
        #           我们再把歌 mux 上去，动作与鼓点就毫无关系，且不报错）。
        audio_t = h3.temporal_shape(plan["render_frames"])[2]
        audio_pin = h3.encode_audio_guide(audio_vae, win_audio, audio_t) if win_audio is not None else None
        video_pin = prev["video_tail"] if prev is not None else None
        positive = h3.add_guide_keyframe(
            positive,
            video_pin,
            audio_pin if audio_pin is not None else (prev["audio_tail"] if prev is not None else None),
            0,
        )

        # ---- 关键帧：按**成片时间轴**（秒）给位置，节点换算成本段渲染窗口内的帧号。
        # 用绝对秒而不是"本段第几帧"，是因为用户说的是"第 8 秒放这张图"；
        # 段的时间轴起点由段几何固定（(k-1)×(W-C)），不随内容变化，所以绝对秒是稳定的。
        applied_keyframes: list[dict] = []
        for slot, (kf_img, kf_time) in enumerate(
            (
                (keyframe_image_1, keyframe_time_1),
                (keyframe_image_2, keyframe_time_2),
                (keyframe_image_3, keyframe_time_3),
            ),
            start=1,
        ):
            if kf_img is None:
                if float(kf_time) >= 0:
                    raise ChainError(f"keyframe_time_{slot}={kf_time} 已设置，但 keyframe_image_{slot} 没接图")
                continue
            if float(kf_time) < 0:
                raise ChainError(f"keyframe_image_{slot} 接了图，但 keyframe_time_{slot} 还是 -1（未启用）")
            local = keyframe_local_frame(kf_time, plan)
            kf_latent, used = h3.encode_keyframe_clip(video_vae, kf_img, width, height)
            if local < plan["head_trim"]:
                raise ChainError(
                    f"关键帧 {slot} 落在 {float(kf_time)}s，换算到本段是第 {local} 帧，"
                    f"但它落在段首衔接窗口里（本段前 {plan['head_trim']} 帧是上一段的尾巴，会被裁掉）。"
                    f"0 秒/段首关键帧只对第 1 段有效；本段最早可用 "
                    f"{(plan['timeline_start_frames'] + plan['head_trim']) / 24:.2f}s"
                )
            if local + used > plan["render_frames"]:
                raise ChainError(
                    f"关键帧 {slot} 放不下：本段从第 {local} 帧起要占 {used} 帧，"
                    f"但渲染窗口只有 {plan['render_frames']} 帧（本段最晚可放到 "
                    f"{(plan['timeline_start_frames'] + plan['render_frames'] - used) / 24:.2f}s）"
                )
            positive = h3.add_guide_keyframe(positive, kf_latent, None, local)
            applied_keyframes.append(
                {
                    "slot": slot,
                    "time_seconds": round(float(kf_time), 3),
                    "local_frame": local,
                    "clip_frames": used,
                }
            )

        latent, _frame_count = h3.empty_av_latent(width, height, plan["render_frames"])
        # 接了自定义 Sigmas 就完全按它走（video_pika_h3_flex_ref2va 的 lcm + ManualSigmas），
        # 不再套 H3 的双时钟 shift，否则会改掉那条 sigma 曲线。
        shifted = model if sigmas is not None else h3.apply_sigma_shift(model, shift_video, shift_audio)
        t_cond = time.time() - t0

        t1 = time.time()
        print(
            f"[RS-H3Chain] 段{plan['index']} 开始采样：{width}x{height} · {plan['render_frames']} 帧"
            f" · {steps} 步 · cfg={cfg} · sampler={sampler_name}"
            + ("（自定义 Sigmas）" if sigmas is not None else f" · scheduler={scheduler}"),
            flush=True,
        )
        sampled = h3.sample_av(
            shifted,
            positive,
            latent,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            sigmas=sigmas,
        )
        t_sample = time.time() - t1
        print(f"[RS-H3Chain] 段{plan['index']} 采样完成（{t_sample:.1f}s），开始 VAE 解码…", flush=True)

        t2 = time.time()
        frames, generated_audio = h3.decode_av(video_vae, audio_vae, sampled)
        t_decode = time.time() - t2
        print(
            f"[RS-H3Chain] 段{plan['index']} 解码完成：{int(frames.shape[0])} 帧"
            f"（{t_decode:.1f}s）"
            + ("，含模型音轨" if generated_audio is not None else "，无模型音轨"),
            flush=True,
        )

        # ---- 裁头：段首被钉住的窗口帧解码后会原样出现，必须裁掉（音频同量裁）
        drop = plan["head_trim"]
        out_frames = media.trim_frames(frames, drop)
        delivery_audio = win_audio if win_audio is not None else generated_audio
        if delivery_audio is None:
            raise ChainError("本段没有可交付的音频（既没有 drive_audio，模型也没生成音频）")
        out_audio = media.trim_audio(delivery_audio, drop)

        # ---- 落盘：先写 context 再写 meta（meta 是"这段可用"的标记，必须最后写）
        video_tail, audio_tail = h3.latent_tails(sampled, plan["context_frames"])
        sampled_video, _sampled_audio = h3._nested_parts(sampled)
        ctx_info = media.write_context(
            context_path(root, plan["index"]),
            video_tail,
            audio_tail,
            chain_id=chain,
            segment_index=plan["index"] - 1,
            context_frames=plan["context_frames"],
            audio_overhang=h3.audio_overhang(sampled),
            source_frames=h3.pixel_frames_from_latent_t(int(sampled_video.shape[2])),
        )
        meta["ref_hashes"] = [media.image_fingerprint(img) for img in refs]
        meta["drop_head_frames"] = drop
        meta["context_shape"] = ctx_info
        meta["elapsed_seconds"] = round(time.time() - t0, 1)
        meta["elapsed"] = {
            "conditioning": round(t_cond, 1),
            "sample": round(t_sample, 1),
            "decode": round(t_decode, 1),
        }
        meta["chain_root"] = str(root)
        write_json_atomic(meta_path(root, plan["index"]), meta)

        report = {
            "chain": chain,
            "chain_root": str(root),
            "segment_index": plan["index"],
            "render_frames": plan["render_frames"],
            "dropped_head_frames": drop,
            "output_frames": int(out_frames.shape[0]),
            "output_seconds": round(int(out_frames.shape[0]) / 24.0, 3),
            "timeline_start_seconds": round(plan["timeline_start_seconds"], 3),
            "has_prev_context": prev is not None,
            "audio_guided": audio_pin is not None,
            "keyframes": applied_keyframes,
            "refs": {
                "images": len(refs),
                "videos": len(ref_videos),
                "audios": len(ref_audios),
            },
            "custom_sigmas": sigmas is not None,
            "sampling_steps": int(len(sigmas) - 1) if sigmas is not None else int(steps),
            "ref_hashes": meta["ref_hashes"],
            "elapsed": meta["elapsed"],
            "next": f"第 {plan['index'] + 1} 段",
        }
        return (out_frames, out_audio, json.dumps(report, ensure_ascii=False, indent=2))


class RS_H3ChainManage:
    """链维护：看状态 / 删某段及其后的状态 / 清空整条链。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain_dir": ("STRING", {"default": "mv-01", "multiline": False}),
                "action": (["status", "reset_from", "reset_all"], {"default": "status"}),
                "from_index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "confirm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "RS/H3"
    OUTPUT_NODE = True

    def run(self, chain_dir, action, from_index, confirm):
        root = resolve_chain_root(chain_dir, _output_root())
        done = list_segments(root)

        if action == "status":
            if not root.is_dir():
                return (f"链目录不存在：{root}",)
            lines = [f"链目录：{root}", f"已完成段：{done if done else '（无）'}"]
            for idx in done:
                try:
                    m = read_json(meta_path(root, idx))
                    hashes = m.get("ref_hashes") or []
                    prev_hashes = (read_json(meta_path(root, idx - 1)).get("ref_hashes") or []) if idx > 1 else []
                    changed = bool(prev_hashes) and hashes != prev_hashes
                    ref_note = f" · 参考{'+'.join(h[:6] for h in hashes) or '无'}"
                    if changed:
                        ref_note += "（本段换了参考）"
                    lines.append(
                        f"  段{idx}: {m.get('width')}x{m.get('height')} · 窗口 {m.get('render_frames')} 帧"
                        f" · 输出 {m.get('output_frames')} 帧 · {m.get('elapsed_seconds')}s{ref_note}"
                    )
                except ChainError as exc:
                    lines.append(f"  段{idx}: meta 读取失败（{exc}）")
            return ("\n".join(lines),)

        if not confirm:
            raise ChainError(f"{action} 会删除状态文件，请把 confirm 打开")

        if action == "reset_all":
            removed = [idx for idx in done if clear_segment(root, idx)]
            return (f"已删除 {len(removed)} 段状态：{removed}",)

        removed = [idx for idx in done if idx >= int(from_index) and clear_segment(root, idx)]
        return (f"已删除第 {from_index} 段及之后共 {len(removed)} 段：{removed}",)
