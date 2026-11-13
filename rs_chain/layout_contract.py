"""只读校验宿主 H3 keyframe 布局的算术语义。

为什么需要这个模块
------------------
续接段的音频尾必须**结束**在视频钉住窗的末端，所以它的
`resolved_frame_index` 是分数、而且通常是负数：

    index = end_coord / FRAME_RESCALE - audio_steps / FRAME_RESCALE

官方 `MiniMaxH3AddGuide` 只接受非负整数，因此这条路没有任何官方节点会走、
也没有任何上游测试覆盖。宿主只要把索引 `int()` 一下或加一个下界检查，
音频就会被静默放到错误的时刻——不报错，只是每段偏 1/40 秒。

这里**不修改**宿主，只在第一次真正需要分数索引时构造几个最小布局，
证明三件事仍然成立：

1. 整数索引仍按 `FRAME_RESCALE × 帧号` 相对目标起点定位；
2. 分数、负数的 `resolved_frame_index` 被按字面值接受；
3. 带 `audio_latent` 的 keyframe 从该位置**向后**铺开，而不是向前。

失败时抛 `ChainError`，由调用方决定回退到索引 0（并把降级写进 report）。
"""

from __future__ import annotations

from .state import ChainError

_CHECKED = None  # None = 未跑；True = 通过；ChainError = 失败（缓存，避免重复构造）


class _Stub:
    """布局只读 shape，不需要真的张量。"""

    def __init__(self, shape):
        self.shape = shape


def _origin(layout) -> float:
    """目标视频行的起点（= 目标时间轴原点）。"""
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            f"期望最后一段是目标视频行，实际是 {kind!r}（{a}..{b}）"
        )
    return float(layout.position_ids[a, 0])


def _segments(layout, kind: str) -> list[tuple[int, int]]:
    return [(a, b) for a, b, k in layout.segments if k == kind]


def _check():
    from comfy.ldm.minimax import model as mm

    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    fr = float(getattr(mm, "FRAME_RESCALE", 0.0))
    if abs(fr - 5.0 / 3.0) > 1e-9:
        raise RuntimeError(
            f"宿主的 FRAME_RESCALE={fr!r}，本节点按 5/3 计算音频索引；"
            "音频与视频的时间轴换算已经变了"
        )

    def build(keyframes=None):
        return mm.PackedLayout(text_len, latent_t, lh, lw, audio_t, keyframes=keyframes)

    video = _Stub((1, 24, 1, lh, lw))

    # 1) 整数索引：距离原点必须是 FRAME_RESCALE × 帧号
    layout = build([{"resolved_frame_index": p, "latent": video} for p in (0, 3)])
    origin = _origin(layout)
    got = [float(layout.position_ids[a, 0]) - origin for a, _ in _segments(layout, "cond")]
    want = [0.0, fr * 3]
    if len(got) != 2 or any(abs(g - w) > 1e-9 for g, w in zip(got, want)):
        raise RuntimeError(
            f"整数 keyframe 的落点变了：实际 {got}，期望 {want}"
            "（宿主不再按 FRAME_RESCALE × 帧号定位）"
        )

    # 2) 分数 + 负索引 + 音频窗：从索引位置向后铺，末端落在指定音频坐标
    #    取一组非整数的组合，避免 int() 截断也能蒙混过关。
    rt, end_coord = 40, 8
    end_frame = end_coord / fr
    index = end_frame - rt / fr
    if index >= 0 or index == int(index):
        raise RuntimeError("内部检查参数写错了：这组数没有构成分数负索引")

    layout = build([{"resolved_frame_index": index, "audio_latent": _Stub((1, 32, 2, rt))}])
    spans = _segments(layout, "cond_audio")
    if len(spans) != 1:
        raise RuntimeError(f"带音频的 keyframe 产生了 {len(spans)} 段 cond_audio，期望 1 段")
    a, b = spans[0]
    if b - a != 2 * rt:
        raise RuntimeError(
            f"钉住的音频窗有 {b - a} 行，期望 {2 * rt} 行（立体声、通道优先）"
        )
    origin = _origin(layout)
    start = float(layout.position_ids[a:b, 0].min()) - origin
    end = start + float(rt)
    if abs(end - fr * end_frame) > 1e-9:
        raise RuntimeError(
            f"钉住的音频窗结束于原点后 {end:.6f}，期望 {fr * end_frame:.6f}"
            "（分数/负索引不再被按字面值接受）"
        )
    if start >= 0.0:
        raise RuntimeError(
            f"钉住的音频窗从原点后 {start:.6f} 才开始，它应该跨过原点、结束在接缝处"
        )


def ensure() -> None:
    """第一次调用时执行检查；失败结果会被缓存并原样抛出。"""
    global _CHECKED
    if _CHECKED is True:
        return
    if isinstance(_CHECKED, ChainError):
        raise _CHECKED
    try:
        _check()
    except ChainError:
        raise
    except Exception as exc:  # noqa: BLE001 —— 统一成可读错误，交给调用方降级
        failure = ChainError(
            "宿主 H3 布局校验失败，音频锚点无法按分数索引放置："
            f"{type(exc).__name__}: {exc}"
        )
        _CHECKED = failure
        raise failure from exc
    _CHECKED = True
