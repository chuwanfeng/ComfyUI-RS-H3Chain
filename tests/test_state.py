"""state.py 的纯标准库测试：`python tests/test_state.py`（不需要 torch / ComfyUI）。

这些断言之所以值得写：段几何算错不会报错，只会表现为"接缝跳一下"或"每段短了 22 帧"，
在成片里极难定位。把它们钉死在单元测试里。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_chain.state import (  # noqa: E402
    FPS,
    ChainError,
    audio_window_start_seconds,
    chain_timeline_start,
    clear_segment,
    context_path,
    list_segments,
    make_meta,
    meta_issues,
    meta_path,
    keyframe_local_frame,
    plan_segment,
    temporal_shape,
    video_latent_t,
    resolve_chain_root,
    sanitize_name,
    segment_dir,
    write_json_atomic,
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


def expect_error(name: str, fn, contains: str = "") -> None:
    global PASS, FAIL
    try:
        fn()
    except ChainError as exc:
        if contains and contains not in str(exc):
            FAIL += 1
            print(f"FAIL  {name}  报错信息里没有「{contains}」：{exc}")
        else:
            PASS += 1
            print(f"PASS  {name}")
        return
    except Exception as exc:  # noqa: BLE001
        FAIL += 1
        print(f"FAIL  {name}  抛了非 ChainError：{type(exc).__name__}: {exc}")
        return
    FAIL += 1
    print(f"FAIL  {name}  没有报错")


# ---------------------------------------------------------------- 帧数换算

check("124 帧 = 37 token / 207 音频步", temporal_shape(124) == (124, 37, 207), str(temporal_shape(124)))
check("22 帧窗口 = 7 token", video_latent_t(22) == 7)
check("158 帧 = 47 token", temporal_shape(158)[1] == 47, str(temporal_shape(158)))

# ---------------------------------------------------------------- 段几何

p1 = plan_segment(124, 22, 1)
check("第 1 段不裁头、产出 = 渲染窗口", p1["head_trim"] == 0 and p1["output_frames"] == 124, str(p1))
check("第 1 段时间轴从 0 起", p1["timeline_start_frames"] == 0)
check("第 1 段音频从 0 取（不会负起点）", audio_window_start_seconds(p1) == 0.0)

p2 = plan_segment(124, 22, 2)
check("第 2 段裁掉 22 帧头", p2["head_trim"] == 22)
check("第 2 段产出 102 帧", p2["output_frames"] == 102, str(p2["output_frames"]))
check("第 2 段时间轴起点 102 帧", p2["timeline_start_frames"] == 102)
check(
    "第 2 段音频取自 102/24 秒（= 渲染窗口起点，衔接帧一起取、随后同量裁掉）",
    abs(audio_window_start_seconds(p2) - 102 / 24) < 1e-9,
    str(audio_window_start_seconds(p2)),
)
check(
    "第 2 段交付音频的起点 = 124/24 秒（与画面首帧同一点）",
    abs(audio_window_start_seconds(p2) + p2["head_trim"] / FPS - 124 / 24) < 1e-9,
    f'{audio_window_start_seconds(p2) + p2["head_trim"] / FPS:.4f}',
)

p3 = plan_segment(158, 39, 3)
check("158 帧 + 39 上下文 → 每段推进 119 帧", p3["advance_frames"] == 119 and p3["timeline_start_frames"] == 238)

expect_error("段长 120 不是 17k+5 → 拒绝", lambda: plan_segment(120, 22, 1), "17k+5")
expect_error("段长 124 / 上下文 20 → 拒绝", lambda: plan_segment(124, 20, 1), "不是支持的窗口")
expect_error("上下文 ≥ 段长 → 拒绝", lambda: plan_segment(22, 22, 1), "必须小于")
expect_error("段号 0 → 拒绝", lambda: plan_segment(124, 22, 0), "从 1 开始")

# ---------------------------------------------------------------- 末段短一点（对准目标时长）

# 10 秒目标：124 + (141 − 22) = 243 帧 = 10.125s（固定 124 窗口只有 226 帧 = 9.42s）
p2b = plan_segment(141, 22, 2, timeline_start_frames=102)
check("末段 141 帧：起点由上一段推出 = 102", p2b["timeline_start_frames"] == 102)
check("末段 141 帧：裁头 22、交付 119 帧", p2b["head_trim"] == 22 and p2b["output_frames"] == 119, str(p2b["output_frames"]))
check(
    "124 + 119 = 243 帧 = 10.125s",
    abs((p1["output_frames"] + p2b["output_frames"]) / FPS - 10.125) < 1e-9,
)
expect_error(
    "起点不是 17 的倍数 → 拒绝（会周期性重影）",
    lambda: plan_segment(141, 22, 2, timeline_start_frames=100),
    "17 的倍数",
)

# ---------------------------------------------------------------- 关键帧换算

check("段1 的 3.0s 是窗口内第 72 帧", keyframe_local_frame(3.0, p1) == 72)
check("段2 的 6.0s 是窗口内第 42 帧", keyframe_local_frame(6.0, p2) == 42, str(keyframe_local_frame(6.0, p2)))
check(
    "段2 的 5.1s 落在衔接窗口里（18 帧 < 22）→ 调用方必须拒绝",
    keyframe_local_frame(5.1, p2) < p2["head_trim"],
    str(keyframe_local_frame(5.1, p2)),
)

# ---------------------------------------------------------------- meta 守卫

meta1 = make_meta(
    chain_id="mv", plan=p1, width=480, height=864, steps=4, sampler="dual_clock_euler",
    scheduler="native_flow", shift_video=12.0, shift_audio=3.0, task_type="auto",
    audio_mode="lock_source", prompt="p1", seed=1,
)
meta2 = make_meta(
    chain_id="mv", plan=p2, width=480, height=864, steps=4, sampler="dual_clock_euler",
    scheduler="native_flow", shift_video=12.0, shift_audio=3.0, task_type="auto",
    audio_mode="lock_source", prompt="p2", seed=2,
)
check("同规格相邻段可以接", meta_issues(meta1, meta2) == [], str(meta_issues(meta1, meta2)))

wrong = dict(meta2)
wrong["width"] = 704
issues = meta_issues(meta1, wrong)
check("画幅变了 → 报出来", any("宽度" in i for i in issues), str(issues))

wrong2 = dict(meta2)
wrong2["segment_index"] = 5
check("段号不连续 → 报出来", any("上一段号" in i for i in meta_issues(meta1, wrong2)))

meta2b = make_meta(
    chain_id="mv", plan=p2b, width=480, height=864, steps=4, sampler="dual_clock_euler",
    scheduler="native_flow", shift_video=12.0, shift_audio=3.0, task_type="auto",
    audio_mode="lock_source", prompt="p2", seed=2,
)
check("末段窗口不同也能接（时间轴连续即可）", meta_issues(meta1, meta2b) == [], str(meta_issues(meta1, meta2b)))

wrong3 = dict(meta2b)
wrong3["timeline_start_frames"] = 124
check("时间轴出现空档 → 报出来", any("时间轴不连续" in i for i in meta_issues(meta1, wrong3)), str(meta_issues(meta1, wrong3)))

# ---------------------------------------------------------------- 路径与状态

check("链名里的斜杠被折成下划线", sanitize_name("a/b c") == "a_b_c", sanitize_name("a/b c"))
check("中文链名保留", sanitize_name("舞蹈 mv 01") == "舞蹈_mv_01", sanitize_name("舞蹈 mv 01"))
check("空链名给默认值", sanitize_name("   ") == "chain")

root = resolve_chain_root("mv-01", "/tmp/out")
check("相对链名落在 output/rs_h3_chain 下", root == Path("/tmp/out/rs_h3_chain/mv-01"), str(root))
abs_root = resolve_chain_root("D:/chains/mv", "/tmp/out")
check("绝对路径原样使用", abs_root == Path("D:/chains/mv"), str(abs_root))

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    write_json_atomic(meta_path(base, 1), {"segment_index": 1})
    context_path(base, 1).write_bytes(b"x")
    check("meta + context 都在才算完成", list_segments(base) == [1], str(list_segments(base)))

    write_json_atomic(meta_path(base, 2), {"segment_index": 2})
    check("缺 context 的半截状态不算完成", list_segments(base) == [1], str(list_segments(base)))

    check("clear_segment 删掉了目录", clear_segment(base, 1) and not segment_dir(base, 1).exists())
    check("删完列表为空", list_segments(base) == [])

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    write_json_atomic(
        meta_path(base, 1),
        {"segment_index": 1, "render_frames": 124, "context_frames": 22, "timeline_start_frames": 0},
    )
    check("第 2 段起点 = 0 + 124 − 22 = 102", chain_timeline_start(base, 2, 22, 141) == 102)
    write_json_atomic(
        meta_path(base, 2),
        {"segment_index": 2, "render_frames": 141, "context_frames": 22, "timeline_start_frames": 102},
    )
    check("第 3 段起点 = 102 + 141 − 22 = 221", chain_timeline_start(base, 3, 22, 124) == 221)
    check("没有上一段 meta 时退化成固定步长公式", chain_timeline_start(base, 9, 22, 124) == 8 * 102)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
