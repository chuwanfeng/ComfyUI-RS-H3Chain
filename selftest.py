"""检查本节点依赖的**核心 API**是否都在。

两种跑法：

    # ① 在 ComfyUI 根目录，用它自己的 python（推荐）
    python custom_nodes/ComfyUI-RS-H3Chain/selftest.py

    # ② 不在 ComfyUI 目录里时，指路给它
    COMFYUI_ROOT=D:\\ComfyUI python selftest.py
    python selftest.py --comfy-root D:\\ComfyUI

找不到 ComfyUI 时**只跑离线部分**（帧数换算 + 上下文读写），并给出该在哪儿跑；
这不是失败，是环境没指对。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OK = "✓"
BAD = "✗"
SKIP = "⚠"

# 这些错误说明"当前解释器看不到 GPU"，不是代码缺东西：
# 在服务器的 pyenv python 里 import comfy.sample 会在 CUDA 初始化时抛这个。
_CUDA_ERROR_MARKERS = (
    "Found no NVIDIA driver",
    "no CUDA-capable device",
    "Torch not compiled with CUDA",
    "CUDA driver version is insufficient",
    "no kernel image is available",
)


def _is_cuda_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker.lower() in text.lower() for marker in _CUDA_ERROR_MARKERS)


def find_comfy_python(root: Path) -> Path | None:
    """猜 ComfyUI 启动时用的解释器（Launch 脚本 source 的那个）。

    判据：venv/bin/python 或 python_embeded/python.exe，且存在。
    """
    candidates = [
        root.parent / "venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
        root / ".venv" / "bin" / "python",
        root / "python_embeded" / "python.exe",
        root.parent / "python_embeded" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _looks_like_comfy(root: Path) -> bool:
    """ComfyUI 根目录的判据用「三件套」而不是单个文件名。

    之前要求 `comfy/__init__.py` + `folder_paths.py` 都在，结果在真实部署里漏判：
    只要其中之一被改名/挪位置（不同分支、不同封装），就报"未找到"，而目录其实就在脚下。
    `main.py` + `comfy/` + `custom_nodes/` 是任何可运行 ComfyUI 都有的。
    """
    return (
        (root / "main.py").is_file()
        and (root / "comfy").is_dir()
        and (root / "custom_nodes").is_dir()
    )


def _why_not(root: Path) -> str:
    missing = [
        name
        for name, ok in (
            ("main.py", (root / "main.py").is_file()),
            ("comfy/", (root / "comfy").is_dir()),
            ("custom_nodes/", (root / "custom_nodes").is_dir()),
        )
        if not ok
    ]
    return "缺 " + "、".join(missing) if missing else "看起来像 ComfyUI"


def find_comfy_root() -> Path | None:
    """按优先级找 ComfyUI 根目录：--comfy-root → 环境变量 → cwd 及上溯 → 本包上溯。"""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--comfy-root" and i + 1 < len(args):
            candidate = Path(args[i + 1]).expanduser().resolve()
            return candidate if _looks_like_comfy(candidate) else None
        if arg.startswith("--comfy-root="):
            candidate = Path(arg.split("=", 1)[1]).expanduser().resolve()
            return candidate if _looks_like_comfy(candidate) else None

    for var in ("COMFYUI_ROOT", "COMFYUI_DIR"):
        env = os.environ.get(var, "").strip()
        if env:
            candidate = Path(env).expanduser().resolve()
            if _looks_like_comfy(candidate):
                return candidate

    bases = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    bases += [HERE, *HERE.parents]
    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        if _looks_like_comfy(base):
            return base
        # 常见的 portable 结构：ComfyUI_windows_portable/ComfyUI
        nested = base / "ComfyUI"
        if _looks_like_comfy(nested):
            return nested

    # 最后一条路：如果当前解释器其实已经能 import comfy（例如装了 PYTHONPATH），
    # 直接反推根目录——比让用户猜哪里没配对更快。
    try:
        import comfy as comfy_pkg

        candidate = Path(comfy_pkg.__file__).resolve().parent.parent
        if _looks_like_comfy(candidate):
            return candidate
    except Exception:  # noqa: BLE001
        pass
    return None


def offline_checks() -> list[str]:
    """不依赖 ComfyUI 的部分：帧数换算 + 上下文文件读写。"""
    problems: list[str] = []

    print("\n== 帧数换算（离线） ==")
    try:
        from rs_chain import state

        checks = [
            ("124 帧吸附后仍是 124", state.align_frame_count(124) == 124),
            ("124 帧 = 37 个 latent token", state.video_latent_t(124) == 37),
            ("124 帧 → 207 个音频步", state.temporal_shape(124) == (124, 37, 207)),
            ("37 token 反算回 124 帧", state.pixel_frames_from_latent_t(37) == 124),
            ("22 帧上下文 = 7 token", state.video_latent_t(22) == 7),
            ("段长 120 会被吸附到 124", state.align_frame_count(120) == 124),
        ]
        for name, ok in checks:
            if ok:
                print(f"  {OK} {name}")
            else:
                problems.append(name)
                print(f"  {BAD} {name}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"帧数换算自检失败：{exc}")
        print(f"  {BAD} {exc}")

    print("\n== 上下文读写（离线） ==")
    try:
        import tempfile

        import torch

        from rs_chain import media
        from rs_chain.state import context_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = torch.zeros((1, 24, 7, 30, 54), dtype=torch.float32)
            audio = torch.zeros((1, 32, 2, 37), dtype=torch.float32)
            media.write_context(
                context_path(root, 2), video, audio,
                chain_id="selftest", segment_index=1, context_frames=22,
                audio_overhang=0.33, source_frames=124,
            )
            back = media.read_context(
                context_path(root, 2), chain_id="selftest", source_index=1, context_frames=22
            )
            if back["video_tail"].shape == video.shape and back["audio_tail"].shape == audio.shape:
                print(f"  {OK} 写入/读回/校验通过：video{tuple(video.shape)} audio{tuple(audio.shape)}")
            else:
                problems.append("上下文读写形状不符")
                print(f"  {BAD} 形状不符")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"上下文读写自检失败：{exc}")
        print(f"  {BAD} {exc}")

    return problems


def comfy_checks() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    skipped: list[str] = []

    print("\n== ComfyUI 核心 API ==")
    probes = [
        ("comfy.nested_tensor", "NestedTensor"),
        ("comfy.sample", "sample"),
        ("comfy.sample", "prepare_noise"),
        ("comfy.model_sampling", "ModelSamplingAV"),
        ("comfy.model_sampling", "CONST"),
        ("node_helpers", "conditioning_set_values"),
        ("comfy.utils", "common_upscale"),
    ]
    for module_name, attr in probes:
        try:
            mod = __import__(module_name, fromlist=[attr])
            getattr(mod, attr)
            print(f"  {OK} {module_name}.{attr}")
        except Exception as exc:  # noqa: BLE001
            if _is_cuda_error(exc):
                skipped.append(f"{module_name}.{attr}（当前解释器没有 GPU）")
                print(f"  {SKIP} {module_name}.{attr}: 跳过 —— 当前解释器看不到 GPU")
            else:
                problems.append(f"{module_name}.{attr} 缺失：{exc}")
                print(f"  {BAD} {module_name}.{attr}: {exc}")

    print("\n== 官方 H3 节点 ==")
    try:
        import comfy_extras.nodes_minimax_h3 as h3nodes

        for cls in ("MiniMaxH3AddGuide", "MiniMaxH3ReferenceToVideo", "MiniMaxH3SigmaShift"):
            if hasattr(h3nodes, cls):
                print(f"  {OK} {cls}")
            else:
                problems.append(f"官方节点 {cls} 不存在（ComfyUI 版本太旧）")
                print(f"  {BAD} {cls}")
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_error(exc):
            skipped.append("comfy_extras.nodes_minimax_h3（当前解释器没有 GPU）")
            print(f"  {SKIP} 跳过 —— 当前解释器看不到 GPU")
        else:
            problems.append(f"comfy_extras.nodes_minimax_h3 导入失败：{exc}")
            print(f"  {BAD} 导入失败：{exc}")

    print("\n== 本包模块 ==")
    try:
        from rs_chain import h3  # noqa: F401

        print(f"  {OK} rs_chain.h3 可导入")
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_error(exc):
            skipped.append("rs_chain.h3（当前解释器没有 GPU）")
            print(f"  {SKIP} 跳过 —— 当前解释器看不到 GPU")
        else:
            problems.append(f"rs_chain.h3 导入失败：{exc}")
            print(f"  {BAD} {exc}")

    print("\n== 宿主 H3 布局语义 ==")
    try:
        from rs_chain import layout_contract

        layout_contract.ensure()
        print(f"  {OK} 分数/负 keyframe 索引与音频窗位置符合预期")
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_error(exc):
            skipped.append("rs_chain.layout_contract（当前解释器没有 GPU）")
            print(f"  {SKIP} 跳过 —— 当前解释器看不到 GPU")
        else:
            problems.append(f"布局校验失败：{exc}")
            print(f"  {BAD} {exc}")

    return problems, skipped


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"cwd   : {Path.cwd()}")

    root = find_comfy_root()
    print(f"ComfyUI 根目录：{root if root else '未找到'}")
    if root is None:
        tried = [Path.cwd().resolve(), *Path.cwd().resolve().parents[:3], HERE.parent]
        for candidate in tried:
            print(f"  · 试过 {candidate}：{_why_not(candidate)}")

    if root is not None:
        sys.path.insert(0, str(root))
        if str(root) not in os.environ.get("PYTHONPATH", ""):
            os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("\n== 环境 ==")
    for name in ("torch", "safetensors"):
        try:
            mod = __import__(name)
            print(f"  {OK} {name} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {BAD} {name}: {exc}")

    has_comfy = root is not None
    if has_comfy:
        try:
            import comfy  # noqa: F401
            import folder_paths  # noqa: F401

            print(f"  {OK} comfy / folder_paths")
        except Exception as exc:  # noqa: BLE001
            has_comfy = False
            print(f"  {BAD} comfy/folder_paths 仍无法导入：{exc}")

    problems = offline_checks()
    skipped: list[str] = []
    if has_comfy:
        comfy_problems, skipped = comfy_checks()
        problems += comfy_problems

    print("\n== 结论 ==")
    if not has_comfy:
        print("  ! 只完成了离线部分（帧数换算 + 上下文读写），ComfyUI 侧的 API 没验。")
        if root is not None:
            print("    已找到 ComfyUI 目录，但当前 python 导入不到 comfy ——")
            print("    大概率用的不是 ComfyUI 那个解释器（整合包是 python_embeded\\python.exe）。")
        print("    请在 ComfyUI 根目录、用它自己的 python 再跑一次：")
        print("      <ComfyUI>/python_embeded/python.exe custom_nodes/ComfyUI-RS-H3Chain/selftest.py")
        print("    或指定路径：")
        print("      COMFYUI_ROOT=D:\\ComfyUI python selftest.py")
        print("    本机没有 ComfyUI 时，这一步只能在服务器上做。")
        if problems:
            for p in problems:
                print(f"  {BAD} {p}")
            return 1
        return 2

    if skipped:
        print(f"  {SKIP} 有 {len(skipped)} 项没验到：当前解释器看不到 GPU，")
        print("    这在服务器上几乎总是「用错了 python」——ComfyUI 启动时 source 的是另一个 venv。")
        candidate = find_comfy_python(root) if root else None
        if candidate:
            print(f"    这台机器上 ComfyUI 用的解释器大概是：{candidate}")
            print(f"    用它再跑：{candidate} custom_nodes/ComfyUI-RS-H3Chain/selftest.py")
        else:
            print("    用启动脚本里 source 的那个 python 再跑一次。")
        for item in skipped:
            print(f"    · {item}")

    if problems:
        for p in problems:
            print(f"  {BAD} {p}")
        return 1
    if skipped:
        return 2
    print(f"  {OK} 核心 API、官方节点、帧数换算、上下文读写全部就绪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
