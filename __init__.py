"""RS - H3 分段续接（ComfyUI-RS-H3Chain）。

独立包，不修改任何第三方节点，**不依赖任何第三方节点包**：
条件编码、跨段引导、双时钟 shift、采样、解码全部走 ComfyUI 核心
（`comfy.sample` / `comfy.nested_tensor` / `node_helpers` /
`comfy.model_sampling` / `comfy_extras.nodes_minimax_h3`）。
本包只负责「一次执行一段 + 段间上下文落盘 + 血缘校验」。
"""

from __future__ import annotations

import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = None

try:
    from .rs_chain.nodes import RS_H3ChainManage, RS_H3ChainSegment

    NODE_CLASS_MAPPINGS = {
        "RS_H3ChainSegment": RS_H3ChainSegment,
        "RS_H3ChainManage": RS_H3ChainManage,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "RS_H3ChainSegment": "RS - H3 分段续接（单段）",
        "RS_H3ChainManage": "RS - H3 链维护（状态/重置）",
    }
    print("[RS-H3Chain] ✅ 已加载 2 个节点：RS_H3ChainSegment / RS_H3ChainManage")
except Exception:  # noqa: BLE001
    print("[RS-H3Chain] ❌ 节点注册失败（其余节点不受影响）：")
    print(traceback.format_exc())

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
