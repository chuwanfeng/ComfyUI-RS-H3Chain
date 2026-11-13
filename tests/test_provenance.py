"""链血缘校验的纯函数测试：`python tests/test_provenance.py`（不需要 torch / ComfyUI）。

为什么单测：血缘错了不会报错，只会让第 7 段接在旧的第 5 段后面；
画面看起来完全正常，直到你发现某个人物/场景回到了几段之前。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_chain.state import provenance_issues  # noqa: E402

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


def has_issue(issues: list[str], keyword: str) -> bool:
    return any(keyword in issue for issue in issues)


# 正常：上一段记录的自身哈希与实际一致，且消费的是当前上上段
prev = {"context_sha256": "aaa", "prev_context_sha256": "bbb", "segment_index": 2}
check("血缘完整且一致 → 无问题",
      provenance_issues(prev, "aaa", {"segment_index": 1}, "bbb") == [])

# 上一段文件被改过
check("上一段自身哈希不符 → 报错",
      has_issue(provenance_issues(prev, "zzz", {"segment_index": 1}, "bbb"), "不一致"))

# 上上段被重跑：上一段记录的是旧哈希
check("上上段被重跑 → 报错",
      has_issue(provenance_issues(prev, "aaa", {"segment_index": 1}, "ccc"), "旧的上上段"))

# 上上段被删
check("上上段被删 → 报错",
      has_issue(provenance_issues(prev, "aaa", None, None), "不存在"))

# 旧格式：两个字段都没有 → 放行（历史链不能被拦）
legacy = {"segment_index": 2}
check("旧格式 meta → 放行",
      provenance_issues(legacy, "whatever", None, None) == [])

# 第 2 段：没有 prev_context_sha256 是正常的（上一段就是第 1 段）
first = {"context_sha256": "aaa"}
check("第 2 段消费第 1 段 → 放行",
      provenance_issues(first, "aaa", None, None) == [])

# 上一段自身哈希为空（旧文件）但记录了上上段 → 仍需校验上上段
mixed = {"prev_context_sha256": "bbb"}
check("只有上上段哈希时仍能发现重跑",
      has_issue(provenance_issues(mixed, "", {"segment_index": 1}, "ccc"), "旧的上上段"))

print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)
