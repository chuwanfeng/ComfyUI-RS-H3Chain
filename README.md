# ComfyUI-RS-H3Chain

MiniMax H3 的**分段续接**节点：一次执行只跑一段，段间上下文落盘，下一段读回来钉住。

**不依赖任何第三方节点包**：条件编码、跨段引导、双时钟 shift、
采样、解码全部走 ComfyUI 核心（`comfy.sample` / `comfy.nested_tensor` / `node_helpers` /
`comfy.model_sampling` / `comfy_extras.nodes_minimax_h3`）。

---

## 为什么要有它

现成的两条长视频路线（T8 InNodeLoop、JZL 无限时长）都是「一次执行跑完整条链」，
由此带来三个绕不过去的限制：

| 问题 | 一条链跑一次 | 本节点（一段一次） |
|---|---|---|
| 中间一段不满意 | 只能从该段截断，后面全部重做 | 删掉第 k 段起的状态，从 k 重跑；前面不动 |
| 跑几分钟的长视频 | 风险集中在一个 prompt 里，中途崩了整条重来 | 任一段失败只重试该段 |
| 先跑 2 段、后面接上 | 不支持（合约里含总时长与提示词，改了直接拒绝） | 状态在盘上，随时继续 |
| 内存 | TE(14.9GB)+扩散(21.6GB)+VAE 全程驻留，32GB 机器 0.4MP 跑两段就 OOM | 一次执行一段，ComfyUI 能在 prompt 之间卸载不用的模型 |

段间**仍然是 22 帧 latent 运动上下文**，不是退化成"上一帧当首帧"。

---

## 两个参考实现，各取所长

读完 T8 与 JZL 的源码后的结论：

| 维度 | T8 InNodeLoop | JZL 无限时长 | 本节点取 |
|---|---|---|---|
| 段间引导 | 自己 patch `extra_conds`，每个 token 一个 keyframe + 帧偏移 | 官方 `minimax_keyframes`，一个多帧 keyframe | **JZL**：官方字段，不 patch 模型，升级不易碎；也不会和其他 keyframe 类节点互斥 |
| 视频/音频 shift | 自研双时钟采样（12 / 3） | 不设，用模型默认 | **T8 的语义**，用官方 `MiniMaxH3SigmaShift` 的同一实现（换 `model_sampling` + `transformer_options`） |
| 采样 | 自研循环（带 preview / 断点回调） | `comfy.sample.sample` | 官方路径（`comfy.sample.sample`） |
| 裁剪 | 像素域裁衔接窗口 | 像素域裁 + settle（默认 12 帧） | 像素域裁衔接窗口；**settle 暂未实现**（它需要额外的时间轴记账，见下） |
| 上下文 | manifest + candidate.context.safetensors | 只在内存里，不落盘 | 落盘的 `video_tail` / `audio_tail`（本包自己的格式，带 sha256） |
| 代码形态 | 一个包 + orchestration / av runtime / delivery 多模块 | 一个 6000 行的巨型节点 | 4 个模块，纯函数可测 |

JZL 在拼接时用 settle（窗口 + 12 帧过渡丢弃）吸掉"模仿引导 → 启动新动作"的 hold→pop。
本节点暂时把 settle 设为 0：每段独立提交时，丢额外帧会让"段的时间轴"和"音频窗口"错位
（要多记一个偏移，跨段会累积）。等 P1 接缝实测后再决定是否按段记账引入。

---

## 安装

把整个目录放进 `ComfyUI/custom_nodes/`：

```
ComfyUI/custom_nodes/ComfyUI-RS-H3Chain/
  __init__.py
  rs_chain/{state,media,h3,nodes}.py
  selftest.py
  examples/RS_H3Chain_segment.workflow.json   # UI 格式：拖进 ComfyUI 直接打开
  examples/RS_H3Chain_segment.api.json        # API 格式：给 RS-Canvas / 脚本提交用
  examples/RS_H3Chain_segment.config.json
```

`*.workflow.json` 是画布格式（`nodes` / `links`），在 ComfyUI 里「Workflow → Open」或直接拖进窗口即可；
`*.api.json` 只有 `class_type`/`inputs`，是给 API 提交用的，画布打不开。
UI 示例里 `render_window_frames` 由「本段时长（秒）」经 `ComfyMathExpression`
（`max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17`）自动换算成 17k+5 帧。

**没有第三方依赖**。要求 ComfyUI 自带 MiniMax H3 支持（`comfy_extras/nodes_minimax_h3.py`
里要有 `MiniMaxH3AddGuide` / `MiniMaxH3ReferenceToVideo` / `MiniMaxH3SigmaShift`）。

装完先自检。**必须用 ComfyUI 自己的解释器**（整合包是 `python_embeded/python.exe`），
在节点目录里直接 `python selftest.py` 是验不到的——那里没有 `comfy`：

```bash
# ① 在 ComfyUI 根目录
python custom_nodes/ComfyUI-RS-H3Chain/selftest.py
# ② 或指路（本机另装了 ComfyUI）
COMFYUI_ROOT=D://ComfyUI python selftest.py
COMFYUI_DIR=/workspace/ComfyUI python selftest.py     # 云容器里启动脚本用的就是这个名字
```

根目录判据是 `main.py` + `comfy/` + `custom_nodes/` 三件套（不看 `folder_paths.py` 的位置——
不同分支/封装会挪它，之前就因为这条误报"未找到"）。找不到时它会逐条打印试过的路径和缺什么。

退出码：`0` = 全部就绪；`1` = 有具体项没通过（会逐条列出）；`2` = 有项没验到
（找不到 ComfyUI，或用错了解释器）。最后一行是
`✓ 核心 API、官方节点、帧数换算、上下文读写全部就绪` 才算能跑。

**看到 `⚠ 跳过 —— 当前解释器看不到 GPU` 时**：几乎一定是用了另一个 python。
`comfy.sample` 一 import 就会初始化 CUDA，解释器看不到 GPU 就抛
`Found no NVIDIA driver`，并连带把"官方 H3 节点""rs_chain.h3"都标成失败。
自检会直接把这台机器上 ComfyUI 的解释器路径打出来（`<ComfyUI>/../venv/bin/python`
或 `python_embeded/python.exe`），照它跑。

---

## 段几何（唯一需要理解的数学）

H3 视频 latent 的 T 轴不均匀：token k 覆盖 `FRAME_PER_TOKEN[k%5]=(1,4,4,4,4)` 帧，5 token = 17 帧。

- **渲染窗口必须是 `17k+5`**：56 / 90 / 124 / 158 / 192 / 226 / 260 / 277 / 294 / 311 / 328 / 345 / 362 / 379；
  官方训练区间约 124–362（5–15 秒）；**续接段要多带 22 帧衔接**，所以「每段 15 秒成片」的窗口是 379；
- **上下文窗口必须是 `17m+5`**：5 / 22 / 39（22 = 官方 Motion Context 基线，7 个 latent token）；
- `窗口 − 上下文 = 17(k−m)` ⇒ 尾窗起点永远落在 token 边界 → **相位自动对齐**；
- 第 1 段：时间轴 0 起，不裁头，产出 = 窗口帧；
- 第 k 段：窗口起点 = 上一段起点 + 上一段窗口 − 上下文（**从上一段 meta 推**，不是写死的
  `(k−1)×(窗口−上下文)`，否则末段换长度后后面全错）；段首 `上下文` 帧是被钉住的上一段尾巴，**解码后裁掉**；
- 音频与画面同一坐标系：本段音频从**窗口起点**取、同样裁掉 `上下文` 帧。
  （早先这里多减了一个上下文，交付音轨整体早 0.917 秒，接缝处像"音乐跳回去一下"——已修）
- **成片总帧数恒为 `17k+5`**：124/22 的固定两段只有 226 帧（9.42s），10 秒（240 帧）不可达。
  要对准目标就让末段换长度（124 + 141 − 22 = 243 帧 = 10.125s），时间轴由 meta 续上，安全。

默认 124 帧 + 22 帧上下文 ⇒ 每段成片 102 帧（4.25 秒），第 1 段 124 帧（5.17 秒）。

---

## 节点：RS - H3 分段续接（单段）

必填：`model / clip / video_vae / audio_vae`、`chain_dir`、`segment_index`、`prompt`、`seed`、
`width / height`（32 的倍数）、`render_window_frames`、`context_frames`、`steps`、`cfg`、
`shift_video / shift_audio`、`sampler_name / scheduler`、`ref_scale`。

可选：

- `drive_audio`：整首歌，节点自己按时间轴取窗；**并且把整段窗口音频编码成 `audio_latent` 钉住**
  —— 不钉的话模型会自编一条音轨、视频跟着那条走，我们再 mux 上真歌，动作与鼓点就毫无关系
  （不报错，但 MV 全废）。接了它就用源音频当交付音轨；
- `ref_image_0..8`：参考图，**顺序即编号**（`ref_image_0` → `<Picture 1>`），最多 9 张；
- `ref_video_0/1/2`：参考视频（IMAGE 批次，2–15s），自动吸附到 5/22/39… 帧；
- `ref_video_audio_0/1/2`：同序号参考视频的原声，编号紧挨在 `<Video k>` 前面；
- `ref_audio_0/1/2`：独立参考音频（音色/环境声参考），最多 3 个；
- `sigmas`：自定义 Sigmas（SIGMAS）。接了它就走 `comfy.sample.sample_custom`，
  `scheduler` / `steps` / `shift_video` / `shift_audio` 全部不参与——`video_pika_h3_flex_ref2va`
  必须用 `KSamplerSelect=lcm` + `ManualSigmas` 这条路线。

音频的三种用法（别混）：

| 输入 | 语义 | 是否交付 | 代价 / 限制 |
|---|---|---|---|
| `drive_audio` | **时间轴音轨**：按段窗口切出、编码成 `audio_latent` 钉在段首；成片直接用它 | 是 | 动作被钉死在这首歌上；每段都要带一个整窗音频 latent |
| `ref_audio_0..2` | **参考音色/环境**：作为 `minimax_refs` 的条件，不占时间轴 | 否（只影响生成） | 不能保证逐拍同步；更像"给模型听一段参考" |
| 提示词描述音乐 | 纯语义（"follow <Audio 1>" / 描述鼓点） | 否 | 最便宜，但节拍、音色都不可控 |

画布面板有一个「音频用途（第 1 个音频给谁）」：
- **驱动音频（时间轴音轨）** → 第 1 个音频进 `drive_audio`，其余进 `ref_audio_0..2`；
- **仅参考音频** → 第 1 个音频也进 `ref_audio_0`，`drive_audio` 完全不接（成片用模型生成的音轨）。

### 关键帧：在成片某个时间点钉一张图 / 一小段

| 输入 | 说明 |
|---|---|
| `keyframe_image_1` | 关键帧图。一批 ≥5 帧就当成小片段（向下吸附到 5/22/39…），少于 5 帧只用第一张；图会**等比铺满画布后居中裁切**（官方 AddGuide 的 `center` crop） |
| `keyframe_time_1` | 放在**成片时间轴**的第几秒（-1 = 不用）。默认 -1 |
| `keyframe_image_2/3` / `keyframe_time_2/3` | 第 2、3 个槽位，规则相同 |

**0 秒关键帧就是首帧**：`keyframe_time_1=0` + 一张图，等价于原来的 `first_frame`；
第 1 段有效，第 2 段起 0 秒落在段首衔接窗口里（会被裁掉），节点会明确报错并给出本段最早可用的秒数。
所以「首帧只在第 1 段」这件事现在由**时间**表达，而不是单独的输入口。

用绝对秒而不是"本段第几帧"：段的时间轴起点由段几何固定（`timeline_start = 上一段起点 + 上一段窗口 − 上下文`），
不随内容变化，所以"第 8 秒放这张图"在整条链里始终指同一个时刻。

**单张图 = 单帧锚点**（官方 AddGuide 规则：<5 帧只用第一张，锚 1 个 latent token ≈ 1~4 帧）。
它是一根"软牵引"：t=0 时基本等于"从这张图开始"（实测首帧与关键帧近乎逐像素一致），
中段则要和前后文竞争，通常只能把动作往那张图的方向带，不会精确停住。

**别用重复帧去加强中段锚点**：把同一张图 `RepeatImageBatch` 成 22 帧（7 token）后，
钉在 3.0s 会让 3.5s 附近**过曝发白**（实测两帧亮度 122/121，正常是 89）——
cond token 与目标 token 在同一时间点混在一起。画布上的「关键帧保持（帧）」默认就是 1，
选 5/22 前先接受这个代价。要让内容硬切换，正确做法是**切成新段**（新 `chain_dir` / `segment_index=1`，
把那张图当首帧），而不是在中段硬钉。

两条硬规则（不满足会明确报错，不会静默丢帧）：

- 换算到本段后必须 ≥ `head_trim`：第 2 段起前 22 帧是上一段的尾巴，会被裁掉，钉在那里等于没钉；
- `起始帧 + 片段帧数 ≤ 渲染窗口`：放不下就报错，并把"本段最晚可放到几秒"算给你。

### 换人物参考：改一张图就是换人

每个段节点有**自己的** `ref_image_0..8`、`ref_video_0..2`、`ref_audio_0..2`，换段换图即可，
不需要重建链、也不需要改提示词（提示词里的 `<Picture 1>` 永远指"本段接的第一张图"）。

要把人物保真度再提高一档：调 `ref_scale`（1.0–5.0，面积倍率）。参考 token 参与每个采样步，
调高更慢但更像。`ref_scale = 1.0` 时是官方的 `match` 行为——**只缩不放**；大于 1 才按面积倍率放大
（短边上限 2048）。

**一个必须知道的行为**：第 k 段的段首 22 帧是上一段的 latent 尾巴，所以**中途换人会有约 0.9 秒的过渡**
（模型会从旧脸morph到新脸）。两种用法：

- 想要"变身/换装"效果 → 直接用，过渡正好是转场；
- 想要干净硬切 → 换人处**断开成新链**（新 `chain_dir`，`segment_index=1`），成片里硬剪。

`RS_H3ChainManage` 的 `status` 会打印每段的参考图指纹（`参考a1b2c3`），参考变了会标
`（本段换了参考）`，不用猜是哪一段换的人。

两个开关：

- `force_rerun`：本段已有状态时允许覆盖（默认拒绝，防止误覆盖已验收的镜头）；
- `allow_no_prev`：第 k 段找不到上一段状态时仍然生成（接缝会跳，只用于对比实验）。

输出：`frames`（IMAGE，已裁头）、`audio`（AUDIO，已同步裁）、`report`（JSON 文本）。
典型接法：`frames + audio → CreateVideo(fps=24) → SaveVideo`。

### 状态目录

```
<chain_dir>/
  seg_00001/tail.context.safetensors   # video_tail [1,24,7,h,w] + audio_tail [1,32,2,37]
  seg_00001/meta.json                  # 画幅/窗口/帧率/提示词哈希/种子/耗时
```

`meta.json` 只在**解码成功之后**才写，所以"半截状态"不会被当成可用；
上下文文件带 sha256 与段号，被改过、属于别的链、窗口不一致都会被明确拒绝。

### 画布接线（RS-Canvas）

| 画布上的口 | 落到节点 | 说明 |
|---|---|---|
| 参考图 | `ref_image_0..8` | 按连线顺序 = `<Picture 1..9>`；最多 9 张 |
| 参考视频 | `ref_video_0..2` + `ref_video_audio_0..2` | 视频画面与原声成对进入 |
| 参考音频 | `ref_audio_0..2` | 独立音色/环境参考，最多 3 个 |
| 驱动音频 | `drive_audio` | 整首歌，节点按时间轴取窗并钉住 |
| 帧引导 | `keyframe_image_1/2/3` | 第 i 张 → 第 i 个关键帧；秒数由面板「关键帧N 时间」注入 |
| 时长（秒） | `render_window_frames` | 由 `chain_plan` 按 `17k+5` 换算（末段可短一点对准目标） |

一条链 = 多个视频节点：`chain_dir` 相同、`segment_index` 递增（1,2,3…）。
目标 10 秒 → 自动排成 124 + 141 帧 = 243 帧 = 10.125s（240 帧不可达）。

### 重跑与加长

| 想做什么 | 做法 |
|---|---|
| 重跑第 5 段 | `RS_H3ChainManage(reset_from=5, confirm=true)` → 依次跑 5、6、…（6 之后必须重跑：它们吃的是 5 的尾巴） |
| 只改第 5 段的提示词 | 同上；1–4 不动 |
| 接着加长 | 直接跑 `segment_index=17` 起，不需要改任何"总时长" |
| 换画幅/窗口 | 画幅变了 → **必须换新的 `chain_dir`**（旧 latent 尾巴对新条件没有意义，节点会明确拒绝）。窗口可以在段间变（末段对准目标时长），时间轴会从上一段 meta 续上 |

---

## 已知限制

1. **每段一次提交**，模型在 TE 与扩散之间来回切换，约多 30–60 秒/段；
2. `settle`（额外丢过渡帧）未实现，接缝处可能有约 0.5 秒的"启动感"；P1 实测后再定；
3. 参考视频会按 768 短边画布缩放，长度超过本段渲染窗口的部分会被裁掉（至少 5 帧）；
4. 音频锚点是"整段钉住"：模型生成的音轨会被丢弃，只用作画面同步信号；
5. 关键帧每个段节点 3 个槽位；更多请把关键帧拆到相邻段（段的时间轴是连续的，效果等价）；
6. 没有语义桥（`MiniMaxH3Mod` 那套 RefMod / 语义桥是另一个维度，可另行接）。
