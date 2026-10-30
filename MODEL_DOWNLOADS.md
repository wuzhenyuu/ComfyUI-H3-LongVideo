# 模型下载与本地安装

本插件负责音频分析、分段审核和顺序调度；H3 视频推理由 ComfyUI 及工作流中的模型节点执行。Git 仓库不包含模型权重。当前推荐 [07 MV 本地／RH](examples/07_H3MV_本地与RH.json)与 [13 口播本地／RH](examples/13_H3口播_独立窗口_本地展开.json)，历史 [04 一体化](examples/04_H3长视频_单节点一体化.json)单独列出，不能混用其默认模型。链接及文件大小核对日期为 **2026-10-30（北京时间）**。

`<ComfyUI>` 表示你的 ComfyUI 根目录，例如本机为 `F:/comfyui`。表中的 GB 为十进制下载大小，GiB 为二进制磁盘大小；它们都不等于运行时显存需求。大模型下载前应预留临时文件与缓存空间。

## 先确定需要哪些模型

| 用途 | 是否需要 | 获取方式 | 默认本地位置 |
| --- | --- | --- | --- |
| Faster-Whisper `large-v3-turbo`：识别歌词、辅助 MV 切点 | MV 音频分析需要；新版口播使用 CPU 停顿检测，不需 ASR 权重 | 首次 MV 分析时缺少完整缓存才自动下载；可提前下载 | `<ComfyUI>/models/faster-whisper/` 内的 Hugging Face 缓存 |
| HDemucs：从混合音频分离人声 | MV／口播选择“提取人声”时需要；“直接使用”可跳过 | 缺失时自动下载，支持断点续传 | `<ComfyUI>/models/H3LongVideo/hdemucs_high_trained.pt` |
| MOSS-Music-8B-Instruct：本地音乐情绪分析 | 可选；点击音乐情绪分析前准备 | 手动下载完整固定版本目录，推理不自动下载 | `<ComfyUI>/models/moss_music/MOSS-Music-8B-Instruct/` |
| H3 主模型、文本编码器、视频 VAE、音频 VAE | 本地视频生成需要；07／13 所选主模型不同，见第 4 节 | 手动下载，在工作流加载器中选择 | `models/diffusion_models`、`models/text_encoders`、`models/vae`，见下表 |
| LoRA 与 latent upscaler | 按所选画布准备；07 与 13 的文件不同 | 手动下载 | `models/loras/`、`models/latent_upscale_models/` |
| 分镜提示词扩写模型 | 自动扩写时需配置可用服务；也可走手写通道 | 使用自己的 OpenAI 兼容 API 或已登录的 Codex CLI；本插件不下载该模型 | 服务端管理，无固定本地权重目录 |

安装插件依赖应使用**启动 ComfyUI 的同一个 Python**，无需另建 Python 环境。例如 Windows PowerShell：

```powershell
# 替换为你启动 ComfyUI 时实际使用的 python.exe；便携版通常在 python_embeded 中。
$ComfyPython = 'F:/comfyui/venv/Scripts/python.exe'
& $ComfyPython -m pip install -r 'F:/comfyui/custom_nodes/ComfyUI-H3-LongVideo/requirements.txt'
```

以上 Python 路径只是示例，以你实际启动命令为准。本插件沿用 ComfyUI 已安装且彼此匹配的 PyTorch／Torchaudio；不要照搬其他模型项目的独立环境安装命令，盲目降级或替换现有 Torch。另需 `ffmpeg`、`ffprobe` 在该服务环境可调用。安装后在方便时重启 ComfyUI 并刷新浏览器；仅复制文件不能让正在运行的旧服务自动加载新后端。

## 1. Faster-Whisper：音频分析时按需下载

当前 `run_asr()` 固定使用 `large-v3-turbo`，在 ComfyUI 同一 Python 的短生命周期子进程中运行。旧工作流保留的 `asr_python`、`asr_model` 字段不用于选择另一个 Python 或模型。

- 模型主页：[Faster-Whisper large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)。该地址目前重定向到维护者的 [dropbox-dash/faster-whisper-large-v3-turbo](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo)；本插件依赖的 `faster-whisper==1.2.1` 仍使用前一个仓库 ID。
- 文件目录：[完整文件列表](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo/tree/main)。应保留 `model.bin`、`config.json`、`tokenizer.json`、`vocabulary.json`（或受支持的 `vocabulary.txt`），同时保留发行包的 `preprocessor_config.json`。
- 权重直链：[model.bin](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/model.bin)，权重为 1,617,884,929 字节，约 1.62 GB；**只下载这个文件不够**。
- 模型卡标注 MIT，核对使用条件见[模型卡](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo)。上游实现与下载映射见 [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)。

建议用下面的提前下载命令保留 Hugging Face 缓存结构。它只在你主动执行时联网，不会运行识别或占用 GPU：

```powershell
& $ComfyPython -c "from faster_whisper.utils import download_model; print(download_model('large-v3-turbo', cache_dir='F:/comfyui/models/faster-whisper'))"
```

不要把另一个仓库 ID 的平铺文件随意放到 `models/faster-whisper` 根目录。默认查找结构如下，`<revision>` 是实际下载版本，本机现有快照为 `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`；ASR 代码没有固定远端 revision：

```text
<ComfyUI>/models/faster-whisper/
└── models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/
    ├── refs/main
    ├── blobs/...
    └── snapshots/<revision>/
        ├── model.bin
        ├── config.json
        ├── preprocessor_config.json
        ├── tokenizer.json
        └── vocabulary.json
```

插件先尝试完整本地快照，只有缓存缺失或不完整才联网下载；实际 `WhisperModel` 加载使用 `local_files_only=True`。迁移到离线机器时复制整个 `models/faster-whisper` 缓存目录，保持 `refs`、`blobs`、`snapshots` 与链接目标完整，避免只复制可能是链接的单个快照文件。

## 2. HDemucs：唱歌模式的可选人声分离

下载：[PyTorch 官方 hdemucs_high_trained.pt](https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt)。放到：

```text
<ComfyUI>/models/H3LongVideo/hdemucs_high_trained.pt
# 本机示例：F:/comfyui/models/H3LongVideo/hdemucs_high_trained.pt
```

精确大小 **334,697,255 字节（约 319.19 MiB）**，插件会校验 SHA-256：

```text
a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816
```

模型与训练来源见 [Torchaudio HDEMUCS_HIGH_MUSDB_PLUS](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.html) 及其链接的 [Demucs 原始项目](https://github.com/facebookresearch/demucs)。权重使用条件需按上游说明核对，不从 Python 包许可证推定全部训练数据或音乐使用权。该下载地址不要求 Hugging Face 登录。已有 `torch.hub.get_dir()/torchaudio/models/hdemucs_high_trained.pt` 时，插件先校验并复制旧缓存；其位置会随 `TORCH_HOME` 等环境设置变化。

MV／口播通过“音频处理”选择：已有干净人声可选“直接使用”，不会加载此分离模型；选择“提取人声”时项目创建阶段整条处理一次并缓存，之后本地／RH 共用。MV 另支持可选 `vocals` 外接人声，连接时优先采用并校验与原曲对齐，不再重复分离，最终仍配回原曲。MV 使用 ASR 辅助切点，当前 Talking Avatar 默认单图／原音，停顿检测无需 ASR；不能把历史04的统一口播入口作为当前默认用法。

## 3. MOSS-Music：手动下载完整模型目录

该模型只用于本地音乐情绪分析，不参与 H3 画面生成，也不是必装项。

- 官方模型：[OpenMOSS-Team/MOSS-Music-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct)。
- 固定修订：`fce7f8304e96cc2d3398b8106456cbb2ecec3139`；[完整固定版本目录](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct/tree/fce7f8304e96cc2d3398b8106456cbb2ecec3139)。
- 四个权重合计 **18,105,036,640 字节（约 18.11 GB / 16.86 GiB）**，另有配置、tokenizer、处理器与模型代码。模型卡标注 Apache-2.0；核对条件见[该修订模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct/blob/fce7f8304e96cc2d3398b8106456cbb2ecec3139/README.md)。
- 路径固定为插件所在 ComfyUI 的 `models/moss_music/MOSS-Music-8B-Instruct`。它当前按插件物理位置计算，不跟随 `extra_model_paths.yaml` 或其他全局模型目录重定向。

下面示例下载整个固定版本，包括模型自定义 `.py` 文件；**不要只下载 `.safetensors`，也不要把四个分片拼接成一个文件**。命令由你主动执行时才联网，可能下载约 18.1 GB：

```powershell
& $ComfyPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='OpenMOSS-Team/MOSS-Music-8B-Instruct', revision='fce7f8304e96cc2d3398b8106456cbb2ecec3139', local_dir='F:/comfyui/models/moss_music/MOSS-Music-8B-Instruct')"
```

将示例中的 `F:/comfyui` 换成你的 `<ComfyUI>`。目录的运行所需文件应完整保留；全量下载还会包含模型卡及 `assets/`：

```text
<ComfyUI>/models/moss_music/MOSS-Music-8B-Instruct/
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── special_tokens_map.json
├── added_tokens.json
├── vocab.json
├── merges.txt
├── chat_template.jinja
├── configuration_moss_music.py
├── modeling_moss_music.py
└── processing_moss_music.py
```

要逐个下载可从[固定版本文件目录](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct/tree/fce7f8304e96cc2d3398b8106456cbb2ecec3139)点击各文件 Download；保留原文件名。推荐完整下载命令，以免遗漏小文件。当前已核对该版本四个分片与索引存在，HF 元数据为 `gated=false`，不需要接受下载门禁；服务端条件变化时以模型页面为准。

加载器使用 `trust_remote_code=True` 读取这个已下载目录中的模型代码，因此请保留并核对官方来源及固定修订。推理强制本地加载，子进程设置离线模式；Transformers 动态模块缓存默认在 `<ComfyUI>/models/moss_music/.hf_modules/`，可以从原模型目录重新生成。离线复用时复制完整模型目录即可，无需先运行在线 API。

本机 Windows／RTX 5090 已完成两段真实音乐样本的独立推理测试；这只证明本地音乐推理路径。当前运行中的 ComfyUI 主服务尚未重启加载新后端，不能据此宣称审核界面完整链路或新版 H3 长视频成片已经验收。其他硬件的速度和资源需求需自行验证，现有记录见 [验证记录](VERIFICATION.md)。

## 4. 当前 07 MV 与 13 口播：按画布选择模型

先按下表准备共同的编码器与两个 VAE，再准备各自的主模型／LoRA。路径均相对于 `<ComfyUI>/models/`。07 编码器保存了 `Qwen/` 子目录，13 保存在根目录；同一文件无需复制两份，导入后在各加载器重新选择即可。

| 使用范围 | 精确文件名与下载 | 目标目录 | 已核对大小 |
| --- | --- | --- | --- |
| 07／13 共同文本编码器 | [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `text_encoders/`（07 默认 `text_encoders/Qwen/`） | 27,141,342,152 字节，27.14 GB |
| 07／13 共同视频 VAE | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) | `vae/` | 2,811,065,184 字节，2.81 GB |
| 07／13 共同音频 VAE | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `vae/` | 605,254,808 字节，605 MB |
| 07 MV 主模型 | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `diffusion_models/MiniMax/` | 20,970,379,616 字节，20.97 GB |
| 07 MV LoRA，强度 `0.7` | [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) | `loras/` | 1,956,193,000 字节，1.96 GB |
| 07 MV SelfLift latent upscaler | [h3_upscaler_lms_v0.1.safetensors](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/latent_upscaler/h3_upscaler_lms_v0.1.safetensors) | `latent_upscale_models/` | 690,593,048 字节，691 MB |
| 13 口播主模型 | [minimax_h3_fl2va_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) | `diffusion_models/MiniMax/` | 34,038,892,334 字节，34.04 GB |
| 13 口播第一 LoRA，强度 `0.8` | [dmad_minimax_h3_4step_full_critic_comfy.safetensors（已核实社区存档）](https://huggingface.co/Kutches/minmax/blob/main/dmad_minimax_h3_4step_full_critic_comfy.safetensors) | `loras/` | 1,240,518,432 字节，1.24 GB |
| 13 口播第二 LoRA，强度 `0.6` | [zhnehisgan_c1-st900.safetensors](https://huggingface.co/RunningHubAI/rh-minimax-h3ailora-lora/resolve/main/zhnehisgan_c1-st900.safetensors) | `loras/` | 149,117,976 字节，149 MB |

07 的 `BasicScheduler` 当前为 `simple / 6步`，SelfLift 读取画布上的真实采样链路；13 当前为原生 `SamplerCustom / euler / simple / 6步 / CFG 1`、双 LoRA。**13 使用 Ref2VA 条件节点，但主模型实际保存为 FL2VA INT8**，不能仅凭条件节点名称把主权重改写为 Ref2VA。以上是随仓库提供的现有实验配置及可下载文件核对，不是本轮完成新 H3 成片或保证跨硬件兼容的证明。

第一口播 LoRA 已找到同名社区存档，但未核实原作者的有效正式下载入口及权重许可证；该链接不能表述为原作者官方发行。请核对来源和适用条款，未获得合适权重时可改用自己有权使用的 RunningHub 模板，或明确配置并验证其他本地组合，不要将相近名字 LoRA 当作等价文件。第二 LoRA 由 RunningHub 代作者发布，模型卡要求遵守原作者／上游许可；latent upscaler 仓库采用混合资产许可，应读[该资产说明](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/latent-upscaler.md)与模型卡，不能把整个目录统称 Apache-2.0。

```text
<ComfyUI>/models/
├── diffusion_models/MiniMax/
│   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors  # 07
│   └── minimax_h3_fl2va_int8_convrot.safetensors         # 13
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
├── vae/
│   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   └── minimax_h3_audio_vae_fp32.safetensors
├── loras/
│   ├── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   ├── dmad_minimax_h3_4step_full_critic_comfy.safetensors
│   └── zhnehisgan_c1-st900.safetensors
└── latent_upscale_models/
    └── h3_upscaler_lms_v0.1.safetensors
```

例如本机 upscaler 目标为 `F:/comfyui/models/latent_upscale_models/h3_upscaler_lms_v0.1.safetensors`。`talking_rh/local_weights.py` 的内存读取器从 ComfyUI 标准 `text_encoders` 路径读同一个文件，没有额外模型或自动下载；只在需要其内存读取方式的旧模式使用，13 的当前画布使用原生 `CLIPLoader`。

### 当前推荐工作流的节点依赖

按07／13 JSON去重后的真实 `type` 核对，以下是完整归属分组；没有采用过时的 `cnr_id` 推断。此次8188节点元数据接口不可达，因此对照本机现有原生源码注册表、外部插件注册源码与发布副本自有节点表，未启动服务或运行模型。

| 归属 | 实际节点类型 | 使用范围 |
| --- | --- | --- |
| 本插件 | `H3LVUnified`（MV Generator）、`H3LVPromptExpand`（MV Prompt Expander） | 07 |
| 本插件 | `H3LVTalkingAvatar`（Talking Avatar） | 13 |
| [SelfLift-Avatar](https://github.com/slmonker/selflift-Avatar) | `SelfLiftAvatarH3Sampler` | 07 |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `MiniMaxChunkFeedForward`、`MiniMaxLowVRAMAttention` | 07 |
| [ComfyUI-YCNodes-MiniMax-H3](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3) | `H3SigmaRefiner` | 07 |
| ComfyUI 原生 H3／AV 条件 | `MiniMaxH3ReferenceToVideo`、`LTXVAudioVAEEncode`、`LTXVConcatAVLatent`、`LTXVSeparateAVLatent` | 07／13 |
| ComfyUI 原生加载／处理 | `UNETLoader`、`CLIPLoader`、`VAELoader`、`LoraLoaderModelOnly`、`LoadAudio`、`VAEDecode`、`ModelAttentionBackend`、`SetLatentNoiseMask`、`SolidMask`、`ConditioningZeroOut`、`ResolutionSelector` | 07／13 |
| ComfyUI 原生采样／视频输出 | `BasicScheduler`、`KSamplerSelect`、`CreateVideo`、`SaveVideo` | 07／13 |
| ComfyUI 原生附加节点 | `MiniMaxH3SigmaShift`、`PreviewAny` | 07 |
| ComfyUI 原生附加节点 | `LoadImage`、`SamplerCustom` | 13 |
| 前端参数／注释虚拟节点 | `MarkdownNote`（07）；`PrimitiveNode`、`Note`（13） | 标准前端功能，不是额外Python模型节点 |

最低功能要求是上述原生节点均存在，`CLIPLoader`支持 `minimax` 类型、所选H3量化格式与当前Torch兼容，且AV latent能保留视频／音频结构。07还选择 `comfy kitchen attention`，需要当前环境提供对应后端；其他后端须在画布显式选择并重新验证。这里没有核实可覆盖全部要求的单一最低版本号，不编造版本下限。原生实现可对照 [ComfyUI源码](https://github.com/Comfy-Org/ComfyUI)，本地注册文件包括 `nodes.py` 与 `comfy_extras/nodes_minimax_h3.py`、`nodes_lt.py`、`nodes_lt_audio.py`、`nodes_resolution.py`、`nodes_model_advanced.py`、`nodes_custom_sampler.py`、`nodes_video.py`等。

13当前没有SelfLift／KJ／YC专用节点；07／13都使用原生 `CreateVideo → SaveVideo`，无需VHS保存节点。只有历史04或自行换成 `VHS_VideoCombine` 时，才需 [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)。RH模式不执行本地视频模型分支，但导入展开画布仍需相应节点定义。旧统一节点的口播入口不再保留；向Talking Avatar迁移的范围及旧素材保留规则见[功能保留对照表](FEATURE_PARITY.md)。

## 5. RunningHub 模板与历史口播附加模型

选择 RH 时，不需要把云端视频权重下载到本机。请把 [08 MV 模板](examples/08_H3MV_RH云端生成模板.json)或 [15 口播模板](examples/15_H3口播_独立窗口_RH云端模板.json)另存到自己的 RunningHub 空间，按该空间的模型资源名称选择。07／13 的本地文件名和采样参数不自动覆盖云端；模型条件和本地环境也不能因为配置检查通过就视为真实云端生成已经通过。

| 云端模板 | 当前保存的差异项 | 云端对应目录／选择器 |
| --- | --- | --- |
| 08 MV | 主模型 `minimax_h3_fl2va_bf16.safetensors`；LoRA `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_rank_21.safetensors`，强度 `0.7`；`h3_upscaler_lms_v0.1.safetensors` | 主模型 `diffusion_models`，LoRA `loras`，SelfLift `latent_upscale_models` |
| 15 口播 | 主模型 `minimax_h3_fl2va_bf16.safetensors`；与 13 相同的 dmad 与 zhnehisgan 双 LoRA，强度 `0.8/0.6` | 主模型 `diffusion_models`，LoRA `loras` |
| 两者共同 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`、`minimax_h3_video_vae_int8_convrot.safetensors`、`minimax_h3_audio_vae_fp32.safetensors` | `text_encoders` 与 `vae` |

云端 BF16 主模型的已核实官方文件为 [Comfy-Org minimax_h3_fl2va_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors)，大小 66,280,487,368 字节（66.28 GB）。08 中 `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_rank_21.safetensors` 的精确官方发布文件尚未核实，不提供猜测直链；请在自己有权访问的 RH 资源中选择对应文件，并核对 [LightX2V 发行目录](https://huggingface.co/lightx2v/Minimax-h3-Turbo)。不能擅自把它改名为另一 LoRA 视作已兼容。

14 仅为精简 RH 控制画布；05／06／09／10／11 用于旧项目与历史采样方式恢复，不能套用当前 13 的模型列表。旧内置模式的 `local_model`、`local_clip`、`local_video_vae`、`local_audio_vae`、`local_upscaler` 分别对应 `diffusion_models`、`text_encoders`、`vae`、`vae`、`latent_upscale_models`，可选 `local_lora` 对应 `loras`。已有项目使用冻结的原配置；历史融合模型来源见 [MATLOWAI/minimax-h3-fused-turbo-int8-convrot](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot)，需按旧项目记录选择，当前推荐工作流不要求另下载它。

旧自动人物遮罩项目可能需要 `selfie_multiclass_256x256.tflite`：放到 `<ComfyUI>/models/mediapipe/`，下载 [Google MediaPipe 官方模型](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite)，说明见 [官方 Image Segmenter 文档](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)。插件只读本地模型，不自动下载。当前 13／15 单图口播不启用人物空间分割，不需要此模型；原有项目按原遮罩配置恢复。

## 6. 历史 04 H3 一体化工作流：五项权重

下表与当前示例 `widgets_values` 实际选择一致。H3 主模型、文本编码器和两个 VAE 来自 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)，属于供 ComfyUI 使用的重打包模型；原始模型和输入规范见 [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)。无需为了本示例再下载原始 `FL2VA/`、`Ref2VA/` 全套目录。

| 工作流节点 | 精确文件名与直接下载 | `<ComfyUI>/models/` 下的目标位置 | 文件大小 |
| --- | --- | --- | --- |
| `UNETLoader`（127） | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `diffusion_models/MiniMax/` | 20,970,379,616 字节，20.97 GB |
| `CLIPLoader`（128），类型 `minimax` | [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `text_encoders/` | 27,141,342,152 字节，27.14 GB |
| 视频 `VAELoader`（119） | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) | `vae/` | 2,811,065,184 字节，2.81 GB |
| 音频 `VAELoader`（120） | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `vae/` | 605,254,808 字节，605 MB |
| `LoraLoaderModelOnly`（201），强度 `1` | [minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors) | `loras/MiniMax/` | 1,956,193,000 字节，1.96 GB |

文件体积来自相应 Hugging Face 仓库文件元数据。五项共约 **53.48 GB**；仅分析音频无需下载这些视频权重。主模型和 LoRA 的 `MiniMax/` 是示例保存的子目录；放到加载器的标准根目录也可以，但导入示例后必须重新选择实际文件名。

```text
<ComfyUI>/models/
├── diffusion_models/MiniMax/
│   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
├── vae/
│   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   └── minimax_h3_audio_vae_fp32.safetensors
└── loras/MiniMax/
    └── minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
```

例如本机主模型完整路径为 `F:/comfyui/models/diffusion_models/MiniMax/minimax_h3_ref2va_pruned_int8_convrot.safetensors`。H3 加载器也可使用 ComfyUI 已正确配置的额外模型搜索路径；选择框能找到文件才表示路径生效。

**示例沿用的采样组合有适用限制。** 文件名中的 FL2V Turbo LoRA 与 Ref2VA 是不同任务家族；当前示例保留既有 `Ref2VA + FL2V Turbo v1.2 / euler / beta / 10步 / shift 12、3` 设置，不能把文件名“4step”当作当前工作流已经使用4步，也不能把这个组合当作上游官方 Ref2VA 推荐配方。改用其他 LoRA 时，应根据 [LightX2V 模型目录及说明](https://huggingface.co/lightx2v/Minimax-h3-Turbo)、[Turbo 作者项目](https://github.com/ModelTC/Minimax-H3-Turbo)选择匹配任务的模型、LoRA 和采样参数，重新验证画面与音频。

下载前应以实际 `widgets_values`选择为准，不把画布 `properties.models`下载元数据视为加载器当前选择。历史示例可能保留另一精度的元数据。替代文件可在 [官方 VAE 目录](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae)与[文本编码器目录](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders)核对：`minimax_h3_video_vae_fp16.safetensors` 为5.21 GB，`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` 为15.69 GB。它们是替代精度选项，无需与当前选项同时下载；更换后在节点里重新选择，并确认当前ComfyUI／Torch支持该量化格式。不要将本插件依赖安装视为H3量化环境已经就绪。

### 历史 04 的外部节点与通用模型许可证

历史 04 示例中 H3、AV latent 拼接／分离、加载器、`ResolutionSelector`、`MiniMaxH3SigmaShift`、`ModelAttentionBackend` 等在当前 ComfyUI 源码中已有原生节点，需要实际包含这些节点的 ComfyUI 版本。JSON 中保存的节点版本是作者环境记录，不是自动安装或兼容保证。

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)：示例的 `VHS_VideoCombine`。改用 README 介绍的原生 `CreateVideo → SaveVideo` 分支时可不使用它；合成仍需要 FFmpeg。
- `comfyui-vrgamedevgirl` 与 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) 均不是历史 04 的必需外部节点；`ModelAttentionBackend` 的当前原生实现在 `comfy_extras/nodes_model_advanced.py`。旧 ComfyUI 若缺少该节点，不能仅凭旧来源字段假定安装 KJNodes 即可修复，应核对当前原生节点支持。07 的 KJ 专用节点依赖已在第 4 节单独列出，其他旧版／自建工作流以真实节点清单为准。

H3 权重采用 [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，应独立阅读，不能由本插件是否公开推定商用许可。Turbo 仓库当前模型元数据未给出明确的 license 字段，应向上游确认适用条款，并同时遵守基础模型条款；不要仅凭相关代码仓库许可证推定 LoRA 权重授权。

核对时上述 H3、Turbo、ASR、MOSS 仓库元数据均为 `gated=false`，公开文件可匿名下载。若今后页面要求登录或接受条款，在 Hugging Face 官方页面完成后再下载；不要把令牌写入工作流、命令示例或本仓库。下载失败时先核对访问条件、网络和磁盘空间，不要反复更换权重来掩盖路径错误。

## 离线使用与排错

完整复制模型文件后，ASR 优先复用缓存、HDemucs 校验复用本地文件、MOSS 只从本地加载，H3 由原生加载器读取所选权重。提示词扩写如果仍选择在线 API 或 Codex CLI，则仍需要对应服务连接；使用手写提示词通道才能跳过这类扩写请求。

找不到模型时，按目标节点核对“文件所在目录、完整原文件名、加载器选择、目录是否多套了一层”。MOSS 报缺文件时补齐同一固定 revision 的全部配套文件；ASR 离线报缺文件时核对 Hugging Face 缓存引用与链接目标；HDemucs 校验失败时核对上面的字节数与 SHA-256。不要把大型权重、模型缓存、私人音频、API Key 或项目输出上传到插件 Git 仓库。
