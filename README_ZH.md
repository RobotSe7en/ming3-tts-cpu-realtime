# Ming-3.0-TTS 0.5B x86 CPU 实时推理优化（`RTF < 1.0`）

[English](README.md) | [中文说明](README_ZH.md) | [▶ 在线 A/B 音频试听](https://robotse7en.github.io/ming3-tts-cpu-realtime/) | [完整实验演进记录](docs/experiments.md)

在一台 6 年前的入门级 4 核桌面 CPU（**Intel Core i3-10100 @ 3.60 GHz**，TrueNAS Linux 虚拟机，仅 4 线程、无独显、无核显直通）上，将 **Ming-3.0-TTS 0.5B** 纯 CPU 推理从原版 `RTF 1.72–2.08` 优化至 **`RTF 0.79–0.99`（快于实时播放）**——直接复用官方 **`Q8_0` GGUF** 权重（加载期并行重排约 `2.3 秒`，无需另转模型），零 swap，峰值内存最高降低 **46%**。

本项目以独立 C++ AVX2 微内核与可重放补丁形式构建于 [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp)（固定 commit `7d20b414`，Apache 2.0）之上，不修改上游基线源码。

---

## 核心 ABBA 对照结果

全部测试在一致的 `systemd-run` 资源约束下（**4 线程、`CPUQuota=400%`、`MemoryMax=6G`、`MemorySwapMax=0`**）按 **ABBA** 交替顺序测得（每块预热 1 次、测 2 次，每版累计 4 次正式测量），重复运行 WAV SHA-256 逐字节确定性一致。

### 一、三套 `RTF ≤ 1` 运行方案（同一二进制、同一份 `Q8_0` GGUF）

| 方案 | 短句 (4s) | 长段 (18–22s) | `灵小柔` (6s) | 峰值内存 | 定位 |
| --- | ---: | ---: | ---: | ---: | --- |
| **原版基线** (`Q8_0`) | `1.72–1.82` | `1.97–2.08` | `1.92` | `1.98 GiB` | 上游对照 |
| **方案 A** ⭐ | **`0.84`** (2.06×) | **`0.99`** (2.10×) | **`0.97`** (1.98×) | **`1.23 GiB`** (-38%) | **默认推荐** · 音质最佳 |
| **方案 B** | **`0.79`** (2.18×) | **`0.87`** (2.25×) | **`0.87`**\* | **`1.09 GiB`** (-45%) | 极限速度 / 最低内存 |
| **方案 C** | **`0.89`** (2.05×) | **`0.99`** (2.01×) | — | **`1.07 GiB`** (-46%) | 保留完整 10 步官方采样 |

- **方案 A（推荐：高音质实时版）**：`AR=q8`（`--fused-fp32-act auto`，保留 AR 上游全部 13 处 BF16 舍入）+ `Flow/Agg=q4g128`（8 步 sway，每步完整 CFG=2.0）+ `VAE=q4_0`。`灵小柔` 30 字（**`0.9337`** 余弦、`CER=0.0000`）与 107 字（**`0.9383`** 余弦）均通过自动筛查，音频核验确认第 2 块机械音已消除（可在下方在线试听页直接对比）。*（可选 `--ar-quant iq4_nl --fused-fp32-act 0 --flow-quant q4g64` 单次实测 `灵小柔` 30/107 字为 `0.89`/`0.93`，内存 `1.11 GiB`，无 `Q4_0` 金属音。）*
- **方案 B（极速 / 最低内存版）**：`AR=q4_0` + `Flow/Agg=q4g64`（8 步 sway，全步 CFG）+ `VAE=q4_0`。速度最快、抗抖动裕度最大、零削波（*`灵小柔` 30 字为单次测量：`CER=0.0000`，余弦 `0.8501`）。
- **方案 C（完整 10 步官方采样版）**：`AR=q4_0` + `Flow/Agg=q4g128`（官方 10 步，全步 CFG）+ `VAE=q4_0`。不改动采样步数与 CFG 策略。

### 二、主推荐方案 A（`AR=q8` BF16 + `Flow=q4g128`）ABBA 明细与筛查

| 测试集 | 原版 RTF | 方案 A RTF | 加速比 | 峰值内存 | CER / 音色余弦 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短句 `zh_short` (3.9s) | `1.740` | **`0.844`** | **2.06×** | `1.33 → 1.22 GiB` | `0.00` / `0.785` |
| 长段 `zh_paragraph` (17.7s) | `2.079` | **`0.991`** | **2.10×** | `1.98 → 1.23 GiB` | `0.04` / `0.709` |
| `灵小柔` 30 字 (6.0s) | `1.921` | **`0.972`** | **1.98×** | `1.35 → 1.22 GiB` | `0.00` / **`0.934`** ✓ |
| `灵小柔` 107 字 (17.9s)¹ | `2.141` | `1.120`¹ | **1.91×** | `1.97 → 1.27 GiB` | `0.01` / **`0.938`** ✓ |

<details>
<summary>展开查看完整 4 次 RTF 分布、削波比例与回归测试结果</summary>

| 测试集 | 原版 | 方案 A | 方案 A 4 次分布 | 原版 → 方案 A CER | 削波比例 |
| --- | ---: | ---: | --- | ---: | ---: |
| `zh_short.txt` | `1.7403` | `0.8442` | `0.8238, 0.8539, 0.8344, 0.8596` | `0.0000 → 0.0000` | `0.000000` |
| `zh_paragraph.txt` | `2.0794` | `0.9908` | `1.0149, 1.0318, 0.9668, 0.9512` | `0.0625 → 0.0417` | `0.000005` |
| `zh_lingxiaorou_short.txt` | `1.9209` | `0.9724` | `0.9808, 0.9684, 0.9570, 0.9764` | `0.0000 → 0.0000` | `0.000000` |
| `zh_paragraph.txt`（`灵小柔`） | `2.1406` | `1.1200` | `1.1672, 1.2344, 1.0674, 1.0728` | `0.0000 → 0.0104` | `0.000000` |
| `zh_regression.txt`（单次） | `1.8157` | `0.8430` | `0.8430`（2.15×） | — | — |
| `zh_long_regression.txt`（单次） | `2.1915` | `0.9491` | `0.9491`（2.31×） | — | — |

</details>

> ¹ 在固定版本 `7d20b414` 中，`灵小柔` 通过 `template_name=freeform` 传入（绕过旧版仅含英文音色的 `voice_id` 枚举校验），不会置位 `controls["ip"]`，因此长文本仍多跑了首段重锚定且每块多约 55 个 JSON token（宿主机空闲单次实测 **`0.984`**；本轮 ABBA 期间宿主机负载略高，原版基线也同步从 `2.02` 涨至 `2.14`）。

全部脱敏后的 ABBA 汇总与 SenseVoice INT8 ASR + CAMPPlus 筛查 JSON 保存在 [`benchmarks/published_results/`](benchmarks/published_results/)。自动化 ASR 与说话人余弦仅用于回归筛查，不等同于正式人工 MOS 评测。

---

## 音频示例（原版 vs 方案 A 实听对比）

> **▶ [打开在线 A/B 音频试听页（浏览器直接点播，无需下载）](https://robotse7en.github.io/ming3-tts-cpu-realtime/)**
>
> 由于 GitHub README 会过滤 `<audio>` 标签，我们在 GitHub Pages 提供了免下载在线 A/B 对比播放器，同时在 [Release `audio-samples`](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/tag/audio-samples) 保留原始 44.1 kHz 无损 WAV 与波形对比视频下载。

| 示例 | A：原版基线（`Q8_0`） | B：方案 A 实时版 | 对比指标 |
| --- | --- | --- | --- |
| **1. `灵小柔` 短句**（30 字，6s） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/01_lingxiaorou_30c_baseline_q8.wav)（`1.92`） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/01_lingxiaorou_30c_planA_ar_q8.wav)（**`0.97`**）· [A/B MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_lingxiaorou_30c.mp4) | **1.98×** · 余弦 **`0.934`** ✓ |
| **2. `灵小柔` 长段落**（107 字，18s） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/02_lingxiaorou_107c_baseline_q8.wav)（`2.14`） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/02_lingxiaorou_107c_planA_ar_q8.wav)（`1.12`¹）· [A/B MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_lingxiaorou_107c.mp4) | **1.91×** · 余弦 **`0.938`** ✓ |
| **3. 长段落无参考**（107 字，18s） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/03_uncond_107c_baseline_q8.wav)（`2.08`） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/03_uncond_107c_planA_ar_q8.wav)（**`0.99`**）· [A/B MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_uncond_107c.mp4) | **2.10×** · 内存 `-38%` |

所有样本均在 4 线程 CPU、固定种子 `1895` 下确定性生成，未做任何后期处理。

---

## 为什么必须从架构与访存下手：屋顶线与核显核查

- **耗时分布（4 线程 `Q8_0` 原版）**：Flow 采样占 **68.4%**（4 秒音频执行 500 次 `M=6` 串行前向），AR 主干占 **16.4%**，VAE 解码占 **10.7%**，潜变量投影（Aggregator）占 **4.3%**。
- **DDR4 内存带宽物理下界**：实测本机 4 线程读带宽约 **`25.35 GB/s`**。纯 `Q8_0` 10 步 Flow 光是 500 次读取 179 MB 权重（共 `89.3 GB`）就至少需要 **`3.52 秒`（单项 `RTF ≥ 0.88`）**，叠加 AR + VAE + 投影后总下界在 `1.35+`，仅靠算子微调在物理上不可能达到 `≤ 1.0`。
- **关于核显（iGPU）**：本 VM 仅暴露虚拟 QXL 显卡，无 `/dev/dri/renderD*`；即便直通 i3-10100 的 UHD 630（Gen9.5，24 EU，无 DP4A/XMX），它仍共享同一条 `25.35 GB/s` 内存总线，且 500 次 `M=6` 极小批量串行前向叠加 GPU 提交/同步延迟反而吃亏。

---

## 核心优化技术

1. **常驻直连 C++ Flow / Aggregator 引擎**（[`src/ming_cpu/flow_engine.cpp`](src/ming_cpu/flow_engine.cpp)）：
   - 用单次跨全部 Flow 步的常驻 `#pragma omp parallel` 区域替代每步 ~200 个 GGML 计算图节点与同步屏障。
   - 预计算 `time_mlp`、每 patch 缓存 `cond_emb`/`hist_emb`/第 0 层 `hist_qkv`、cond/uncond 共享第 0 层 `x_emb` QKV、加载期预计算 Aggregator `cls_qkv`；L1 栈内融合 `RMSNorm+BF16+量化`、`RoPE+Attention+BF16`、`bias+BF16+残差/GELU LUT`；为 `M=1,2` 新增 4 通道展开内核，支持批量 Aggregator（`chunk ≤ 3`）并缓存分块间重复的 `prompt_latents` 投影。
2. **组量化非对称 4-bit AVX2 微内核（`Q4_1_G64` / `Q4_1_G128`）**：
   - 将每个输出通道打包为连续 32B 对齐 slab（64 或 128 维一组，无符号 4-bit），配合组主序 `Q8GroupInputBuffer` 消除 `4096` 步长 4 KiB L1d 缓存组冲突；内层采用 `vpmaddubsw + vpaddw`（零 `vpsignb` 开销，最大 16 位中间和 `15,240 < 32,767` 无溢出）。全部 AVX2/FMA/F16C 路径通过 `__builtin_cpu_supports` 运行时门控，保留完整标量回退与 `Q8_0` 路径。
3. **AR / VAE 加载期 `CPU_REPACK` 与融合 FP32 激活**（[`src/ming_cpu/repack_weights.cpp`](src/ming_cpu/repack_weights.cpp)）：
   - 加载期 OpenMP 并行从 `Q8_0` GGUF 重排到 GGML `ggml_backend_cpu_repack_buffer_type()`（`Q4_0_8x8_q8_0`，总加载耗时约 `2.3 秒`）；`--fused-fp32-act auto` 仅对 4-bit 量化模块关闭中间人工 BF16 截断以解锁融合 `SwiGLU`，同时在 `Q8_0` AR 上保留原版 BF16 舍入确保长文本稳定。
4. **8 步均匀 `u` Sway 网格与首段 Latent 复用**：
   - 修复早期非均匀 EPSS 节点末步跳变 `dt = 0.707` 导致的吞字问题，在 `u` 空间均匀采样后做余弦 sway 变换并全程保留 `CFG = 2.0`；无参考长文本分块时按精确 `patch_samples = output_dim * patch_size` 直接复用首段 `latent`（保留 `CAMPPlus`），省约 `1.0 秒` 和 `380 MiB` 内存（可用 `--reuse-anchor-latent 0` 关闭）。
5. **两轮实听排错（为什么方案 A 采用 `AR=q8` + `--fused-fp32-act auto`）**：
   - **第一轮（`AR=q4_0` 电话音）**：仅看 SenseVoice ASR CER 时，`AR=q4_0` 长文本表现很好（`0.02–0.03`），但与 Mac Metal 输出盲听对比后发现第 2 分块起出现明显金属/电话压缩感；逐项隔离精准定位为 24 层 `AR=q4_0`。
   - **第二轮（`AR=q8` 强开 `fused_fp32_act=1` 机械音）**：精听 `灵小柔` 107 字长段落时在第 2 块后半段发现机械音；单变量消融隔离出元凶为移除 AR 主干 13 处上游 BF16 舍入。方案 A 因此改用 **`--fused-fp32-act auto`**（AR 保留上游 BF16 舍入，仅 4-bit VAE 走融合 FP32）并配合 `Flow/Agg=q4g128`，彻底消除第 2 块机械音，`灵小柔` 30/107 字音色余弦达 **`0.9337`** / **`0.9383`**（通过），短句、无参考长段与 `灵小柔` 30 字均稳在 **`RTF 0.84–0.99`**。
6. **真·帧级实时流式：24 层滑窗环形 KV Cache VAE 与有状态 ISTFT**：
   - 上游 `ming3_tts` 流式解码每块重算最多 `380` 个历史 patch（`30.4 秒`）的 24 层 Qwen2 VAE（$O(N^2)$ 复杂度），导致 4 帧流式 RTF 恶化至 **`2.99`**。
   - 新增 **128 槽 `F32` 环形滑窗（`window=64`）KV Cache**（24 层共约 `3 MiB`）、精确 1-patch 左右邻 `Interpolate1d(Linear)` 边界与 **3 帧重叠有状态 ISTFT**，将流式 VAE 降为 $O(N)$（`left_context_frames: 380 → 1`）。
   - `ming3_stream_vae_parity` 双圈环形回绕单测验证 `grow(8)` 与 `fixed(3)` 切块逐样本比特一致（`max_abs = 0`），真实「灵小柔」30 字波形与离线余弦达 **`0.99987`**，流式 RTF 从 **`2.987 → 0.937`**（加速 **3.19×**，甚至快于离线 `0.988`），端到端首包 **`0.87 ~ 1.24 s`**，200 ms 缓冲下 **`0 ms` 断流**。

---

## 快速复现

### 1. 准备固定版本 `audio.cpp` 与官方 `Q8_0` 模型

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/0xShug0/audio.cpp.git .cache/audio.cpp
git -C .cache/audio.cpp fetch --depth 1 origin 7d20b414da088c3ffe56bd8803398d9d158cb847
git -C .cache/audio.cpp checkout --detach 7d20b414da088c3ffe56bd8803398d9d158cb847
git -C .cache/audio.cpp sparse-checkout set app include src external model_specs tools webui/native/demo_voices

mkdir -p models
curl --noproxy '*' --fail --location --retry 3 \
  --output models/ming-3.0-tts-0.5b-q8_0.gguf \
  https://modelscope.cn/models/audio-cpp/Ming-3.0-TTS-0.5B-GGUF/resolve/master/ming-3.0-tts-0.5b-q8_0.gguf
# 文件大小：1742280896 字节
# SHA-256：fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1
```

### 2. 应用补丁、编译并运行 CTest 数值一致性测试

```bash
git -C .cache/audio.cpp worktree add --no-checkout --detach "$PWD/.cache/audio.cpp-ming-fast" HEAD
git -C .cache/audio.cpp-ming-fast sparse-checkout set app include src external model_specs tools webui/native/demo_voices
git -C .cache/audio.cpp-ming-fast checkout --detach HEAD
git -C .cache/audio.cpp-ming-fast apply "$PWD/src/patches/ming_fast_cpu.patch"

cmake -S benchmarks -B .cache/bench-build -DCMAKE_BUILD_TYPE=Release \
  -DTTS_BUILD_MING_ENGINE=ON -DTTS_BUILD_MING_FAST=ON -DTTS_BUILD_MING_PROFILE=ON
cmake --build .cache/bench-build --target ming3_rtf ming3_fast_rtf ming3_fast_profile_rtf ming3_fast_server ming3_fast_engine_test ming3_stream_vae_test -j 2
ctest --test-dir .cache/bench-build --output-on-failure
```

### 3. 部署为实时 HTTP / 流式 TTS 服务

编译 `ming3_fast_server` 后，可用 `deploy/` 中的后端配置（单热模型 `ming3-tts-stream` 常驻约 `1.4 GiB` / `0 Swap`）配合无第三方依赖的 HTTP/1.1 网关（`deploy/tts_gateway.py`，进程内 `libsoxr` 流式重采样）直接提供 OpenAI WAV 与 `AYTS/1` 流式接口：

```bash
# 复制并按实际仓库路径修改 systemd --user 模板
mkdir -p ~/.config/systemd/user
cp deploy/ming3-tts.service.example ~/.config/systemd/user/ming3-tts.service
cp deploy/ming3-tts-gateway.service.example ~/.config/systemd/user/ming3-tts-gateway.service
systemctl --user daemon-reload && systemctl --user enable --now ming3-tts.service ming3-tts-gateway.service

# OpenAI 兼容离线 WAV 接口（复用常驻热模型，无切换重载）
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"ming3-tts","input":"你好，这是默认灵小柔实时语音服务。","voice":"灵小柔"}' \
  --output out.wav

# 帧级实时流式 AYTS/1 PCM 接口（24 kHz 单声道 s16le；默认首包 ~1.2s，grow 模式 ~0.87s）
curl --no-buffer --noproxy '*' http://127.0.0.1:8080/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"model":"ming3-tts","input":"你好，这是流式语音合成测试。","voice":"灵小柔","response_format":"pcm","sample_rate":24000}' \
  --output speech.ayts
```

### 4. 运行 ABBA 基准对照

```bash
# 方案 A（推荐高音质实时版：AR=q8 + fused=auto 保留 AR BF16 + Flow/Agg=q4g128 8 步 + VAE=q4_0）
python3 benchmarks/compare_ming_fast.py --fixture zh_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g128 --agg-quant q4g128 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g128 --agg-quant q4g128 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1 --skip-regressions

# 方案 A + 官方内置音色「灵小柔」
INSTRUCT='{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null, "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}, "IP": "灵小柔"}]}'
python3 benchmarks/compare_ming_fast.py --fixture zh_lingxiaorou_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g128 --agg-quant q4g128 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1 \
  --template-name freeform --instruct "$INSTRUCT" --tag-suffix lingxiaorou --skip-regressions

# 方案 B（极速最低内存：AR=q4_0 + Flow/Agg=q4g64 8 步 + VAE=q4_0）
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --ar-quant q4_0 --vae-quant q4_0 --flow-steps 8 --cfg-interval 1 --skip-regressions

# 方案 C（保留完整 10 步官方采样：AR=q4_0 + Flow/Agg=q4g128 10 步 + VAE=q4_0）
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g128 --ar-quant q4_0 --vae-quant q4_0 --flow-steps 10 --cfg-interval 1 --skip-regressions
```

## 开源协议与致谢

采用 [Apache License 2.0](LICENSE)。感谢 [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp)（及 `ggml`）与 [`inclusionAI/Ming-3.0-TTS-0.5B`](https://modelscope.cn/models/audio-cpp/Ming-3.0-TTS-0.5B-GGUF)。
