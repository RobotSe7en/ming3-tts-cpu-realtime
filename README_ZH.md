# Ming-3.0-TTS 0.5B x86 CPU 实时推理优化（`RTF < 1.0`）

[English](README.md) | [中文说明](README_ZH.md) | [▶ 在线 A/B 音频试听](https://robotse7en.github.io/ming3-tts-cpu-realtime/) | [完整实验演进记录](docs/experiments.md)

在一台 6 年前的入门级 4 核桌面 CPU（**Intel Core i3-10100 @ 3.60 GHz**，TrueNAS Linux 虚拟机，仅 4 线程、无独显、无核显直通）上，将 **Ming-3.0-TTS 0.5B** 纯 CPU 推理从原版 `RTF 1.70–1.99` 优化至 **`RTF 0.79–0.97`（快于实时播放）**——直接复用官方 **`Q8_0` GGUF** 权重（加载期并行重排约 `2.3 秒`，无需另转模型），零 swap，峰值内存最高降低 **46%**。

本项目以独立 C++ AVX2 微内核与可重放补丁形式构建于 [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp)（固定 commit `7d20b414`，Apache 2.0）之上，不修改上游基线源码。

---

## 核心 ABBA 对照结果

全部测试在一致的 `systemd-run` 资源约束下（**4 线程、`CPUQuota=400%`、`MemoryMax=6G`、`MemorySwapMax=0`**）按 **ABBA** 交替顺序测得（每块预热 1 次、测 2 次，每版累计 4 次正式测量），重复运行 WAV SHA-256 逐字节确定性一致。

### 一、三套 `RTF ≤ 1` 运行方案（同一二进制、同一份 `Q8_0` GGUF）

| 方案 | 短句 (4s) | 长段 (23s) | `灵小柔` (6s) | 峰值内存 | 定位 |
| --- | ---: | ---: | ---: | ---: | --- |
| **原版基线** (`Q8_0`) | `1.70–1.82` | `1.92–1.99` | `1.84` | `1.98 GiB` | 上游对照 |
| **方案 A** ⭐ | **`0.86`** (1.97×) | **`0.97`** (1.98×) | `1.01` (1.81×) | **`1.31 GiB`** (-34%) | **默认推荐** · 音质最佳 |
| **方案 B** | **`0.79`** (2.18×) | **`0.87`** (2.25×) | **`0.87`**\* | **`1.09 GiB`** (-45%) | 极限速度 / 最低内存 |
| **方案 C** | **`0.89`** (2.05×) | **`0.99`** (2.01×) | — | **`1.07 GiB`** (-46%) | 保留完整 10 步官方采样 |

- **方案 A（推荐：高音质均衡版）**：`AR=q8`（`--fused-fp32-act auto`，保留 AR 上游原版 BF16 舍入）+ `Flow/Agg=q4g64`（8 步 sway，全步 CFG）+ `VAE=q4_0`。四组测试全部 `0.0000` CER，`灵小柔` 107 字说话人余弦 **`0.9295`**（通过），彻底消除长文第 2 块机械音。
- **方案 B（极速 / 最低内存版）**：`AR=q4_0` + `Flow/Agg=q4g64`（8 步 sway，全步 CFG）+ `VAE=q4_0`。速度最快、抗抖动裕度最大、零削波（*`灵小柔` 30 字为单次测量：`CER=0.0000`，余弦 `0.8501`）。
- **方案 C（完整 10 步官方采样版）**：`AR=q4_0` + `Flow/Agg=q4g128`（官方 10 步，全步 CFG）+ `VAE=q4_0`。不改动采样步数与 CFG 策略。

### 二、主推荐方案 A（`AR=q8`、`fused=auto`）ABBA 明细与自动质量筛查

| 测试集 | 原版 RTF | 方案 A RTF | 加速比 | 峰值内存 | CER / 音色余弦 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短句 `zh_short` (4.0s) | `1.697` | **`0.862`** | **1.97×** | `1.33 → 1.24 GiB` | `0.00` / `0.842` |
| 长段 `zh_paragraph` (22.8s) | `1.916` | **`0.970`** | **1.98×** | `1.98 → 1.31 GiB` | `0.00` / `0.869` |
| `灵小柔` 30 字 (5.8s) | `1.837` | `1.012` | **1.81×** | `1.35 → 1.24 GiB` | `0.00` / `0.887` |
| `灵小柔` 107 字 (18.2s)¹ | `2.042` | `1.064`¹ | **1.92×** | `1.97 → 1.27 GiB` | `0.00` / **`0.930`** ✓ |

<details>
<summary>展开查看完整 4 次 RTF 分布、削波比例与回归测试结果</summary>

| 测试集 | 原版 | 方案 A | 方案 A 4 次分布 | 原版 → 方案 A CER | 削波比例 |
| --- | ---: | ---: | --- | ---: | ---: |
| `zh_short.txt` | `1.6974` | `0.8622` | `0.8592, 0.8651, 0.9599, 0.8592` | `0.0000 → 0.0000` | `0.000000` |
| `zh_paragraph.txt` | `1.9161` | `0.9696` | `0.9656, 0.9624, 0.9847, 0.9737` | `0.0625 → 0.0000` | `0.000046` |
| `zh_lingxiaorou_short.txt` | `1.8368` | `1.0124` | `1.0007, 1.0196, 1.0696, 1.0052` | `0.0000 → 0.0000` | `0.000000` |
| `zh_paragraph.txt`（`灵小柔`） | `2.0423` | `1.0643` | `1.0567, 1.0538, 1.0980, 1.0719` | `0.0000 → 0.0000` | `0.000000` |
| `zh_regression.txt`（单次） | `1.6744` | `0.9098` | `0.9098`（1.84×） | — | — |
| `zh_long_regression.txt`（单次） | `1.8428` | `0.9593` | `0.9593`（1.92×） | — | — |

</details>

> ¹ 在固定版本 `7d20b414` 中，`灵小柔` 通过 `template_name=freeform` 传入（绕过旧版仅含英文音色的 `voice_id` 枚举校验），不会置位 `controls["ip"]`，因此长文本仍多跑了首段重锚定且每块多约 55 个 JSON token。

全部脱敏后的 ABBA 汇总与 SenseVoice INT8 ASR + CAMPPlus 筛查 JSON 保存在 [`benchmarks/published_results/`](benchmarks/published_results/)。自动化 ASR 与说话人余弦仅用于回归筛查，不等同于正式人工 MOS 评测。

---

## 音频示例（原版 vs 方案 A 实听对比）

> **▶ [打开在线 A/B 音频试听页（浏览器直接点播，无需下载）](https://robotse7en.github.io/ming3-tts-cpu-realtime/)**
>
> 由于 GitHub README 会过滤 `<audio>` 标签，我们在 GitHub Pages 提供了免下载在线 A/B 对比播放器，同时在 [Release `audio-samples`](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/tag/audio-samples) 保留原始 44.1 kHz 无损 WAV 与波形对比视频下载。

| 示例 | A：原版基线（`Q8_0`） | B：方案 A 高音质版 | 对比指标 |
| --- | --- | --- | --- |
| **1. `灵小柔` 短句**（30 字，6s） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/01_lingxiaorou_30c_baseline_q8.wav)（`1.84`） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/01_lingxiaorou_30c_planA_ar_q8.wav)（`1.01`）· [A/B MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_lingxiaorou_30c.mp4) | **1.81×** · CER `0.00` |
| **2. `灵小柔` 长段落**（107 字，18s） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/02_lingxiaorou_107c_baseline_q8.wav)（`2.04`） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/02_lingxiaorou_107c_planA_ar_q8.wav)（`1.06`¹）· [A/B MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_lingxiaorou_107c.mp4) | **1.92×** · 余弦 **`0.930`** |
| **3. 长段落无参考**（107 字，23s） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/03_uncond_107c_baseline_q8.wav)（`1.92`） | [WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/03_uncond_107c_planA_ar_q8.wav)（**`0.97`**）· [A/B MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_uncond_107c.mp4) | **1.98×** · 内存 `-34%` |

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
   - **第二轮（`AR=q8` 强开 `fused_fp32_act=1` 机械音）**：在 `灵小柔` 107 字长段落精听对比中又抓出强行移除 AR 主干 13 处上游 BF16 舍入会在第 2 块后半段触发一段机械音；改为 **`--fused-fp32-act auto`**（保留 AR 上游 BF16 舍入，仅 4-bit VAE 走融合 FP32）后，长段机械音 100% 消失，`灵小柔` 107 字音色余弦升至 **`0.9295`**，同时常规短句/长段仍稳守 **`RTF 0.86–0.97`**。

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
cmake --build .cache/bench-build --target ming3_rtf ming3_fast_rtf ming3_fast_profile_rtf ming3_fast_engine_test -j 2
ctest --test-dir .cache/bench-build --output-on-failure
```

### 3. 运行 ABBA 基准对照

```bash
# 方案 A（推荐高音质：AR=q8 + fused=auto 保留 AR BF16 + Flow/Agg=q4g64 8 步 + VAE=q4_0）
python3 benchmarks/compare_ming_fast.py --fixture zh_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --agg-quant q4g64 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --agg-quant q4g64 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1 --skip-regressions

# 方案 A + 官方内置音色「灵小柔」
INSTRUCT='{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null, "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}, "IP": "灵小柔"}]}'
python3 benchmarks/compare_ming_fast.py --fixture zh_lingxiaorou_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --agg-quant q4g64 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1 \
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
