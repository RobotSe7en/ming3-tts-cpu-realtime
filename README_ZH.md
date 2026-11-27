# Ming-3.0-TTS 0.5B x86 CPU 实时推理优化（`RTF < 1.0`）

[English](README.md) | [中文说明](README_ZH.md) | [完整实验演进记录](docs/experiments.md)

在一台 6 年前的入门级 4 核桌面 CPU（**Intel Core i3-10100 @ 3.60 GHz**，TrueNAS Linux 虚拟机，仅 4 线程、无独显、无核显直通）上，将 **Ming-3.0-TTS 0.5B** 纯 CPU 推理从原版 `RTF 1.72–1.99` 优化至 **`RTF 0.79–0.95`（快于实时播放）**——直接复用官方 **`Q8_0` GGUF** 权重（加载期并行重排约 `2.3 秒`，无需另转模型），零 swap，峰值内存最高降低 **46%**。

本项目以独立 C++ AVX2 微内核与可重放补丁形式构建于 [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp)（固定 commit `7d20b414`，Apache 2.0）之上，不修改上游基线源码。

---

## 核心 ABBA 对照结果

以下全部数据均在一致的 `systemd-run` 资源约束下（**4 线程、`CPUQuota=400%`、`MemoryMax=6G`、`MemorySwapMax=0`**）按 **ABBA** 交替顺序测得（`原版 → 优化版 → 优化版 → 原版`，每块预热 1 次、测量 2 次，每组累计 4 次正式测量）；单次及跨块生成的 WAV 均通过逐字节 SHA-256 确定性校验。

### 一、三套 `RTF ≤ 1` 运行方案（同一二进制、同一份 `Q8_0` GGUF）

| 方案 | 核心配置 | 短句 (`zh_short.txt`) 中位 RTF | 107 字自动分块长段落 中位 RTF | `灵小柔` 30 字短句 中位 RTF | 长段落峰值 RSS | 特点与适用场景 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **原版基线** | 未修改 `audio.cpp`（`Q8_0`，10 步全步 CFG） | `1.69–1.82` | `1.91–1.99` | `1.8066` | `1.98 GiB` | 对照基线 |
| **方案 A（推荐：高音质均衡版）** | `AR=q8`（融合 FP32 激活）+ `Flow/Agg=q4g64`（8 步 sway，全步 CFG）+ `VAE=q4_0` | **`0.8717`**（**1.94×**） | **`0.9327`**（**2.05×**） | **`0.9525`**（**1.90×**） | **`1.25 GiB`**（-37%） | **实听音质最好**：四组测试全部 `0.0000` CER，`灵小柔` 说话人余弦 **`0.9539`**，无长文第 2 段金属感 |
| **方案 B（极速 / 最低内存版）** | `AR=q4_0` + `Flow/Agg=q4g64`（8 步 sway，全步 CFG）+ `VAE=q4_0` | **`0.7884`**（**2.18×**） | **`0.8747`**（**2.25×**） | `0.8674`* | **`1.09 GiB`**（-45%） | 速度最快、抗抖动裕度最大、零削波；适合短句或固定音色场景 |
| **方案 C（完整 10 步官方采样版）** | `AR=q4_0` + `Flow/Agg=q4g128`（官方完整 10 步，全步 CFG）+ `VAE=q4_0` | **`0.8917`**（**2.05×**） | **`0.9881`**（**2.01×**） | — | **`1.07 GiB`**（-46%） | 不改动任何采样步数与 CFG 策略，纯靠引擎与 4-bit 访存压进 1.0 |

*\*方案 B 在 `灵小柔` 30 字短句为单次测量（`CER=0.0000`，`clip=0.000000`，说话人余弦 `0.8501`）。*

### 二、主推荐方案 A（`AR=q8`）完整 ABBA 与自动质量筛查明细

| 测试集 | 原版 RTF 中位数 | 方案 A RTF 中位数 | 方案 A 四次 RTF 分布 | 加速比 | 峰值 RSS（原版 → 方案 A） | ASR CER（原版 → 方案 A） | CAMPPlus 音色余弦 | 削波比例 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `zh_short.txt`（4.0s） | `1.6940` | **`0.8717`** | `0.9018, 0.8599, 0.8787, 0.8647` | **1.94×** | `1.33 → 1.24 GiB` | `0.0000 → 0.0000` | `0.8579`（无参考） | `0.000000` |
| `zh_paragraph.txt`（107 字，20.9s） | `1.9088` | **`0.9327`** | `0.9292, 0.9333, 0.9365, 0.9321` | **2.05×** | `1.98 → 1.25 GiB` | `0.0625 → 0.0000` | `0.8785`（无参考） | `0.000039` |
| `zh_lingxiaorou_short.txt`（`灵小柔`，6.0s） | `1.8066` | **`0.9525`** | `0.9531, 0.9488, 0.9716, 0.9519` | **1.90×** | `1.35 → 1.24 GiB` | `0.0000 → 0.0000` | **`0.9539`**（通过） | `0.000000` |
| `zh_paragraph.txt`（`灵小柔`，18.5s）¹ | `2.0229` | `1.0766`¹ | `1.0307, 1.0146, 1.1226, 1.1655` | **1.88×** | `1.97 → 1.26 GiB` | `0.0000 → 0.0000` | **`0.9166`**（通过） | `0.000000` |
| `zh_regression.txt`（单次回归） | `1.6888` | **`0.8861`** | `0.8861` | **1.91×** | `1.32 → 1.24 GiB` | — | — | — |
| `zh_long_regression.txt`（单次回归） | `1.8672` | **`0.9004`** | `0.9004` | **2.07×** | `1.93 → 1.25 GiB` | — | — | — |

> ¹ 在固定版本 `7d20b414` 中，`灵小柔` 通过 `template_name=freeform` 传入（绕过旧版仅含英文音色的 `voice_id` 枚举校验），不会置位 `controls["ip"]`，因此长文本仍多跑了首段重锚定且每块多约 55 个 JSON token；在新版原生支持 `template_name=ip, voice_id=灵小柔` 的 `audio.cpp` 中会自动跳过首段重锚定，长文本速度会更快。

全部脱敏后的 ABBA 汇总与 SenseVoice INT8 ASR + CAMPPlus 筛查 JSON 保存在 [`benchmarks/published_results/`](benchmarks/published_results/)。自动化 ASR 与说话人余弦仅用于回归筛查，不等同于正式人工 MOS 评测。

---

## 音频示例（原版 vs 方案 A 实听对比）

以下均为上述正式 ABBA 压测直接输出的未后期 44.1 kHz PCM16 原始音频（托管在 [GitHub Release `audio-samples`](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/tag/audio-samples)，避免大体积音频污染 Git 仓库）：

| 示例 | 音色与文本 | A：原版基线（`Q8_0`） | B：方案 A 实时版（`AR=q8`, `Flow=q4g64`） | A/B 波形对比视频 | 关键指标 |
| --- | --- | --- | --- | --- | --- |
| **1. 短句（推荐试听）** | 内置「灵小柔」（30 字，6.0s） | [原版 WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/01_lingxiaorou_30c_baseline_q8.wav)（`RTF 1.81`） | [方案 A WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/01_lingxiaorou_30c_planA_ar_q8_rtf095.wav)（**`RTF 0.95`**） | [A/B 对比 MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_lingxiaorou_30c.mp4) | **1.90×**，CER `0.0000`，余弦 **`0.9539`** |
| **2. 长段落（无参考）** | 自动分块（107 字，20.9s） | [原版 WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/03_uncond_107c_baseline_q8.wav)（`RTF 1.91`） | [方案 A WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/03_uncond_107c_planA_ar_q8_rtf093.wav)（**`RTF 0.93`**） | [A/B 对比 MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_uncond_107c.mp4) | **2.05×**，CER `0.0000`，峰值内存 `-37%` |
| **3. 长段落（「灵小柔」）** | 内置「灵小柔」（107 字，18.5s） | [原版 WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/02_lingxiaorou_107c_baseline_q8.wav)（`RTF 2.02`） | [方案 A WAV](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/02_lingxiaorou_107c_planA_ar_q8.wav)（`RTF 1.08`¹） | [A/B 对比 MP4](https://github.com/RobotSe7en/ming3-tts-cpu-realtime/releases/download/audio-samples/ab_demo_lingxiaorou_107c.mp4) | **1.88×**，CER `0.0000`，余弦 **`0.9166`** |

> 说明：所有样本均在 4 线程 CPU、固定种子 `1895` 下确定性生成，未做任何降噪、响度归一化或后期剪辑。

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
   - 加载期 OpenMP 并行从 `Q8_0` GGUF 重排到 GGML `ggml_backend_cpu_repack_buffer_type()`（`Q4_0_8x8_q8_0`，总加载耗时约 `2.3 秒`）；关闭约 550 次中间人工 BF16 截断（`--fused-fp32-act 1`）解锁融合 `SwiGLU`。
4. **8 步均匀 `u` Sway 网格与首段 Latent 复用**：
   - 修复早期非均匀 EPSS 节点末步跳变 `dt = 0.707` 导致的吞字问题，在 `u` 空间均匀采样后做余弦 sway 变换并全程保留 `CFG = 2.0`；无参考长文本分块时按精确 `patch_samples = output_dim * patch_size` 直接复用首段 `latent`（保留 `CAMPPlus`），省约 `1.0 秒` 和 `380 MiB` 内存（可用 `--reuse-anchor-latent 0` 关闭）。
5. **实听排错（为什么方案 A 坚持保留 `AR=q8`）**：
   - 仅看 SenseVoice ASR CER 时，`AR=q4_0` 长文本表现很好（`0.02–0.03`），但与 Mac Metal 输出盲听对比后发现：**第 2 分块起出现明显金属/电话压缩感**。
   - 逐项隔离（`reuse=0` → `VAE=q8` → `Agg=q8` → `AR=q8`）精准定位元凶为 24 层 `AR=q4_0` 在长声学上下文下的隐状态退化；将 AR 恢复为原版 `Q8_0` 并开启 `--fused-fp32-act 1` 后金属感 100% 消失，`灵小柔` 音色余弦升至 **`0.9539`**，且依然稳守 **`RTF 0.87–0.95`**。

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
# 方案 A（推荐高音质：AR=q8 + 融合 FP32 激活 + Flow/Agg=q4g64 8 步 + VAE=q4_0）
python3 benchmarks/compare_ming_fast.py --fixture zh_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --agg-quant q4g64 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act 1 --flow-steps 8 --cfg-interval 1
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --agg-quant q4g64 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act 1 --flow-steps 8 --cfg-interval 1 --skip-regressions

# 方案 A + 官方内置音色「灵小柔」
INSTRUCT='{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null, "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}, "IP": "灵小柔"}]}'
python3 benchmarks/compare_ming_fast.py --fixture zh_lingxiaorou_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --agg-quant q4g64 --ar-quant q8 --vae-quant q4_0 --fused-fp32-act 1 --flow-steps 8 --cfg-interval 1 \
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
