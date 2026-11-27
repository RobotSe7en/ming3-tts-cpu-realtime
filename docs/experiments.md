# x86 CPU TTS 推理优化与基准

用于优化或重新实现 x86 CPU 上的 TTS 推理。目前只以 **RTF** 为优化指标，同时监测 CPU 和内存，控制对宿主机的影响。

当前优化对象为 Ming-3.0-TTS 0.5B。Qwen 的已有数据保留为历史对照。

## 准备源码和模型

需要 Linux、Git、CMake 3.20+、支持 C++17 和 OpenMP 的编译器、Python 3，以及可用的 systemd 用户服务。无需安装 PyTorch。

首次测试使用未修改的 `audio.cpp`，版本固定为 `7d20b414da088c3ffe56bd8803398d9d158cb847`：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/0xShug0/audio.cpp.git .cache/audio.cpp
git -C .cache/audio.cpp fetch --depth 1 origin 7d20b414da088c3ffe56bd8803398d9d158cb847
git -C .cache/audio.cpp checkout --detach 7d20b414da088c3ffe56bd8803398d9d158cb847
git -C .cache/audio.cpp sparse-checkout set app include src external model_specs tools webui/native/demo_voices
```

模型来自 [ModelScope 上的官方 audio-cpp GGUF 仓库](https://modelscope.cn/models/audio-cpp/Ming-3.0-TTS-0.5B-GGUF)。下载时绕过代理：

```bash
mkdir -p models
curl --noproxy '*' --fail --location --retry 3 \
  --output models/ming-3.0-tts-0.5b-q8_0.gguf \
  https://modelscope.cn/models/audio-cpp/Ming-3.0-TTS-0.5B-GGUF/resolve/master/ming-3.0-tts-0.5b-q8_0.gguf
sha256sum models/ming-3.0-tts-0.5b-q8_0.gguf
```

文件大小应为 `1742280896` 字节，SHA-256 应为：

```text
fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1
```

## 构建与运行

所有命令在仓库根目录执行。仅启用 CPU 后端，编译 Ming 和 Qwen3 TTS；构建限制为两路并行、最多 4 GiB 内存和零 swap：

```bash
cmake -S benchmarks -B .cache/bench-build -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODELS=ming3_tts,qwen3_tts
systemd-run --user --wait --pipe --collect \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=128 -p Nice=10 \
  cmake --build "$PWD/.cache/bench-build" --target ming3_rtf -j 2
python3 benchmarks/run_ming3.py --threads 2 --warmups 1 --iterations 3
```

推理默认限制为 6 GiB 内存、两个逻辑 CPU 的计算配额和零 swap。系统可用内存连续三秒低于 4 GiB 时会停止测试。虚拟机报告的逻辑 CPU 数不用于自动扩展线程数。

上游 FFT 线程池会按系统报告的 CPU 数创建等待线程，因此当前虚拟机可能出现约 130 个进程线程；有效推理并行度和 CPU 总计算配额仍由上述设置控制。测试服务允许最多 256 个任务，避免等待线程触发线程创建失败。

## 测量口径与结果

- 固定输入见 `benchmarks/zh_short.txt`；中文、普通离线 TTS、种子 `1895`，其余生成选项保持上游默认值。
- 同一个模型和会话先预热一次，再运行三次。计时覆盖 `session->run`，包含文本处理、声学生成和音频解码，排除加载、预热和 WAV 写入。
- `RTF = 合成墙钟秒数 / 实际生成音频秒数`，越低越好；小于 `1` 表示生成速度快于实时播放。每次音频时长由样本数、声道数和采样率计算。
- `benchmarks/results/` 下每次测试单独保存 `metadata.json`、`events.jsonl`、`summary.json`、逐秒资源记录和生成音频。CPU 占用以一个逻辑 CPU 为 `100%`；RSS 和 cgroup 内存分别记录。
- 性能结论使用预热后三次 RTF 的中位数。该结果对应此虚拟机、线程配额、模型精度和短文本输入。

## 首次基线：2026-11-09

i3-10100 / TrueNAS 虚拟机，Q8_0，CPU 后端，推理并行度 2，CPU 配额 200%。预热一次后：

| 次数 | 合成耗时（秒） | 音频时长（秒） | RTF |
| --- | ---: | ---: | ---: |
| 1 | 18.831 | 4.000 | 4.708 |
| 2 | 19.814 | 4.000 | 4.953 |
| 3 | 17.397 | 4.000 | 4.349 |

**RTF 中位数：4.708，未达到实时生成。** 测量期间 CPU 平均占用分别为 189.2%、186.5%、190.9%，峰值 RSS 为 1.320 GiB，测试进程未使用 swap。三次生成的单声道 44.1 kHz WAV 的 SHA-256 完全一致。

原始记录位于 `benchmarks/results/ming3-q8-t2-20261109T085831Z/`。该数据不代表整颗 CPU 使用全部八个硬件线程时的性能。

## Ming 分阶段计时

计时仅加在独立 worktree 的四个源文件中，原始 checkout 保持不变。构建时将计时对象链接到单独的 `ming3_profile_rtf`，复用原有运行库。计时器在内存中累加，嵌套阶段扣除子阶段耗时，合成结束后统一输出。

```bash
git -C .cache/audio.cpp worktree add --no-checkout --detach "$PWD/.cache/audio.cpp-profile" HEAD
git -C .cache/audio.cpp-profile sparse-checkout set app include src external model_specs tools webui/native/demo_voices
git -C .cache/audio.cpp-profile checkout --detach HEAD
git -C .cache/audio.cpp-profile apply "$PWD/benchmarks/ming3_stage_timing.patch"
cmake -S benchmarks -B .cache/bench-build -DTTS_BUILD_MING_PROFILE=ON
systemd-run --user --wait --pipe --collect \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=128 -p Nice=10 \
  cmake --build "$PWD/.cache/bench-build" --target ming3_profile_rtf -j 2
python3 benchmarks/run_ming3.py --profile --threads 2 --warmups 1 --iterations 3
```

本轮三次均生成 4 秒音频，以下为平均耗时和总耗时占比：

| 阶段 | 平均耗时（秒） | 占比 |
| --- | ---: | ---: |
| Flow 采样 | 8.457 | 72.60% |
| 主干预填充及自回归解码 | 1.472 | 12.64% |
| VAE 解码（不含 iSTFT） | 1.259 | 10.81% |
| 潜变量投影 | 0.455 | 3.91% |
| iSTFT 音频重建 | 0.004 | 0.04% |
| 文本前端、随机数及其他 | 0.002 | 0.01% |

每次生成 50 个声学 patch，调用 500 次 flow 计算图；flow 中约 8.451 秒用于计算图执行，其余约 0.006 秒在图外。三次 RTF 中位数为 **2.947**。原始数据位于 `benchmarks/results/ming3-q8-profile-t2-20261109T091311Z/`。

随后未插桩的原始二进制对照测得 RTF 中位数 **2.844**，见 `benchmarks/results/ming3-q8-t2-20261109T091440Z/`。两组及首次基线的 WAV SHA-256 全部一致。原始程序未经优化也明显快于首次测试，说明运行环境存在波动；不能将这次差异计为优化收益。

## Qwen3-TTS 0.6B Base

audio.cpp 支持该模型的 Q8_0 包。模型来自 [ModelScope 镜像](https://modelscope.cn/models/HereIsMark/audio.cpp-gguf/files)，下载到同一个 `models/` 目录：

```bash
curl --noproxy '*' --fail --location --retry 3 \
  --output models/qwen3-tts-12hz-0.6b-base-q8_0.gguf \
  https://modelscope.cn/models/HereIsMark/audio.cpp-gguf/resolve/master/Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf
python3 benchmarks/run_qwen3.py --threads 2 --warmups 1 --iterations 3
```

文件大小为 `1991217728` 字节，SHA-256 为 `c7d0e6c9485a25ae1f6a4c0796f0023230a0051c44cbd2bd05d73770f2cce637`。运行器会验证模型大小和校验值。

Base 使用参考音频进行克隆：本用例采用上游 `demo_2_man.wav`（7.616 秒）及 `benchmarks/qwen_reference.txt` 中的原文。同一参考在预热和三次测量中复用，保留默认提示缓存。目标中文文本与 Ming 相同，其余生成选项采用默认值，种子固定为 `1895`。这是参考条件下的 Base 用例，与 Ming 的无参考普通 TTS 路径有所不同。

本机 2 路并行、200% CPU 配额下，预热一次后测三次：

| 次数 | 合成耗时（秒） | 音频时长（秒） | RTF |
| --- | ---: | ---: | ---: |
| 1 | 17.190 | 3.680 | 4.671 |
| 2 | 17.109 | 3.680 | 4.649 |
| 3 | 15.369 | 3.680 | 4.176 |

**RTF 中位数：4.649。** 峰值 RSS 为 3.579 GiB，CPU 平均占用为 191.9%–192.4%，测试进程 swap 为零。三次生成的 WAV 完全一致。结果位于 `benchmarks/results/qwen3-0.6b-base-q8-t2-20261109T091807Z/`。

## 4 线程测试：2026-11-09

沿用相同 Q8 模型、文本、种子及生成选项，改为 4 路推理并行度和 400% CPU 配额；保持 6 GiB 内存上限、禁止 swap。两个模型依次运行，各预热一次后测三次。

```bash
python3 benchmarks/run_ming3.py --threads 4 --warmups 1 --iterations 3
python3 benchmarks/run_qwen3.py --threads 4 --warmups 1 --iterations 3
```

| 模型 | 三次 RTF | RTF 中位数 | 峰值 RSS | CPU 占用范围 |
| --- | --- | ---: | ---: | ---: |
| Ming 0.5B Q8 | 2.625、1.914、1.731 | **1.914** | 1.323 GiB | 393.8%–398.2% |
| Qwen3-TTS 0.6B Base Q8 | 5.875、6.721、5.257 | **5.875** | 3.572 GiB | 324.8%–340.3% |

Ming 每次生成 4.000 秒音频，Qwen 每次生成 3.680 秒音频。所有 WAV 均与对应模型此前 2 线程的输出 SHA-256 一致。测试进程 swap 均为零，没有触发内存上限或 OOM；系统可用内存最低分别为 14.20 GiB、9.19 GiB。

两者本轮均未达到 RTF ≤ 1。Qwen 的 4 线程结果没有优于此前 2 线程记录，但这是跨时段测试，且本轮机器负载更高；未据此计算严格的线程加速比或认定退化原因。

原始数据：

- Ming：`benchmarks/results/ming3-q8-t4-20261109T121144Z/`。
- Qwen：`benchmarks/results/qwen3-0.6b-base-q8-t4-20261109T121256Z/`。

## Ming CPU 路径优化

当前改动位于 `src/patches/ming_flow_cpu.patch`，针对固定版本 `7d20b414da088c3ffe56bd8803398d9d158cb847`。原始 checkout 保持不变；独立的 `ming3_cpu_opt_rtf` 链接替换后的 flow 对象。

改动包括 CPU QKV 投影打包、预计算固定位置的 RoPE 表、融合保留逐步 BF16 舍入的旋转计算，以及最后一层只计算最终输出 token 的后续路径。注意力的 K/V 上下文仍完整保留。Q8 权重、10 次 flow 迭代、随机种子及生成参数保持一致。其他后端保留原来的执行路径。

首次创建优化 worktree 并应用组合补丁：

```bash
git -C .cache/audio.cpp worktree add --no-checkout --detach "$PWD/.cache/audio.cpp-ming-opt" HEAD
git -C .cache/audio.cpp-ming-opt sparse-checkout set app include src external model_specs tools webui/native/demo_voices
git -C .cache/audio.cpp-ming-opt checkout --detach HEAD
git -C .cache/audio.cpp-ming-opt apply "$PWD/src/patches/ming_flow_cpu.patch"
cmake -S benchmarks -B .cache/bench-build -DTTS_BUILD_MING_CPU_OPT=ON -DTTS_BUILD_MING_OP_PROFILE=OFF
systemd-run --user --wait --pipe --collect \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=128 -p Nice=10 \
  cmake --build "$PWD/.cache/bench-build" --target ming3_cpu_opt_rtf ming3_cpu_rope_test -j 2
ctest --test-dir .cache/bench-build -R ming3_cpu_rope_parity --output-on-failure
python3 benchmarks/run_ming3.py --variant cpu-opt --threads 4
```

`ming_flow_qkv_only.patch` 和 `ming_flow_qkv_rope.patch` 保存分步实验快照；`ming_flow_cpu.patch` 包含最终组合。单独 QKV 打包未测得明确收益，组合结果通过交替对照验证。

`tests/ming_cpu_rope_test.cpp` 对 12 组 batch、长度和头数配置重复运行，要求融合结果与原始 GGML 图逐位一致。端到端对照还检查每次生成的 WAV SHA-256，覆盖短文本、另一句中文及自动分块文本。固定位置表标为长期保留的输入，防止其缓存空间在图执行中被复用。

长段落基准使用 `benchmarks/zh_paragraph.txt`，共 107 个字符，保留默认自动分块：

```bash
python3 benchmarks/compare_ming_cpu.py --fixture zh_paragraph.txt --iterations 2 --skip-regressions
```

对照顺序为原版、改版、改版、原版（ABBA），每块预热一次、测两次，每个版本累计四次测量；依次运行以控制资源占用。命令保存各块完整记录、音频一致性检查以及汇总 `comparison.json`。

短文本先行验证：原版 RTF 中位数 `1.7293`，改版 `1.6248`，降低 `6.04%`；各六次测量，峰值 RSS 均约 `1.324 GiB`，进程 swap 为零。全部输出逐字节一致。记录位于 `benchmarks/results/ming-cpu-comparison-t4-20261109T131208Z/`。

107 字长段落交替对照的结果如下。每次生成的音频均为 22.100 秒，每个版本累计四次测量：

| 指标 | 原版 | CPU 优化版 |
| --- | ---: | ---: |
| RTF 中位数 | 1.9842 | **1.8444** |
| 合成耗时中位数 | 43.851 秒 | **40.762 秒** |
| 峰值 RSS | 1.988 GiB | 1.968 GiB |
| 进程 swap | 0 | 0 |

本次 RTF 降低 **7.05%**，对应约 **1.076 倍**速度。全部长文本输出与原版逐字节一致，尚未达到 RTF ≤ 1。原版四次 RTF 为 `1.9209、1.9507、2.0178、2.3106`，改版为 `1.8451、1.8329、1.8437、1.9761`；仍存在 VM 运行条件波动，应保留完整分布。汇总及各块原始记录位于 `benchmarks/results/ming-cpu-comparison-t4-20261109T131723Z/comparison.json`。

四组文本（初始短句、另一句中文、11 秒分块语音、22.1 秒段落）均通过端到端音频一致性检查。CTest 的 12 组形状测试也通过。较早未通过回归的实验保留了失败记录，不计入上述结果。

如需算子诊断，在已经应用阶段计时补丁的 profile worktree 中应用 `benchmarks/ming3_cpu_op_timing.patch`，并启用 CMake 选项 `TTS_BUILD_MING_OP_PROFILE=ON`。日志的 `[MING_OP_PROFILE]` 行包含预热在内的 flow 算子墙钟耗时及同步开销，仅用于诊断，不用于最终 RTF 对照。

## Ming 专用 CPU 引擎（实验版）

`src/ming_cpu/` 实现 Ming Flow 和潜变量投影使用的专用 CPU 算子，入口为 `ming3_engine_rtf`。当前仍复用 audio.cpp 的模型加载、文本前端、自回归主干、VAE 和 GGML 图执行设施；这是可运行的专用计算路径，尚未重写整个推理运行时。保持 Q8_0 权重、10 步 Flow、种子与生成参数一致，以逐字节音频一致性约束本轮优化。

在之前的 QKV、RoPE 和末层裁剪基础上，新增：

- 小矩阵 Q8_0 专用调度，按输出通道静态分工，四个线程共享激活量化结果，各自调用 GGML CPU 矩阵微内核并完成偏置与 BF16 舍入，无需在矩阵计算和后处理之间同步。实际检查 CPU 的 AVX2/FMA/F16C 功能，保留备用内核、普通 CPU 路径和大矩阵回退。
- RMSNorm、BF16 舍入与权重乘法融合，以及残差加法与舍入融合，保持原来的中间舍入顺序。
- BF16 GELU 查表：在加载期间用本次构建的原版 GGML GELU 计算全部 65,536 种编码，运行时查表得到舍入后的结果。每个网络增加 256 KiB 表，不引入新的 GELU 近似。
- FFT 工作线程上限固定为 4，避免这台 VM 报告的 128 vCPU 导致创建 128 个等待线程。推理及性能对照固定使用 4 线程、400% CPU 配额，内存上限仍为 6 GiB、禁止 swap。

VM 内可见 AVX2、FMA、F16C；当前内核需要的指令没有被屏蔽。虚拟拓扑为 8 核、每核 16 线程，不能当作宿主机实际并行能力。无需修改 VM 设置。新增记录包含虚拟化类型、客体 `/proc/stat` 原始计数和逐秒累计 steal ticks；这些计数不能单独量化裸机与 VM 的性能差距。

创建独立 worktree，直接应用完整引擎补丁（无需先应用旧的 CPU 补丁）：

```bash
git -C .cache/audio.cpp worktree add --detach "$PWD/.cache/audio.cpp-ming-engine" HEAD
git -C .cache/audio.cpp-ming-engine apply "$PWD/src/patches/ming_cpu_engine.patch"
cmake -S benchmarks -B .cache/bench-build -DCMAKE_BUILD_TYPE=Release -DTTS_BUILD_MING_ENGINE=ON
systemd-run --user --wait --pipe --collect \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=128 -p Nice=10 \
  cmake --build "$PWD/.cache/bench-build" --target ming3_engine_rtf ming3_cpu_linear_test -j 2
ctest --test-dir .cache/bench-build -R ming3_cpu_linear_parity --output-on-failure
python3 benchmarks/run_ming3.py --variant engine --threads 4 --warmups 1 --iterations 3
python3 benchmarks/compare_ming_cpu.py --candidate engine --threads 4 --iterations 2
python3 benchmarks/compare_ming_cpu.py --candidate engine --threads 4 --fixture zh_paragraph.txt --iterations 2 --skip-regressions
```

线性层、归一化、残差和 GELU 的一致性测试覆盖 72 组输入形状与线程配置，各重复四次，检查 AVX2 和普通 CPU 路径；另穷举全部 BF16 GELU 编码。`compare_ming_cpu.py` 对同一输入按 ABBA 顺序运行，要求各次 WAV SHA-256 完全一致，默认额外检查另一句中文和自动分块回归文本。运行元数据保存引擎补丁、专用算子源码、编译配置、二进制与模型校验值。

专用矩阵调度使用此固定版本 GGML 的内部 GEMM 入口；升级上游后需要重新检查接口与数值一致性。默认启用的 `GGML_LLAMAFILE` 提供该入口，关闭时保留独立的小矩阵实现。

需要分阶段计时时，先按前文准备 profile worktree，然后：

```bash
cmake -S benchmarks -B .cache/bench-build -DTTS_BUILD_MING_ENGINE=ON -DTTS_BUILD_MING_PROFILE=ON -DTTS_BUILD_MING_OP_PROFILE=OFF
systemd-run --user --wait --pipe --collect \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=128 -p Nice=10 \
  cmake --build "$PWD/.cache/bench-build" --target ming3_engine_profile_rtf -j 2
python3 benchmarks/run_ming3.py --variant engine --profile --threads 4 --warmups 1 --iterations 1
```

### 专用路径结果：2026-11-10（本地时间）

最终配置采用按输出通道分工的原生 GEMM 调度。相同 Q8_0 模型、种子、10 步采样、4 线程、400% CPU 配额和 CPU 亲和性下，短句及长段落分别运行 ABBA；每块预热一次、测量两次，每个版本各四次正式测量：

| 输入 | 原版 RTF 中位数 | 专用路径 RTF 中位数 | RTF 降低 | 专用路径峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| 4.000 秒短句 | 1.7618 | **1.5836** | 10.12% | 1.323 GiB |
| 22.100 秒自动分块段落 | 2.1577 | **1.8262** | 15.36% | 1.982 GiB |

**尚未达到 RTF ≤ 1。** 两组是本轮对照结果，不与之前的 CPU 优化版做跨时段加速比比较。VM 波动仍明显，应同时查看全部正式测量：

| 输入 | 原版各次 RTF | 专用路径各次 RTF |
| --- | --- | --- |
| 短句 | 1.7488、1.7431、1.7758、1.7749 | 1.6388、1.5485、1.6055、1.5617 |
| 长段落 | 2.0134、2.7622、2.2719、2.0436 | 1.7994、2.5979、1.8177、1.8346 |

四组文本（短句、另一句中文、11 秒分块文本、22.1 秒段落）的输出均与原版 WAV 逐字节一致。两项 CTest 通过，包含 72 组融合算子配置、65,536 种 BF16 GELU 编码及 12 组 RoPE 配置。专用路径进程总线程数峰值为 8，包含推理线程和 FFT 等待线程；计算并行度与 CPU 配额仍为 4。所有测试进程 swap 为零，cgroup 内存峰值约 1.997 GiB，未触发内存限制或 OOM。

最终诊断运行中，Flow 图执行约 4.520 秒，是 4 秒音频合成的主要瓶颈；阶段耗时之和与合成计时之差约 1.425 微秒，音频哈希也通过。诊断计时单独保存，不并入上述 ABBA 性能数据。

完整结果（本地原始归档，不纳入 Git 追踪）：

- 短句 ABBA：`benchmarks/results/ming-engine-comparison-t4-20261109T161127Z/comparison.json`
- 长段落 ABBA：`benchmarks/results/ming-engine-comparison-t4-20261109T161522Z/comparison.json`
- 汇总校验、源码与二进制校验值：`benchmarks/results/ming-engine-native-validation-t4-20261109T161127Z/validation.json`（同目录含确切命令、CTest 日志及各次启动日志）

前一版自写小矩阵内核的结果也保留在 `benchmarks/results/ming-engine-validation-t4-20261109T155013Z/`：短句未测得收益，长文本存在较大波动；这些记录不混入最终配置的数据。原始 checkout 和原有 CPU 优化路径均保留。

## 虚拟机核显核查与 CPU 带宽屋顶线分析（2026-11-26）

- **核显状态**：当前虚拟机 PCI 设备仅见 `Red Hat, Inc. QXL paravirtual graphic card`，无 `/dev/dri/renderD*` 渲染节点，也未直通 Intel iGPU，且 `benchmarks/CMakeLists.txt` 仅启用 CPU 后端（`ENGINE_ENABLE_VULKAN=OFF`）。即便通过 VFIO 直通宿主机 i3-10100 的 UHD 630（Gen9.5，24 EU，无 DP4A/XMX），它仍与 CPU 共享同一条双通道 DDR4 内存总线（本机 4 线程实测读取带宽约 `25.35 GB/s`），而 Ming 耗时占比近 70% 的 Flow 阶段由 500 次极小批量（`M=6`）串行前向组成，GPU 提交与同步开销反而难以优于 CPU 本地 L1/L2 流水。
- **Q8_0 物理下界**：在 4 核 8 线程 i3-10100（AVX2/FMA/F16C，无 AVX-512/VNNI）和 `25.35 GB/s` 内存带宽下，严格保持 Q8_0 权重与 10 步全步 CFG 时，Flow 500 次前向仅读取约 `89.3 GB` 权重即需约 `3.52 秒`（对应 4 秒音频仅 Flow 项即 `RTF ≥ 0.88`），加上 AR、VAE 与投影已超过 `1.35`；因此突破 `RTF ≤ 1` 需要同时消除计算图调度开销与降低访存字节数（4-bit 权重重排）。

## Ming 常驻直连 CPU 引擎与加载期 4-bit 重排（RTF ≤ 1 达成）

新增实现位于 `src/ming_cpu/flow_engine.{h,cpp}`、`src/ming_cpu/repack_weights.{h,cpp}` 与可重放补丁 `src/patches/ming_fast_cpu.patch`（基于固定版本 `7d20b414da088c3ffe56bd8803398d9d158cb847`，未修改 `.cache/audio.cpp`），独立入口为 `ming3_fast_rtf`（含分阶段计时版 `ming3_fast_profile_rtf`）。仍直接读取校验通过的 `models/ming-3.0-tts-0.5b-q8_0.gguf`，在加载期并行重排权重（总加载耗时约 `2.3 秒`），无需额外模型文件。

核心改动：

- **常驻 OpenMP 直连 C++ Flow / Aggregator 引擎（`DirectPatchNetwork`）**：用单次跨全部 Flow 步的常驻 `#pragma omp parallel` 区域取代每步 ~200 个 GGML 图节点与同步屏障；预计算并缓存每时间步 `time_mlp`、每 patch 缓存 `cond_emb`/`hist_emb` 与第 0 层 `hist_qkv`、共享第 0 层 cond/uncond `x_emb` QKV、加载期预计算 Aggregator `cls_qkv`；L1 栈内融合 `RMSNorm + BF16 + 量化`、融合 `RoPE + 多头注意力 + BF16` 与 `bias + BF16 + residual/GELU 查表`；针对 `M=1,2` 增加 4 通道展开内核，支持 Aggregator 批量处理（`chunk ≤ 3`）并缓存长文本分块间重复的 `prompt_latents` 投影。
- **组量化非对称 4-bit AVX2 微内核（`Q4_1_G64` / `Q4_1_G128`）**：按 64 或 128 维分组打包为无符号 4-bit 权值与每通道连续 32B 对齐 slab，配组主序 `Q8GroupInputBuffer` 消除 `4096` 步长 4 KiB L1d 组相联冲突；内层使用 `vpmaddubsw + vpaddw`（零 `vpsignb` 开销，最大中间和 `15,240 < 32,767` 无溢出）。全部 AVX2/FMA/F16C 路径均通过运行时 CPU 特性检测（`__builtin_cpu_supports`）门控，保留完整标量回退与 `Q8_0` 兼容路径。
- **AR / VAE 加载期 `CPU_REPACK`（`Q4_0_8x8_q8_0`）与融合 FP32 激活**：加载期并行将 AR（24 层）和 VAE（8 层）的打包 QKV、Gate+Up 与输出权重重排进 GGML `ggml_backend_cpu_repack_buffer_type()`，并在量化 CPU 路径关闭中间人工 BF16 截断以解锁融合 `SwiGLU`。
- **长文本分块首段 Latent 复用与均匀 `u` 8 步 Sway 网格**：自动分块锚定首段音色时保留 `CAMPPlus` 说话人提取，并按精确 `patch_samples = output_dim * patch_size` 直接切片复用首段已生成的 `latent`（可通过 `--reuse-anchor-latent 0` 关闭并回退完整 `MingAudioEncoder`），减少约 `1.0 秒` 重编码耗时与约 `380 MiB` 峰值内存；另提供 5–10 步 `flow_time_grid`（10 步与官方网格逐位一致，8 步使用均匀 `u` 后经余弦 sway 变换）。

创建独立 worktree、应用补丁、构建并运行单元测试与基准：

```bash
git -C .cache/audio.cpp worktree add --no-checkout --detach "$PWD/.cache/audio.cpp-ming-fast" HEAD
git -C .cache/audio.cpp-ming-fast sparse-checkout set app include src external model_specs tools webui/native/demo_voices
git -C .cache/audio.cpp-ming-fast checkout --detach HEAD
git -C .cache/audio.cpp-ming-fast apply "$PWD/src/patches/ming_fast_cpu.patch"
cmake -S benchmarks -B .cache/bench-build -DCMAKE_BUILD_TYPE=Release \
  -DTTS_BUILD_MING_ENGINE=ON -DTTS_BUILD_MING_FAST=ON -DTTS_BUILD_MING_PROFILE=ON
systemd-run --user --wait --pipe --collect \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=128 -p Nice=10 \
  cmake --build "$PWD/.cache/bench-build" --target ming3_fast_rtf ming3_fast_profile_rtf ming3_fast_engine_test -j 2
ctest --test-dir .cache/bench-build --output-on-failure

# 主推荐配置（q4g64 + 8 步 sway + 全步 CFG=2.0）
python3 benchmarks/compare_ming_fast.py --fixture zh_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --ar-quant q4_0 --vae-quant q4_0 --flow-steps 8 --cfg-interval 1 --reuse-anchor-latent 1
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g64 --ar-quant q4_0 --vae-quant q4_0 --flow-steps 8 --cfg-interval 1 --reuse-anchor-latent 1 --skip-regressions

# 保留完整 10 步官方采样网格配置（q4g128 + 10 步 + 全步 CFG=2.0）
python3 benchmarks/compare_ming_fast.py --fixture zh_short.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g128 --ar-quant q4_0 --vae-quant q4_0 --flow-steps 10 --cfg-interval 1 --reuse-anchor-latent 1
python3 benchmarks/compare_ming_fast.py --fixture zh_paragraph.txt --iterations 2 --threads 4 --reference baseline \
  --flow-quant q4g128 --ar-quant q4_0 --vae-quant q4_0 --flow-steps 10 --cfg-interval 1 --reuse-anchor-latent 1 --skip-regressions
```

### 正式 ABBA 验证结果：2026-11-26（i3-10100 4C/8T VM，4 线程 / 400% CPUQuota / 6 GiB MemoryMax / 0 swap）

以下均与未修改原版 `ming3_rtf`（`baseline`）按 ABBA 顺序交替对比，每块预热 1 次、测量 2 次，每组累计 4 次正式测量；所有单次及跨块生成的 WAV 均通过确定性 SHA-256 一致性校验，进程 swap 均为 `0`，5/5 CTest 全部通过（含 `ming3_fast_engine_parity`）。

#### 1. 主推荐配置：`q4g64 + ar/vae=q4_0 + 8 步 + 全步 CFG=2.0`

| 文本 | 原版 RTF 中位数 | 优化版 RTF 中位数 | RTF 降低 / 加速比 | 优化版各次 RTF | 峰值 RSS（原版 → 优化版） |
| --- | ---: | ---: | ---: | --- | ---: |
| `zh_short.txt`（短句） | 1.7187 | **0.7884** | 54.13% / **2.18×** | 0.7870, 0.7898, 0.8003, 0.7835 | 1.327 → **1.077 GiB** |
| `zh_paragraph.txt`（107 字自动分块） | 1.9723 | **0.8747** | 55.65% / **2.25×** | 0.8621, 0.8688, 0.8806, 0.8833 | 1.978 → **1.086 GiB** |
| `zh_regression.txt`（回归短句，单次） | 1.7587 | **0.8350** | 52.52% / 2.11× | 0.8350 | 1.321 → 1.077 GiB |
| `zh_long_regression.txt`（回归分块，单次） | 1.9427 | **0.8233** | 57.62% / 2.36× | 0.8233 | 1.934 → 1.086 GiB |

#### 2. 完整 10 步官方采样配置：`q4g128 + ar/vae=q4_0 + 10 步 + 全步 CFG=2.0`

| 文本 | 原版 RTF 中位数 | 优化版 RTF 中位数 | RTF 降低 / 加速比 | 优化版各次 RTF | 峰值 RSS（原版 → 优化版） |
| --- | ---: | ---: | ---: | --- | ---: |
| `zh_short.txt`（短句） | 1.8243 | **0.8917** | 51.12% / **2.05×** | 0.8876, 0.8940, 0.8893, 0.9098 | 1.327 → **1.059 GiB** |
| `zh_paragraph.txt`（107 字自动分块） | 1.9869 | **0.9881** | 50.27% / **2.01×** | 1.0011, 1.0291, 0.9751, 0.9666 | 1.978 → **1.068 GiB** |
| `zh_regression.txt`（回归短句，单次） | 1.8470 | **0.9594** | 48.06% / 1.93× | 0.9594 | 1.321 → 1.059 GiB |
| `zh_long_regression.txt`（回归分块，单次） | 2.0393 | **0.9285** | 54.47% / 2.20× | 0.9285 | 1.934 → 1.067 GiB |

#### 自动语音质量筛选（SenseVoice INT8 ASR + CAMPPlus，仅自动化筛查，未进行人工主观听感评测）

由于本优化使用了 4-bit 权重量化（非逐字节一致），使用 `benchmarks/assess_ming_audio.py` 对上述 ABBA 输出进行了自动化筛查（结果见 `benchmarks/results/ming-fast-quality-20261126T145100Z/quality.json`）：

| 配置与文本 | 原版 ASR CER | 优化版 ASR CER | 优化版削波比例 | 跨版 CAMPPlus 余弦（无参考音频随机采样） |
| --- | ---: | ---: | ---: | ---: |
| `q4g64 s=8` / `zh_short.txt` | 0.0000 | **0.0000** | **0.000000** | 0.7187 |
| `q4g64 s=8` / `zh_paragraph.txt` | 0.0625 | **0.0208** | **0.000000** | 0.5089 |
| `q4g64 s=8` / `zh_regression.txt` | 0.0000 | **0.0000** | **0.000000** | — |
| `q4g64 s=8` / `zh_long_regression.txt` | 0.0000 | **0.0000** | 0.000055 | — |
| `q4g128 s=10` / `zh_short.txt` | 0.0000 | **0.0000** | **0.000000** | 0.7371 |
| `q4g128 s=10` / `zh_paragraph.txt` | 0.0625 | **0.0312** | 0.000006 | 0.6187 |
| `q4g128 s=10` / `zh_regression.txt` | 0.0000 | **0.0000** | **0.000000** | — |
| `q4g128 s=10` / `zh_long_regression.txt` | 0.0000 | **0.0000** | 0.000009 | — |

说明：无参考音频 TTS（无 `voice_ref`）会在首块随机采样音色，因此量化后与原版 Q8_0 之间的跨版说话人余弦低于 `0.90` 触发了 `review_flag`；四组文本的 SenseVoice ASR CER 均持平或低于原版。完整原始记录见：

- `q4g64 s=8` 短句与回归 ABBA：`benchmarks/results/ming-fast-comparison-q4g64-ar_q4_0-vae_q4_0-s8-c1-t4-20261126T143150Z/comparison.json`
- `q4g64 s=8` 长段落 ABBA：`benchmarks/results/ming-fast-comparison-q4g64-ar_q4_0-vae_q4_0-s8-c1-t4-20261126T143453Z/comparison.json`
- `q4g128 s=10` 短句与回归 ABBA：`benchmarks/results/ming-fast-comparison-q4g128-ar_q4_0-vae_q4_0-s10-c1-t4-20261126T144115Z/comparison.json`
- `q4g128 s=10` 长段落 ABBA：`benchmarks/results/ming-fast-comparison-q4g128-ar_q4_0-vae_q4_0-s10-c1-t4-20261126T144426Z/comparison.json`
- 自动质量筛选汇总：`benchmarks/results/ming-fast-quality-20261126T145100Z/quality.json`
- 诊断分阶段计时（阶段未计入差值 `< 1 µs`）：`benchmarks/results/ming3-q8-profile-fast-t4-20261126T142634Z/summary.json`、`benchmarks/results/ming3-q8-profile-fast-t4-20261126T142558Z/summary.json`
- 已脱敏并纳入仓库追踪的 Plan B / Plan C 汇总 JSON：[`planB_q4g64_s8_short_abba.json`](../benchmarks/published_results/planB_q4g64_s8_short_abba.json)、[`planB_q4g64_s8_paragraph_abba.json`](../benchmarks/published_results/planB_q4g64_s8_paragraph_abba.json)、[`planC_q4g128_s10_short_abba.json`](../benchmarks/published_results/planC_q4g128_s10_short_abba.json)、[`planC_q4g128_s10_paragraph_abba.json`](../benchmarks/published_results/planC_q4g128_s10_paragraph_abba.json)、[`planBC_quality_screening.json`](../benchmarks/published_results/planBC_quality_screening.json)

## 听感排查与方案 A 确立（`AR=q8 + fused_fp32_act=1`，2026-11-27）

在与 Mac Metal 生成结果及原版 Q8_0 的直接人工试听对比中发现：尽管 Plan B（`ar=q4_0`）的 SenseVoice ASR CER 仅为 `0.0208`，但在 107 字自动分块长文本的第 2 块起出现了轻微的「金属音 / 电话压缩感」。

经单变量消融（逐一恢复 `reuse_anchor_latent=0`、`VAE=q8`、`Aggregator=q8`、`AR=q8`），最终精确定位根因是 **24 层自回归主干（AR）的 4-bit 量化（`q4_0`）** 在长上下文条件预测中引入了声学隐变量漂移；将 AR 保持原版 `Q8_0` 精度并开启 `--fused-fp32-act 1` 后，金属音与电话音彻底消失，而整体仍保持实时（`RTF < 1.0`）。该组合确立为 **Plan A（默认推荐音质优先实时方案）**。

### Plan A 正式 ABBA 验证（4 线程 / 400% CPUQuota / 6 GiB MemoryMax / 0 swap）

每组按 ABBA 顺序与未修改原版 `ming3_rtf` 对照，各 4 次正式测量，全部确定性 SHA-256 一致、零 swap、SenseVoice ASR CER 全部为 `0.0000`：

| 文本与音色 | 原版 RTF 中位数 | Plan A RTF 中位数 | 加速比 | Plan A 各次 RTF | 峰值 RSS | Plan A ASR CER | 跨版 CAMPPlus 余弦 | 脱敏汇总 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `zh_short.txt`（无参考短句，4.0s） | 1.6940 | **0.8717** | **1.94×** | 0.9018, 0.8599, 0.8787, 0.8647 | 1.244 GiB | 0.0000 | 0.8579 | [`planA_ar_q8_short_abba.json`](../benchmarks/published_results/planA_ar_q8_short_abba.json) |
| `zh_paragraph.txt`（无参考 107 字，20.9s） | 1.9088 | **0.9327** | **2.05×** | 0.9292, 0.9333, 0.9365, 0.9321 | 1.255 GiB | 0.0000 | 0.8785 | [`planA_ar_q8_paragraph_abba.json`](../benchmarks/published_results/planA_ar_q8_paragraph_abba.json) |
| `zh_lingxiaorou_short.txt`（内置 IP「灵小柔」30 字，6.0s） | 1.8066 | **0.9525** | **1.90×** | 0.9531, 0.9488, 0.9716, 0.9519 | 1.244 GiB | 0.0000 | **0.9539**（`review_flag=false`） | [`planA_ar_q8_lingxiaorou_short_abba.json`](../benchmarks/published_results/planA_ar_q8_lingxiaorou_short_abba.json) |
| `zh_paragraph.txt`（内置 IP「灵小柔」107 字 `freeform`，18.5s） | 2.0229 | **1.0766** | **1.88×** | 1.0307, 1.0146, 1.1226, 1.1655 | 1.258 GiB | 0.0000 | **0.9166**（`review_flag=false`） | [`planA_ar_q8_lingxiaorou_paragraph_abba.json`](../benchmarks/published_results/planA_ar_q8_lingxiaorou_paragraph_abba.json) |

补充回归（单次）：`zh_regression.txt` `1.6888 → 0.8861`（1.91×），`zh_long_regression.txt` `1.8672 → 0.9004`（2.07×）。自动质量筛选完整报告见 [`planA_quality_screening.json`](../benchmarks/published_results/planA_quality_screening.json)。

说明：在固定基线 `7d20b414` 中，`model_specs/ming3_tts.json` 尚未在 `voice_id` 枚举中列出中文 IP 名称，因此通过 `template_name=freeform` 传入 `{"IP": "灵小柔"}` 时，`session.cpp` 未设置 `controls["ip"]`，导致 107 字长段落仍在第 1 块后触发 `anchor_voice` 并在后续分块重复预填充 65 个参考 latent 与约 55 个 JSON 控制 token，使 107 字 `灵小柔` RTF 为 `1.0766`（较无参考 107 字的 `0.9327` 略高，但仍较原版提速 `1.88×` 且音色余弦达 `0.9166`）。
