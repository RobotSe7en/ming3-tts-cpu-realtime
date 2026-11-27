BUILD_DIR ?= .cache/bench-build
JOBS ?= 2
THREADS ?= 4

.PHONY: help setup build test server install-systemd bench clean

help:
	@echo "Ming-3.0-TTS 0.5B Real-Time x86 CPU Engine & Streaming Service"
	@echo ""
	@echo "Targets:"
	@echo "  make setup            Fetch pinned audio.cpp, apply patch, download GGUF, build & test"
	@echo "  make build            Configure & build server, benchmark binaries and unit tests (JOBS=$(JOBS))"
	@echo "  make test             Run CTest parity suite (AVX2 microkernels + streaming KV-cache VAE)"
	@echo "  make server           Run local Plan A server + AYTS/1 streaming gateway in foreground"
	@echo "  make install-systemd  Install & start persistent systemd --user services"
	@echo "  make bench            Run Plan A short-sentence ABBA benchmark ($(THREADS) threads)"
	@echo "  make clean            Remove build directory ($(BUILD_DIR))"

setup:
	./scripts/setup.sh

build:
	cmake -S benchmarks -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --target ming3_fast_server ming3_rtf ming3_fast_rtf ming3_fast_profile_rtf ming3_cpu_linear_test ming3_fast_engine_test ming3_stream_vae_test -j $(JOBS)

test:
	OMP_NUM_THREADS=$(THREADS) OMP_DYNAMIC=FALSE MING_VAE_QUANT=q4_0 ctest --test-dir $(BUILD_DIR) --output-on-failure

server:
	./scripts/start_server.sh

install-systemd:
	./scripts/install_systemd.sh

bench:
	python3 benchmarks/compare_ming_fast.py --fixture zh_short.txt --iterations 2 --threads $(THREADS) \
		--reference baseline --flow-quant q4g128 --agg-quant q4g128 --ar-quant q8 \
		--vae-quant q4_0 --fused-fp32-act auto --flow-steps 8 --cfg-interval 1

clean:
	rm -rf $(BUILD_DIR)
