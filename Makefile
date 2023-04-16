RUNNER_ARGS ?= --platform rusticl --device llvmpipe

all: runner test.spv
	./runner $(RUNNER_ARGS) test.spv test

runner: runner.c
	$(CC) -o runner runner.c -lOpenCL

test.spv: test.spvasm
	spirv-as $< -o $@ --target-env spv1.4
	spirv-val $@

.PHONY: all
