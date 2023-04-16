# SPIR-V alignment

Test program to see what alignment SPIR-V implementations align a variable at. The test is equivalent to the following kernel:

```cpp
using A = uint8_t[8];

struct B {
    uint32_t a, b;
};

__local uint8_t a;
__local A b;

void test(__local uint8_t* global* out) {
    *(__local B*)b = {1, 8};

    out[0] = &a;
    out[1] = (__local uint8_t*) &b;
}
```

# Usage

Run `make` to run the test, which reports the shared memory address of `a` and `b` (in that order).

Additional arguments can be passed to the runner using `make RUNNER_ARGS="<args">`. It accepts the arguments `--device <opencl device>` and `--platform <opencl platform>`.

## Results

### Rusticl
Rusticl aligns `a` and `b` to 1 byte:
```
./runner --platform rusticl test.spv test
selected platform 'rusticl' and device 'llvmpipe (LLVM 15.0.7, 256 bits)'
0x9 0xA
```

### Intel CPU OpenCL
Intel's CPU runtime (using SPIRV-LLVM-Translator) aligns `b` to 8 bytes:
```
./runner --platform Intel test.spv test
selected platform 'Intel(R) OpenCL' and device 'AMD Ryzen 7 3700X 8-Core Processor             '
0x7FEB885F7E00 0x7FEB885F7E08
```
This runtime honors alignment given by OpDecorate, until alignment of 256 bytes. The alignment is also influenced by the `OpStore` to the `(__local B*)b`. The alignment is not influenced by the `Aligned` attribute on the `OpStore`.
