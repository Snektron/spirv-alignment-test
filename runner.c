#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#define CHECK_CL(x) \
    { \
        cl_int _status = (x); \
        if (_status != CL_SUCCESS) { \
            fprintf(stderr, __FILE__ ":%d: error: opencl returned %d\n", __LINE__, _status); \
            exit(EXIT_FAILURE); \
        } \
    }

char* platform_get_name(cl_platform_id platform) {
    size_t size;
    CHECK_CL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &size));
    char* name = malloc(size);
    CHECK_CL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, name, NULL));
    return name;
}

char* device_get_name(cl_device_id device) {
    size_t size;
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &size));
    char* name = malloc(size);
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, size, name, NULL));
    return name;
}

bool device_supports_spirv(cl_device_id device) {
    size_t il_bytes;
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_ILS_WITH_VERSION, 0, NULL, &il_bytes));
    cl_name_version* versions = malloc(il_bytes);
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_ILS_WITH_VERSION, il_bytes, versions, NULL));

    for (size_t i = 0; i < il_bytes / sizeof(cl_name_version); ++i) {
        if (strstr(versions[i].name, "SPIR-V") != 0) {
            return true;
        }
    }

    return false;
}

int main(int argc, char* argv[]) {
    char* module_path = NULL;
    char* kernel_name = NULL;
    char* platform_query = NULL;
    char* device_query = NULL;

    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        if (strcmp(arg, "--platform") == 0) {
            if (++i == argc) {
                fprintf(stderr, "--platform expects argument <platform>\n");
                return EXIT_FAILURE;
            }
            platform_query = argv[i];
        } else if (strcmp(arg, "--device") == 0) {
            if (++i == argc) {
                fprintf(stderr, "--device expects argument <device>\n");
                return EXIT_FAILURE;
            }
            device_query = argv[i];
        } else if (!module_path) {
            module_path = arg;
        } else if (!kernel_name) {
            kernel_name = arg;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg);
            return EXIT_FAILURE;
        }
    }

    if (!module_path || !kernel_name) {
        printf("usage: %s [--platform <platform>] [--device <device>] <module.spv> <kernel name>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cl_uint num_platforms;
    CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms));
    cl_platform_id* platforms = malloc(num_platforms * sizeof(cl_platform_id));
    CHECK_CL(clGetPlatformIDs(num_platforms, platforms, NULL));

    cl_platform_id platform;
    cl_device_id device;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char* platform_name = platform_get_name(platforms[i]);
        if (platform_query && strstr(platform_name, platform_query) == 0) {
            free(platform_name);
            continue;
        }

        cl_uint num_devices;
        cl_int status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (status == CL_DEVICE_NOT_FOUND) {
            free(platform_name);
            continue;
        }
        CHECK_CL(status);
        cl_device_id* devices = malloc(num_devices * sizeof(cl_device_id));
        CHECK_CL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

        for (cl_uint j = 0; j < num_devices; ++j) {
            char* device_name = device_get_name(devices[j]);
            if (device_query && strstr(device_name, device_query) == 0) {
                free(device_name);
                continue;
            }
            if (device_supports_spirv(devices[j])) {
                platform = platforms[i];
                device = devices[j];

                printf("selected platform '%s' and device '%s'\n", platform_name, device_name);

                free(platforms);
                free(devices);
                free(device_name);
                goto device_selected;
            }

            free(device_name);
        }

        free(devices);
        free(platform_name);
    }
    free(platforms);

    fprintf(stderr, "failed to select device\n");
    return EXIT_FAILURE;

device_selected:
    int fd = open(module_path, O_RDONLY);
    if (!fd) {
        fprintf(stderr, "failed to load module\n");
        return EXIT_FAILURE;
    }
    off_t size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    uint32_t* module = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (module == MAP_FAILED) {
        fprintf(stderr, "failed to load module\n");
    }

    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
    cl_int status;
    cl_context context = clCreateContext(props, 1, &device, NULL, NULL, &status);
    CHECK_CL(status);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
    CHECK_CL(status);

    cl_program program = clCreateProgramWithIL(
        context,
        module,
        size,
        &status
    );
    CHECK_CL(status);

    CHECK_CL(clBuildProgram(program, 1, &device, NULL, NULL, NULL));

    cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
    CHECK_CL(status);

    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * 2, NULL, &status);
    CHECK_CL(status);

    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf));

    cl_event kernel_done;
    size_t global_wrk = 1;
    size_t local_wrk = 1;
    CHECK_CL(clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        NULL,
        &global_wrk,
        &local_wrk,
        0,
        NULL,
        &kernel_done
    ));

    uint64_t result[2];
    CHECK_CL(clEnqueueReadBuffer(
        queue,
        buf,
        CL_TRUE,
        0,
        8 * 2,
        &result,
        1,
        &kernel_done,
        NULL
    ));

    printf("0x%lX 0x%lX\n", result[0], result[1]);

    clReleaseMemObject(buf);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    munmap(module, size);
}
