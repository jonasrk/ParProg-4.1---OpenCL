//taken from http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854?pgno=3 and modified

#define PROGRAM_FILE "parsum.cl"
#define KERNEL_FUNC "parsum"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
    
    cl_platform_id platform;
    cl_device_id dev;
    int err;
    
    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }
    
    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if(err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if(err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }
    
    return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
    
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;
    
    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    
    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);
    
    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        
        /* Find size of log and // print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        // printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    
    return program;
}

void print_device_info(cl_device_id dev){
    
    // taken from http://dhruba.name/2012/08/14/opencl-cookbook-listing-all-devices-and-their-critical-attributes/
    
    int j = 0;
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;
    
    // print device name
    clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("%d. Device: %s\n", j+1, value);
    free(value);
    
    // print hardware device version
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, valueSize, value, NULL);
    printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
    free(value);
    
    // print software driver version
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, valueSize, value, NULL);
    printf(" %d.%d Software version: %s\n", j+1, 2, value);
    free(value);
    
    // print c version supported by compiler for device
    clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
    free(value);
    
    // print parallel compute units
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

    
}

int main(int argc, char *argv[]) {
    
    /* Parsing commandline arguments */
    
//    unsigned long start_index = atol(argv[1]);
    unsigned long end_index = atol(argv[2]);
    unsigned long final_sum = 0;
    unsigned long amount_of_numbers_to_add_including_zero = end_index + 1;
    unsigned long actual_sum;
    int components_of_vector_type = 4;
    int components_per_work_item = components_of_vector_type * 2;
    int work_group_size = 4;
    int cycle_amount = 4062 * work_group_size * components_per_work_item;
    unsigned long  number_of_work_items_per_cycle = cycle_amount/components_per_work_item;
    
    
    
    // printf("start_index: %ld end_index: %ld\n\n", start_index, end_index);
    
    for (int cycle = 0; cycle <= amount_of_numbers_to_add_including_zero / cycle_amount; cycle ++) {
        
        /* OpenCL structures */
        cl_device_id device;
        cl_context context;
        cl_program program;
        cl_kernel kernel;
        cl_command_queue queue;
        cl_int i, j, err;
        size_t local_size, global_size;
        
        /* Data and buffers */
       
        
        // printf("number_of_work_items_per_cycle: %ld \n\n", number_of_work_items_per_cycle);
        
        
        unsigned long   total;
        cl_mem input_buffer, sum_buffer;
        cl_int num_groups;
        
        // printf("\nDebug A\n");
        
        /* Initialize data */
        
        global_size = number_of_work_items_per_cycle;
        
        // printf("\nDebug B\n");
        
        unsigned long data[cycle_amount*components_per_work_item];
        
        // printf("\nDebug B.1\n");
        
        for(i=0; i<cycle_amount*components_per_work_item; i++) {
            data[i] = (i + cycle*cycle_amount < amount_of_numbers_to_add_including_zero ? i + cycle*cycle_amount : 0.0f);
        }
        
        
        // printf("\nDebug C\n");
        /* Create device and context */
        device = create_device();
        
        print_device_info(device);
        
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if(err < 0) {
            perror("Couldn't create a context");
            exit(1);
        }
        
        
        
        /* Build program */
        program = build_program(context, device, PROGRAM_FILE);
        
        /* Create data buffer */
        
        // printf("\nglobal_size: %lu\n", global_size);
        local_size = work_group_size;
        // printf("\nlocal_size: %lu\n", local_size);
        num_groups = global_size/local_size;
        unsigned long sum[num_groups];
        
        // printf("\nnum_groups: %i\n", num_groups);
        
        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                      CL_MEM_COPY_HOST_PTR, cycle_amount * sizeof(unsigned long ), data, &err);
        sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                    CL_MEM_COPY_HOST_PTR, num_groups * sizeof(unsigned long ), sum, &err);
        if(err < 0) {
            perror("Couldn't create a buffer");
            exit(1);
        };
        
        // printf("\nDebug 1\n");
        
        /* Create a command queue */
        queue = clCreateCommandQueue(context, device, 0, &err);
        // printf("\nDebug 1.1\n");
        if(err < 0) {
            perror("Couldn't create a command queue");
            exit(1);
        };
        // printf("\nDebug 2\n");
        /* Create a kernel */
        kernel = clCreateKernel(program, KERNEL_FUNC, &err);
        if(err < 0) {
            perror("Couldn't create a kernel");
            exit(1);
        };
        // printf("\nDebug 3\n");
        /* Create kernel arguments */
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
        err |= clSetKernelArg(kernel, 1, local_size * sizeof(int ), NULL);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &sum_buffer);
        if(err < 0) {
            perror("Couldn't create a kernel argument");
            exit(1);
        }
        // printf("\nDebug 4\n");
        /* Enqueue kernel */
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL);
        if(err < 0) {
            perror("Couldn't enqueue the kernel");
            exit(1);
        }
        // printf("\nDebug 5\n");
        /* Read the kernel's output */
        err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0,
                                  sizeof(sum), sum, 0, NULL, NULL);
        if(err < 0) {
            perror("Couldn't read the buffer");
            exit(1);
        }
        // printf("\nDebug 6\n");
        /* Check result */
        total = 0;
        for(j=0; j<num_groups; j++) {
            //       // printf("\n sum of group %i is %f\n", j, sum[j]);
            total += sum[j];
        }
        
        
        final_sum += total;
        
        /* Deallocate resources */
        clReleaseKernel(kernel);
        clReleaseMemObject(sum_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseCommandQueue(queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        
    }
    
    actual_sum = (end_index)/2*(end_index+1) + (end_index % 2 == 0 ? 0 : end_index/2 + 1);
    // printf("\nComputed sum = %ld\n", final_sum);
    // printf("Checksum = %ld\n", actual_sum);
    if(final_sum != actual_sum)
          printf("Check failed.\n");
    else
          printf("Check passed.\n");
    
    
    FILE * Output;
	Output = fopen("output.txt", "a");
	fprintf(Output, "%ld", final_sum);
	fclose(Output);
    
    
    return 0;
}
