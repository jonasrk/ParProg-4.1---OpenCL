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

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

int main(int argc, char *argv[]) {
    
   /* Parsing commandline arguments */
    
    unsigned long start_index = atol(argv[1]);
    unsigned long end_index = atol(argv[2]);
    
    printf("start_index: %ld end_index: %ld\n\n", start_index, end_index);

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t local_size, global_size;

   /* Data and buffers */
    unsigned long  number_of_work_items = end_index/8 + (end_index % 8 == 0 ? 0 : 1);
    number_of_work_items += ( number_of_work_items % 4 == 0 ? 0 : 4 - number_of_work_items % 4 );
    
    printf("number_of_work_items: %ld \n\n", number_of_work_items);
    
   
   unsigned long   total, actual_sum;
   cl_mem input_buffer, sum_buffer;
   cl_int num_groups;

   /* Initialize data */
    
    global_size = number_of_work_items;
    global_size += (global_size % 8 == 0 ? 0 : 8 - (global_size % 8));
    
    unsigned long data[global_size*8];
    
    for(i=0; i<(global_size*8); i++) {
       data[i] = (i < end_index ? 1.0f*i : 0.0f);
   }

   /* Create device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create data buffer */
    
    printf("\nglobal_size: %lu\n", global_size);
   local_size = 4;
    printf("\nlocal_size: %lu\n", local_size);
   num_groups = global_size/local_size;
    unsigned long sum[num_groups];
    
    printf("\nnum_groups: %i\n", num_groups);
    
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                  CL_MEM_COPY_HOST_PTR, end_index * sizeof(unsigned long ), data, &err);
    sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                CL_MEM_COPY_HOST_PTR, num_groups * sizeof(unsigned long ), sum, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(kernel, 1, local_size * sizeof(int ), NULL);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &sum_buffer);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   /* Enqueue kernel */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                 &local_size, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0, 
         sizeof(sum), sum, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   /* Check result */
   total = 0;
   for(j=0; j<num_groups; j++) {
//       printf("\n sum of group %i is %f\n", j, sum[j]);
      total += sum[j];
   }
   actual_sum = 1 * end_index/2*(end_index-1);
   printf("\nComputed sum = %ld\n", total);
   printf("Checksum = %ld\n", actual_sum);
   if(total != actual_sum)
      printf("Check failed.\n");
   else
      printf("Check passed.\n");

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(sum_buffer);
   clReleaseMemObject(input_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
