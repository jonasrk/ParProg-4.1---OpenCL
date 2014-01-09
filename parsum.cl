//taken from http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854?pgno=3 and modified

__kernel void parsum(__global ulong4* data,
                     __local unsigned long * local_result, __global unsigned long * group_result) {
    
    //    int my_index = get_global_id(0);
    //    group_result[get_group_id(0)] = my_index;
    
    unsigned long  sum;
    ulong4 input1, input2, sum_vector;
    uint global_addr, local_addr;
    
    global_addr = get_global_id(0) * 2;
    input1 = data[global_addr];
    input2 = data[global_addr+1];
    sum_vector = input1 + input2;
    
    local_addr = get_local_id(0);
    local_result[local_addr] = sum_vector.s0 + sum_vector.s1 +
    sum_vector.s2 + sum_vector.s3;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(get_local_id(0) == 0) {
        sum = 0.0f;
        for(int i=0; i<get_local_size(0); i++) {
            sum += local_result[i];
        }
        group_result[get_group_id(0)] = sum;
    }
}
