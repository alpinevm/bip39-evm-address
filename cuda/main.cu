extern "C" __global__ void square_num(const int *input, const int input_count, int *output)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_idx < input_count)
    {
        output[thread_idx] = input[thread_idx] * input[thread_idx];
    }
}