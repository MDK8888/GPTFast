
__device__ void int4_matmul_kernel(
    float* __restrict__ out,
    float* __restrict__ activation,
    float* __restrict__ weight,
    float* __restrict__ scales,
    float* __restrict__ zeros,
    int out_width,
    int out_height,
    int weight_width,
    int weight_height,
    int groupsize,
    int qmin,
    int qmax
){
    int matmul_dim = weight_height;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= out_height || col >= out_width) {
        return;
    }

    int mid = (int)((qmax + qmin + 1)/2);

    int sz_index = (int)(col / groupsize);
    int sz_width = (int)(weight_width / groupsize);

    for(int i = 0; i < matmul_dim; i++){
        float activation_element = activation[row * matmul_dim + i];
        float weight_element = weight[i * weight_width + col];
        weight_element-= mid;
        
        float scale_element = scales[i * sz_width + sz_index];
        weight_element *= scale_element;
        float zero_element = zeros[i * sz_width + sz_index];
        weight_element += zero_element; 

        out[row * weight_width + col] += weight_element * activation_element;
    }
}