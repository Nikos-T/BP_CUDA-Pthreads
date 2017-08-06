
int forward_step_wrap(float *weight, float *bias, float *a, float *a_new, unsigned int columns, unsigned int rows);

int output_error_wrap(float *aL, float *y, float *deltaL, unsigned int output_size);

int backward_step_wrap(float *weight, float *a, float *delta, float *delta_new, unsigned int columns, unsigned int rows);

int sum_weights_wrap_test(float *weightGs, float *result1, unsigned int columns, unsigned int rows, unsigned int samples);

int weight_gradient_wrap(float *a, float *delta, float *weightG, unsigned int columns, unsigned int rows);