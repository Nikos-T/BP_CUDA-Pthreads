#include "header.h"
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

unsigned int threads_active = 0;
pthread_mutex_t threads_active_mutex;
pthread_attr_t attr;							//probly not needed here

typedef struct{
	float *alphas, *deltas, *biases, *weights, *weightGs, *output, *input;
	unsigned int *layer_sizes;
	unsigned int n_layers;
}thread_data;

void * back_propagation_thread(void *arg) {
	thread_data *data = (thread_data *)arg;
	float *weight_pointers[data->n_layers-1], *bias_pointers[data->n_layers-1], *weightG_pointers[data->n_layers-1], *alpha_pointers[data->n_layers-1], *delta_pointers[data->n_layers-1];

	weight_pointers[0] = data->weights;
	bias_pointers[0] = data->biases;
	weightG_pointers[0] = data->weightGs;
	alpha_pointers[0] = data->alphas;
	delta_pointers[0] = data->deltas;
	for (unsigned int i=1; i<data->n_layers-1; i++) {
		weight_pointers[i] = weight_pointers[i-1] + data->layer_sizes[i]*data->layer_sizes[i-1];
		bias_pointers[i] = bias_pointers[i-1] + data->layer_sizes[i];
		weightG_pointers[i] = weightG_pointers[i-1] + data->layer_sizes[i]*data->layer_sizes[i-1];
		alpha_pointers[i] = alpha_pointers[i-1] + data->layer_sizes[i];
		delta_pointers[i] = delta_pointers[i-1] + data->layer_sizes[i];
	}

	forward_step_wrap(weight_pointers[0], bias_pointers[0], data->input, alpha_pointers[0], data->layer_sizes[0], data->layer_sizes[1]);
	for (unsigned int i=1; i<data->n_layers-1; i++) {
		forward_step_wrap(weight_pointers[i], bias_pointers[i], alpha_pointers[i-1], alpha_pointers[i], data->layer_sizes[i], data->layer_sizes[i+1]);
	}

	//output error dL
	output_error_wrap(alpha_pointers[data->n_layers-2], data->output, delta_pointers[data->n_layers-2], data->layer_sizes[data->n_layers-1]);

	//backward pass
	for (unsigned int i=data->n_layers-2; i>0; i--) {
		backward_step_wrap(weight_pointers[i], alpha_pointers[i-1], delta_pointers[i], delta_pointers[i-1], data->layer_sizes[i], data->layer_sizes[i+1]);
	}

	//calculate partial weight derivatives
	weight_gradient_wrap(data->input, delta_pointers[0], weightG_pointers[0], data->layer_sizes[0], data->layer_sizes[1]);
	for (unsigned int i=1; i<data->n_layers-1; i++) {
		weight_gradient_wrap(alpha_pointers[i-1], delta_pointers[i], weightG_pointers[i], data->layer_sizes[i], data->layer_sizes[i+1]);
	}

	pthread_mutex_lock(&threads_active_mutex);
	//computed++;
	threads_active--;
	pthread_mutex_unlock(&threads_active_mutex);

	pthread_exit(0);
}

int read_io(char *input_file_name, char *output_file_name, float *inputs, float *outputs, unsigned int input_size, unsigned int output_size, unsigned int samples) {
	FILE *fp;

	fp = fopen(input_file_name, "rb");
	if (fp == NULL) {
		printf("Error opening file %s.\n", input_file_name);
		return -1;
	}
	if (fread(inputs, sizeof(float), samples*input_size, fp) != samples*input_size) {
		printf("Error reading %s.\n", input_file_name);
		perror("Error");
		return -2;
	}
	fclose(fp);

	fp = fopen(output_file_name, "rb");
	if (fp == NULL) {
		printf("Error opening file %s.\nExiting...\n", output_file_name);
		return -1;
	}
	if (fread(outputs, sizeof(float), samples*output_size, fp) != samples*output_size) {
		printf("Error reading %s.\n", output_file_name);
		return -2;
	}
	fclose(fp);

	return 0;
}

int initialize_network(unsigned int n_layers, unsigned int *layer_sizes, float *weights, float *biases) {
	unsigned int weight_index=0, biases_index=0;
	for (unsigned int i=0; i<n_layers-1; i++) {
		for (unsigned int j=0; j<layer_sizes[i]*layer_sizes[i+1]; j++) {
			weights[weight_index + j] = 0;
		}
		for(unsigned int j=0; j<layer_sizes[i+1]; j++) {
			biases[biases_index + j] = 0;
		}
		weight_index += layer_sizes[i]*layer_sizes[i+1];
		biases_index += layer_sizes[i+1];
	}
	return 0;
}

int main(int argc, char **argv) {

unsigned int n_layers = 3, layer_sizes[3] = {784, 15, 10}, input_size, output_size, samples = 20000, weights_size=0, biases_size=0;
float *weights, *biases, *inputs, *outputs, *weightGs;
input_size = layer_sizes[0];
output_size = layer_sizes[n_layers-1];

unsigned int wait_var = atoi(argv[2]);
unsigned int nthreads = atoi(argv[1]);

for (unsigned int i=0; i<n_layers-1; i++) {
	weights_size += layer_sizes[i]*layer_sizes[i+1];
	biases_size += layer_sizes[i+1];
}
weights = (float *)malloc(weights_size*sizeof(float));
biases = (float *)malloc(biases_size*sizeof(float));
if (weights == NULL) {
	printf("Failed to allocate weights.\nExiting...\n");
	return -1;
}
if (biases == NULL) {
	printf("Failed to allocate biases.\nExiting...\n");
	return -1;
}
inputs = (float *)malloc(input_size*samples*sizeof(float));
outputs = (float *)malloc(output_size*samples*sizeof(float));		//maybe also put inp/outp pointers??
if (inputs == NULL) {
	printf("Failed to allocate inputs.\nExiting...\n");
	return -1;
}
if (outputs == NULL) {
	printf("Failed to allocate outputs.\nExiting...\n");
	return -1;
}

if (read_io("/home/nterzopo/Parallel_4/Training_Data/inputs60000x784.mydata", "/home/nterzopo/Parallel_4/Training_Data/outputs60000x10.mydata", inputs, outputs, input_size, output_size, samples) != 0) {
	printf("Function read_io() failed.\nExiting...\n");
	return -2;
}

if (initialize_network(n_layers, layer_sizes, weights, biases) != 0) {
	printf("Function initialize_network() failed.\nExiting...\n");
	return -3;
}


weightGs = (float *)malloc(samples*weights_size*sizeof(float));
if (weightGs == NULL) {
	printf("Failed to allocate weightGs.\nExiting...\n");
	return -1;
}

//one epoch in the nn first sample
float *alphas, *deltas;
unsigned int ad_size=0;
for (unsigned int i=1; i<n_layers; i++) {
	ad_size += layer_sizes[i];
}
alphas = (float *)malloc(samples*ad_size*sizeof(float));
deltas = (float *)malloc(samples*ad_size*sizeof(float));
if (alphas == NULL) {
	printf("Failed to allocate alphas.\nExiting...\n");
	return -1;
}
if (deltas == NULL) {
	printf("Failed to allocate deltas.\nExiting...\n");
	return -1;
}

thread_data samples_data[samples];
pthread_t threads[samples];
for (unsigned int i=0; i<samples; i++) {
	samples_data[i].alphas = alphas+i*ad_size;
	samples_data[i].deltas = deltas+i*ad_size;
	samples_data[i].biases = biases;
	samples_data[i].weights = weights;
	samples_data[i].weightGs = weightGs+i*weights_size;
	samples_data[i].output = outputs+i*output_size;
	samples_data[i].input = inputs+i*input_size;
	samples_data[i].layer_sizes = layer_sizes;
	samples_data[i].n_layers = n_layers;
}
printf("samples_data created.\n");

/*working but too many threads cause segmentation fault
for (unsigned int i=0; i<100; i++) {
	pthread_create(&threads[i], NULL, back_propagation_thread, &samples_data[i]);

	printf("Thread %u created.\n", i);
}

for (unsigned int i=0; i<100; i++) {
	pthread_join(threads[i], NULL);
	printf("Thread %u joined.\n", i);
}*/
if (pthread_mutex_init(&threads_active_mutex, NULL)!=0) {
	printf("Failed to initialize mutex.\nExiting...");
	return -10;
}
if (pthread_attr_init(&attr)!=0) {
	printf("Failed to initialize pthread attribute.\nExiting...");
	return -10;
}
if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED)!=0) {
	printf("Failed to set detached.\nExiting...");
	return -10;
}
struct timeval t1, t2;
double elapsedTime;
unsigned int created=0;
printf("nthreads  = %u, wait_var = %u.\n", nthreads, wait_var);
printf("Before while.\n");
gettimeofday(&t1, NULL);
while (created<samples) {
	if (threads_active<nthreads) {
		if (pthread_create(&threads[created], &attr, back_propagation_thread, &samples_data[created])!=0) {
			printf("Failed to create thread.\nExiting...");
			return -10;
		}
		//printf("Created thread %u.\n", created);
		created++;
		pthread_mutex_lock(&threads_active_mutex);
		threads_active++;
		pthread_mutex_unlock(&threads_active_mutex);
		continue;
	}
	usleep(wait_var);
}
while (threads_active>0) {
	//printf("Waiting for all threads to finish.\n");
	usleep(1000000);
}
gettimeofday(&t2, NULL);
elapsedTime = (t2.tv_sec - t1.tv_sec)*1000;
elapsedTime += (t2.tv_usec - t1.tv_usec)/1000;
FILE *fp;
fp = fopen("test.txt", "a");
if (fp==NULL)
	printf("could not open file.\n");
fprintf(fp, "threads = %u, wait = %u, time = %.1f\n", nthreads, wait_var, elapsedTime);
fclose(fp);
printf("Finished. Took %.1fms\n", elapsedTime);
printf("Computed %u samples: ", samples);
printf("\n");
//Here all weight gradients have been computed by gpu
//if (sum_weights_wrap_test(weightGs, weights, ))



free(alphas);
free(weights);
free(weightGs);
free(deltas);
free(biases);
pthread_mutex_destroy(&threads_active_mutex);
pthread_attr_destroy(&attr);

/*
// check if weight and bias pointers work correctly
for (unsigned int i=0; i<n_layers-1; i++) {
	printf("weight%u%u:\n", i+1, i);
	for (unsigned int j=0; j<layer_sizes[i+1]; j++) {
		for (unsigned int k=0; k<layer_sizes[i]; k++) {
			printf("%.1f, ", weight_pointers[i][j*layer_sizes[i]+k]);
		}
		printf("\n");
	}
	printf("bias%u:\n", i+1);
	for (unsigned int j=0; j<layer_sizes[i+1]; j++) {
		printf("%.1f, ", bias_pointers[i][j]);
	}
	printf("\n");
	getchar();
}
//check weight gradient
printf("weightG21:\n");
for (unsigned int i=0; i<layer_sizes[2]; i++) {
	for (unsigned int j=0; j<layer_sizes[1]; j++) {
		printf("%.4f, ", weightG_pointers[1][i*layer_sizes[1]+j]);
	}
	printf("\n");
}
printf("\n");



//test read_io
	unsigned int samples = 60000, input_size = 784, output_size = 10;
	float *inputs, *outputs;
	inputs = (float *)malloc(samples*input_size*sizeof(float));
	outputs = (float *)malloc(samples*output_size*sizeof(float));
	if (inputs == NULL) {
		printf("Could not allocate memory to inputs.\nExiting...\n");
		return -1;
	}
	if (outputs == NULL) {
		printf("Could not allocate memory to outputs.\nExiting...\n");
		return -1;
	}

	if (read_io("/home/nterzopo/Parallel_4/BP-CUDA/data/inputs60000x784.mydata", "../data/outputs60000x10.mydata", inputs, outputs, input_size, output_size, samples) != 0) {
		printf("Error in read_io().\nExiting...\n");
		return 0;
	}
	getchar();
	printf("input[59998]:\n");
	for (unsigned int i=0; i<28; i++) {
		for (unsigned int j=0; j<28; j++) {
			if (inputs[784*59998+28*i+j]>0) printf("x ");
			else printf("  ");
		}
		printf("\n");
	}
	getchar();
	printf("output[59999]:\n");
	for (unsigned int i=59998; i<59999; i++) {
		for (unsigned int j=0; j<10; j++) {
			printf("%.1f, ", outputs[10*i+j]);
		}
		printf("\n");
	}
	printf("\n");

	free(inputs);
	free(outputs);

float *weightGs, *result;
unsigned int size = 20, samples = 10000;
FILE *fp;
weightGs = (float *)malloc(samples*size*sizeof(float));
result = (float *)malloc((samples/1024+1)*size*sizeof(float));
if (weightGs == NULL) {
	printf("Could not allocate memory to weightGs.\nExiting...\n");
	return -1;
}
if (result == NULL) {
	printf("Could not allocate memory to result.\nExiting...\n");
	return -1;
}
fp = fopen("../data/sumWeights.mydata", "rb");
if (fp == NULL) {
	printf("Error opening file sumWeights.mydata.\nExiting...\n");
	return -1;
}
if (fread(weightGs, sizeof(float), size*samples, fp) != size*samples) {
	printf("Error reading sumWeights.mydata.\nExiting...\n");
	return -1;
}
fclose(fp);
printf("\nweightGs1\n");
for (unsigned int i=0; i<20; i++) {
	printf("%.4f, ", weightGs[i]);
}
printf("\n");

getchar();
sum_weights_wrap_test(weightGs, result, 4, 5, samples);
printf("\nresult_row1\n");
for (unsigned int i=0; i<20; i++) {
	printf("%.4f, ", result[i]/10);
}
printf("\n");

unsigned int L, *layer_sizes;
FILE *fp;
//time_t start, end;
float **weights, **biases, **a, **delta, *x, *y;

{	// Parse arguments
if (argc < 2) {
	printf("Usage:\n./name_of_program L l1 l2 l3 ... lL\nOR\n./name_of_program L\nWhere\nL is the length of the neural network\nand\nl1 l2 l3 ... lL the size of the corresponding layer.\nIf no layer sizes are passed then the program will attempt to read them from \"../data/layer_sizes.mydata\"\nExiting...\n");
	return 1;
}
L = atoi(argv[1]);
if ((argc != L+2) && (argc != 2)) {
	printf("Usage:\n./name_of_program L l1 l2 l3 ... lL\nOR\n./name_of_program L\nWhere\nL is the length of the neural network\nand\nl1 l2 l3 ... lL the size of the corresponding layer.\nIf no layer sizes are passed then the program will attempt to read them from \"../data/layer_sizes.mydata\"\nExiting...\n");
	return 1;
}
layer_sizes = (unsigned int *)malloc(L*sizeof(unsigned int));
if (layer_sizes==NULL) {
	printf("Could not allocate memory to layer_sizes.\nExiting...\n");
	return -1;
}
if (argc == L+2) {
	for (unsigned int i=2; i<argc; i++) {
		layer_sizes[i-2] = atoi(argv[i]);
	}
} else {
	fp = fopen("../data/layer_sizes.mydata", "rb");
	if (fp == NULL) {
		printf("Error opening file layer_sizes.mydata.\nExiting...\n");
		return -1;
	}
	if (fread(layer_sizes, sizeof(unsigned int), L, fp) != L) {
		printf("Error reading layer_sizes.mydata. Check if number of layers is correct.\nExiting...\n");
		return -1;
	}
	fclose(fp);
}
}

{	// Mallocs
weights = (float **)malloc((L-1)*sizeof(float *));
biases = (float **)malloc((L-1)*sizeof(float *));
a = (float **)malloc((L-1)*sizeof(float *));
delta = (float **)malloc((L-1)*sizeof(float *));
x = (float *)malloc(layer_sizes[0]*sizeof(float));
y = (float *)malloc(layer_sizes[L-1]*sizeof(float));
if (weights == NULL) {
	printf("Could not allocate memory to weights.\nExiting...\n");
	return -1;
}
if (biases == NULL) {
	printf("Could not allocate memory to biases.\nExiting...\n");
	return -1;
}
if (a == NULL) {
	printf("Could not allocate memory to a.\nExiting...\n");
	return -1;
}
if (delta == NULL) {
	printf("Could not allocate memory to delta.\nExiting...\n");
	return -1;
}
if (x == NULL) {
	printf("Could not allocate memory to x.\nExiting...\n");
	return -1;
}
if (y == NULL) {
	printf("Could not allocate memory to y.\nExiting...\n");
	return -1;
}
for (unsigned int i=0; i<L-1; i++) {
	weights[i] = (float *)malloc(layer_sizes[i]*layer_sizes[i+1]*sizeof(float));
	biases[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	a[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	delta[i] = (float *)malloc(layer_sizes[i+1]*sizeof(float));
	if (weights[i] == NULL) {
		printf("Could not allocate memory to weights[%u].\nExiting...\n", i);
		return -1;
	}
	if (biases[i] == NULL) {
		printf("Could not allocate memory to biases[%u].\nExiting...\n", i);
		return -1;
	}
	if (a[i] == NULL) {
		printf("Could not allocate memory to a[%u].\nExiting...\n", i);
		return -1;
	}
	if (delta[i] == NULL) {
		printf("Could not allocate memory to delta[%u].\nExiting...\n", i);
		return -1;
	}
}
}

{	// Read files
fp = fopen("../data/input.mydata", "rb");
if (fp == NULL) {
	printf("Error opening file input.mydata.\nExiting...\n");
	return -1;
}
if (fread(x, sizeof(float), layer_sizes[0], fp) != layer_sizes[0]) {	// weigths[0] is guaranteed to be larger than input x
	printf("Error reading input.mydata.\nExiting...\n");
	return -1;
}
fclose(fp);

printf("\ninputT:\n");
for (unsigned int i=0; i<layer_sizes[0]; i++) {
	printf("%.4f, ", x[i]);
}
printf("\n");

fp = fopen("../data/output.mydata", "rb");
if (fp == NULL) {
	printf("Error opening file output.mydata.\nExiting...\n");
	return -1;
}
if (fread(y, sizeof(float), layer_sizes[L-1], fp) != layer_sizes[L-1]) {	// weigths[0] is guaranteed to be larger than input x
	printf("Error reading output.mydata.\nExiting...\n");
	return -1;
}
fclose(fp);
printf("\noutputT\n");
for (unsigned int i=0; i<layer_sizes[L-1]; i++) {
	printf("%.4f, ", y[i]);
}
printf("\n");
fp = fopen("../data/weights.mydata", "rb");
if (fp == NULL) {
	printf("Error opening file weights.mydata.\nExiting...\n");
	return -1;
}
for (unsigned int i=0; i<L-1; i++) {
	if (fread(weights[i], sizeof(float), layer_sizes[i]*layer_sizes[i+1], fp) != layer_sizes[i]*layer_sizes[i+1]) {
		printf("Error reading weights[%u]\nExiting...\n", i);
		return -1;
	}
	printf("\nweights%u%u\n",i+1, i);
	for (unsigned int j=0; j<layer_sizes[i+1]; j++) {
		for (unsigned int k=0; k<layer_sizes[i]; k++) {
			printf("%.4f, ", weights[i][j*layer_sizes[i]+k]);
		}
		printf("\n");
	}

}
fclose(fp);

fp = fopen("../data/biases.mydata", "rb");
if (fp == NULL) {
	printf("Error opening file biases.mydata.\nExiting...\n");
	return -1;
}
for (unsigned int i=0; i<L-1; i++) {
	if (fread(biases[i], sizeof(float), layer_sizes[i+1], fp) != layer_sizes[i+1]) {
		printf("Error reading biases[%u]\nExiting...\n", i);
		return -1;
	}
	printf("\nbiasesT%u\n", i);
	for (unsigned int j=0; j<layer_sizes[i+1]; j++) {
		printf("%.4f, ", biases[i][j]);
	}
	printf("\n");
}
}
getchar();
// test forward pass OK!
forward_step_wrap(weights[0], biases[0], x, a[0], layer_sizes[0], layer_sizes[1]);
for (unsigned int i=1; i<L-1; i++) {
	forward_step_wrap(weights[i], biases[i], a[i-1], a[i], layer_sizes[i], layer_sizes[i+1]);
}
for (unsigned int j=0; j<L-1; j++) {
printf("aT[%u]=\n", j);
for (unsigned int i=0; i<layer_sizes[j+1]; i++) {
	printf("%.4f, ", a[j][i]);
}
printf("\n");
}

// test output error OK!
output_error_wrap(a[L-2], y, delta[L-2], layer_sizes[L-1]);

printf("deltaT[%u]=\n", L-2);
for (unsigned int i=0; i<layer_sizes[L-1]; i++) {
	printf("%.4f, ", delta[L-2][i]);
}
printf("\n");

// test backward pass OK!
for (unsigned int i=L-2; i>0; i--) {
	backward_step_wrap(weights[i], a[i-1], delta[i], delta[i-1], layer_sizes[i], layer_sizes[i+1]);
}
for (unsigned int i=L-2; i>0; i--) {
	printf("deltaT[%u]=\n", i-1);
	for (unsigned int j=0; j<layer_sizes[i]; j++) {
		printf("%.4f, ", delta[i-1][j]);
	}
	printf("\n");
}
*/


return 0;
}