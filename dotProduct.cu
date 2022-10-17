/*---------------------------------------------------------------------------
* UNIVERSIDAD DEL VALLE DE GUATEMALA
* FACULTAD DE INGENIERIA
*
* Nombres:
* Jose Pablo Kiesling Lange - 21581
* Pablo Andres Zamora Vasquez - 21780
* Laboratorio No.06
------------------------------------------------------------------------------*/

#include <stdio.h>
#include <cuda_runtime.h>

/************************************************************************
 * Kernel ejecutado por Device
 * Opera cada elemento de los vectores A y B, guardandolo en la posición
 * correspondiente en C
 * @param A 
 * @param B 
 * @param C 
 * @param numElements 
 */
__global__ void
vectorMultiply(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements){
        C[i] = A[i] * B[i];
    }
}
//***********************************************************************


/************************************************************************
 * Rutina ejecutada por Host
 */
int main(void)
{
    //-------------------- 0. Variables de control--------------------
    cudaError_t err = cudaSuccess;
    int numElements = 768;
    size_t size = numElements * sizeof(float);
    float result = 0.0;
  
    //-------------------- 1.1 Reservar memoria en Host para vectores A B y C --------------------
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL){
        fprintf(stderr, "No se ha podido reservar memoria en Host para los vectores\n");
        exit(EXIT_FAILURE);
    }

    //-------------------- 1.2 Reservar memoria en Device para vectores A B y C --------------------
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido reservar memoria en Device para el vector A (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido reservar memoria en Device para el vector B (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido reservar memoria en Device para el vector C (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //-------------------- 2. Generar valores para operandos A B en Host --------------------
    for (int i = 0; i < numElements; ++i){
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    //-------------------- 3. Pasar valores de A & B de Host a Device --------------------
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido copiar el vector A de Host a Device (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido copiar el vector B de Host a Device (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //-------------------- 4. Ejecutar Kernel en DEVICE --------------------
    int threadsPerBlock = 768;
    int blocks = 1;
    vectorMultiply<<< blocks, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido ejecutar el Kernel vectorMultiply (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    //-------------------- 5.1 Copiar resultado de DEVICE a HOST --------------------
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido copiar el vector C de Device a Host (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //-------------------- 5.2 Sumar secuencialmente los elementos del vector C --------------------
    for (int i = 0; i < numElements; i++){
        result += h_C[i];
        printf("Producto de los elementos [%d] de A & B: %f\n", i, h_C[i]);
    }
    printf("\n----------------------------------------\n");
    printf("RESULTADO FINAL del producto punto: %f", result);

    
    //-------------------- 6. Liberar memoria del HOST --------------------
    free(h_A);
    free(h_B);
    free(h_C);
 
    //-------------------- 7. Liberar memoria Global --------------------
    err = cudaFree(d_A);
    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido liberar la memoria del vector A (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido liberar la memoria del vector B (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_C);
    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido liberar la memoria del vector C (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaDeviceReset();
    if (err != cudaSuccess){
        fprintf(stderr, "No se ha podido reiniciar el Device (codigo de error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    return 0;
}

