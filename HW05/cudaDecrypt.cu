#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "cuda.h"
#include "functions.c"

//compute a*b mod p safely
__device__ unsigned int modprodcu(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int za = a;
  unsigned int ab = 0;

  while (b > 0) {
    if (b%2 == 1) ab = (ab +  za) % p;
    za = (2 * za) % p;
    b /= 2;
  }
  return ab;
}

//compute a^b mod p safely
__device__ unsigned int modExpcu(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int z = a;
  unsigned int aExpb = 1;

  while (b > 0) {
    if (b%2 == 1) aExpb = modprodcu(aExpb, z, p);
    z = modprodcu(z, z, p);
    b /= 2;
  }
  return aExpb;
}
__global__ void search(unsigned int p, unsigned int g, unsigned int h, unsigned int* x){

    unsigned int myX = (unsigned int)(threadIdx.x+blockIdx.x*blockDim.x);
	unsigned int myY = (unsigned int)(threadIdx.y+blockIdx.y*blockDim.y);

	//find the secret key
	unsigned int i = myY*blockDim.x*gridDim.x+myX;
	if(i < p) {
    	if (modExpcu(g,i+1,p)==h)
       		*x=i+1;
	}
}
int main (int argc, char **argv) {

  /* Part 2. Start this program by first copying the contents of the main function from 
     your completed decrypt.c main function. */

  //declare storage for an ElGamal cryptosytem
  unsigned int n, p, g, h;
  unsigned int Nints;

  //get the secret key from the user
  //printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  //char stat = scanf("%u",&x);

  unsigned int* h_x = (unsigned int*)malloc(sizeof(unsigned int));
  *h_x = 0;

  //printf("Reading file.\n");

  FILE* f = fopen("bonus_public_key.txt", "r");
  fscanf(f, "%u\n%u\n%u\n%u\n", &n, &p, &g, &h);
  fclose(f);
  f = fopen("bonus_message.txt", "r");
  fscanf(f, "%u\n", &Nints);
  unsigned int* Zmessage = (unsigned int*) malloc(Nints*sizeof(unsigned int));
  unsigned int* a = (unsigned int*) malloc(Nints*sizeof(unsigned int));
  for(int i = 0; i < Nints; i++){
	  fscanf(f, "%u %u\n", &Zmessage[i], &a[i]);
  }
  fclose(f);
//---------------------------------------------------------------------------------------------------------------
    
  unsigned int* d_x;
  cudaMalloc(&d_x, sizeof(unsigned int));
  dim3 B(32, 32, 1);
  int N = (n-10+1)/2;
  if(N < 0)
	  N = 0;
  N = 1 << N;
  dim3 G(N,N,1);

  double startTime = clock();
  search <<< G,B >>> (p, g, h, d_x);
  cudaDeviceSynchronize();
  double endTime = clock();

  double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
  double work = (double) p;
  double throughput = work/totalTime;

  printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
  cudaMemcpy(h_x,d_x,sizeof(unsigned int),cudaMemcpyDeviceToHost);
  printf("x=%u\n", *h_x);
  cudaFree(d_x);
//--------------------------------------------------------------------------------------------------------------

  unsigned int Nchars = Nints*(n-1)/8;
  printf("Nchars=%u\n", Nchars);
  ElGamalDecrypt(Zmessage, a, Nints, p, *h_x);
  unsigned char* message = (unsigned char*) malloc(Nchars*sizeof(unsigned char));
  convertZToString(Zmessage, Nints, message, Nchars);
  printf("Decrypted message: \"%s\"\n", message);
  free(h_x);
  return 0;
  /* Q4 Make the search for the secret key parallel on the GPU using CUDA. */
}
