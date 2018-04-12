#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "omp.h"
#include "functions.h"

int main (int argc, char **argv) {

  int Nthreads = atoi(argv[1]);

  omp_set_num_threads(Nthreads);

	//seed value for the randomizer 
  double seed = clock(); //this will make your program run differently everytime
  //double seed = 0; //uncomment this and your program will behave the same everytime it's run

  srand(seed);

  //declare storage for an ElGamal cryptosytem
  unsigned int p, g, h, x;

  //begin with rank 0 getting user's input
	unsigned int n;

  printf("Enter a number of bits: "); fflush(stdout);
  char status = scanf("%u",&n);

  //make sure the input makes sense
  if ((n<9)||(n>31)) {//Updated bounds. 8 is no good (need to encode chars)
  	printf("Unsupported bit size.\n");
		return 0;  	
  }
  printf("\n");

	//NOTE: supressed all outputs except for Nthreads, time, and throughput, for ease of making graphs
	//n = 20;
	//printf("Entering %u bits (manual override)\n", n);

  //setup an ElGamal cryptosystem
  setupElGamal(n,&p,&g,&h,&x);

  int bufferSize = 1024;
  unsigned char *message = (unsigned char *) malloc(bufferSize*sizeof(unsigned char));

  //populate the string with a message
  strcpy(message, "Hello, this is the message as a string.");
  //strcpy(message, "four");
  printf("Message = \"%s\"\n", message);

  /* Q1.1 Finish this line   */
  unsigned int charsPerInt = (n-1)/8;

  padString(message, charsPerInt);
  printf("Padded Message = \"%s\"\n", message);

  unsigned int Nchars = strlen(message);
  unsigned int Nints  = strlen(message)/charsPerInt;

  //storage for message as elements of Z_p
  unsigned int *Zmessage = 
      (unsigned int *) malloc(Nints*sizeof(unsigned int)); 
  
  //storage for extra encryption coefficient 
  unsigned int *a = 
      (unsigned int *) malloc(Nints*sizeof(unsigned int)); 

  // cast the string into an unsigned int array
  convertStringToZ(message, Nchars, Zmessage, Nints);
  
  //Encrypt the Zmessage with the ElGamal cyrptographic system
  ElGamalEncrypt(Zmessage,a,Nints,p,g,h);

  printf("The encrypted text is: [ ");
  for (unsigned int i=0;i<Nints;i++) {
    printf("(%u,%u) ", Zmessage[i], a[i]);
  }
  printf("]\n");

  //Converting to Ciphertext *Bonus* -------------------------------------------------------------------------
  printf("\n*BONUS* Ciphertext: ");
  unsigned int lpi = 0; //letters (mod 26) per integer
  while(pow(26,lpi) < p){
	  lpi++;
  }
  //printf("LPI: %u\n", lpi);
  unsigned int* cs = malloc(lpi*sizeof(unsigned int));
  unsigned int* cs2 = malloc(lpi*sizeof(unsigned int));
  for(unsigned int i = 0; i < Nints; i++){
	  unsigned int temp = Zmessage[i];
	  unsigned int temp2 = a[i];
	  for(int j = lpi-1; j >= 0; j--){
		  cs[j] = (temp % 26) + 65;
		  cs2[j] = (temp2 % 26) + 65;
		  temp = temp / 26;
		  temp2 = temp2 / 26;
	  }
	  for(int j = 0; j < lpi; j++)
	  	printf("%c", (unsigned char)cs[j]);
	  printf(" ");
	  for(int j = 0; j < lpi; j++)
	  	printf("%c", (unsigned char)cs2[j]);
	  printf(" ");
  }
  free(cs);
  free(cs2);
  printf("\n\n");
  //End Bonus ------------------------------------------------------------------------------------------------

  //Decrypt the Zmessage with the ElGamal cyrptographic system
  ElGamalDecrypt(Zmessage,a,Nints,p,x);

  convertZToString(Zmessage, Nints, message, Nchars);

  printf("Decrypted Message = \"%s\"\n", message);
  printf("\n");


  //Suppose we don't know the secret key. Use OpenMP threads to try and find it in parallel
  //printf("Using %d OpenMP threads to find the secret key...\n", Nthreads);

  /* Q2.3 Parallelize this loop with OpenMP   */
  double startTime = omp_get_wtime();
  int found = 0;
	#pragma omp parallel for shared(found)
  for (unsigned int i=0; i < p-1; i++) {
    if(found == 1){
		continue;
	}
	if (modExp(g,i+1,p)==h) {
      printf("Secret key found! x = %u \n", i+1);
	  found = 1;
    } 
  }
  double endTime = omp_get_wtime();

  double totalTime = endTime-startTime;
  double work = (double) p;
  double throughput = work/totalTime;

  printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);

  //printf("%u,%g,%g\n", Nthreads, totalTime, throughput);
  free(message);
  free(Zmessage);
  free(a);

  return 0;
}
