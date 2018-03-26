#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "functions.h"

int main (int argc, char **argv) {

  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  //seed value for the randomizer 
  double seed = clock()+rank; //this will make your program run differently everytime
  //double seed = rank; //uncomment this and your program will behave the same everytime it's run

  srand(seed);

  //delcare sotrage for an ElGamal cryptosystem
  unsigned int p, g, h;

  if(rank == 0) {
  	//begin with rank 0 getting user's input
  	unsigned int n, x;

  	/* Q3.1 Make rank 0 setup the ELGamal system and
    	broadcast the public key information */
  	//printf("Enter a number of bits: "); fflush(stdout);
  	//char status = scanf("%u",&n);

  	//make sure the input makes sense
  	//if ((n<3)||(n>31)) {//Updated bounds. 2 is no good, 31 is actually ok
    //	printf("Unsupported bit size.\n");
    //	return 0;   
  	//}
  	//printf("\n");

  	//setup an ElGamal cryptosystem
	n = 22; //seems like a reasonable number to fix n at
  	setupElGamal(n,&p,&g,&h,&x);
  }
  
  MPI_Bcast(&p, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&g, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);


  //Suppose we don't know the secret key. Use all the ranks to try and find it in parallel
  if (rank==0)
    printf("Using %d processes to find the secret key...\n", size);

  /*Q3.2 We want to loop through values i=0 .. p-2
     determine start and end values so this loop is 
     distributed amounst the MPI ranks  */
  unsigned int N = p-1; //total loop size
  unsigned int start, end;
  
  start = rank*N/size; 
  end = start + N/size;

  double Stime = MPI_Wtime();
  //loop through the values from 'start' to 'end'
  for (unsigned int i=start;i<end;i++) {
    if (modExp(g,i+1,p)==h)
      printf("Secret key found! x = %u \n", i+1);
  }
  if(rank == 0) {
  	double Etime = MPI_Wtime();
  	double time = Etime - Stime;
  	double tp = N/time;
  	printf("Time taken: %f\n", time);
  	printf("Throughput: %f\n", tp);
  }
  
  MPI_Finalize();

  return 0;
}
