#include <stdio.h>
#include <stdlib.h>

void main() {

	FILE* file = fopen("data.txt","r");//open a file called data.txt for reading
	
	int n;
	fscanf(file, "%d", &n);//reads an int from a file and stores it in &n

	int* data = (int*) malloc(n*sizeof(int));

	for (int m = 0; m < n; m++) {
		fscanf(file, "%d", data + m);
	}
	fclose(file);

	for (int m = 0; m < n; m++) {
		printf("data[%d] = %d\n", m, data[m]);
	}
	free(data);

}
