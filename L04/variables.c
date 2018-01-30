#include <stdio.h>
#include <stdlib.h>

void main() {

	int a,b,c;//allocates an integer called 'a'
	int *pt_a;//allocates a pointer to an integer, called pt_a

	a=13;
	b=4;
	pt_a = &a;//&a is the address of a
	int* pt_b = &b;//&a is the address of a
	c = *(pt_a+1);//*pt_a is the value that pt_a+1 points to (i.e. b)

	int *array;
	int alsoArray[10];//this also allocates an array of size 10 entries
	array = (int *) malloc(10*sizeof(int));//allocates memory for an array of size 10

	int n = 0;
	for(n = 0; n<10; n++){
		if (n==3)
			continue;//skip the remaining statements in the loop and do another iteration
		else printf("Not 3");
		array[n] = n;
	}

	n = 0;
	while(n<10) {
		printf("array[%d] = %d\n", *(array+n));
		n++;
	}

	/* more data types */
	float fa;//32-bit floating point number
	double da;//64-bit floating point number
	char ch;//single character
	char* str;//string (array of characters)
	unsigned int ui;//unsigned integer
	long li;//64-bit integer

	printf("The size of an int is %ld\n", sizeof(int));
	printf("a = %d \n",a);
	printf("b = %d \n",b);
	printf("c = %d \n",c);
	printf("a+b = %d \n",a+b);
	printf("a-b = %d \n",a-b);
	printf("a*b = %d \n",a*b);
	printf("a/b = %d \n",a/b);
	printf("a mod b = %d \n",a%b);
	printf("a is located at %p\n", pt_a);
	printf("b is located at %p\n", pt_b);

	int *array;
	array = (int*) malloc(10*sizeof(int));
	for(int i = 0; i < 10; i++) {
		printf("a[%d] = %d\n",i,a[i]);
	}
}
