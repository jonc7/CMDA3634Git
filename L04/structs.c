#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {

	float x,y,z;

} point;

void pointPrintPoint(point p) {

	printf("Point has coordinates (%f,%f,%f)\n", p.x,p.y,p.z);

}

void pointSetZero(point* p) {

	/*(*p).x = 0;
	p->y = 0; these also work
	p->z = 0;*/
	
	point pp = *p;
	pp.x = 0;
	pp.y = 0;
	pp.z = 0;

}

float pointDistanceToOrigin(point p) {

	float dist = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
	return dist;

}

void main() {

	point p;
	
	p.x = 1.0;
	p.y = 2.0;
	p.z = 3.0;
	float dist;
	dist = pointDistanceToOrigin(p);
	
	printf("dist = %f\n",dist);	
	pointSetZero(&p);
	pointPrintPoint(p);

}
