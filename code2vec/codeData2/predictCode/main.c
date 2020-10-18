#include <stdio.h>
#include <assert.h>

int main(){
	int i=0,y;
	float x;
	char a[500],b[500];
	while((a[i]=getchar())!='\n'){
		i++;
	}
	a[i]='\0';
	x=i;
	i=0;
	while((b[i]=getchar())!='\n'){
		i++;
	}
	b[i]='\0';
	y=i;
	
	float count=0;
	int j=0;
	i=0;
	
	printf("%d\n", y);

	while(a[i]!='\0'){
		if(b[j]==a[i]){
			count ++;
			j++; 
			printf("j %d\n", j);
		}
		if(j!=y-1 && i==x-1){
			j++;
			i=0;
		}
		assert(j < y);
		if(j==y-1){
			break;
		}
		i++;
	}
	
	
	if(count/x>=0.5){
		printf("Yes");
	}else{
		printf("No");
	}
	return 0;
}