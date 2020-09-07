#include<stdio.h>

int main()
{
	int a[200];
	int n;
	scanf("%d",&n);
	for (int i=1;i<=n;i++)
	{
		scanf("%d",&a[i]);
	}
	int ans=0;
	for (int i=1;i<n;i++)
	{
		int min1=10000,min2=10000,mt1,mt2;
		for (int j=1;j<=n;j++)
		{
			if (a[j]<=min1)
			{
				mt2=mt1;
				min2=min1;
				min1=a[j];
				mt1=j;
			}
			else
			{
				if (a[j]<min2)
				{
					min2=a[j];
					mt2=j;
				}
			}
		}
		a[mt1]=a[mt1]+a[mt2];
		ans+=a[mt1];
		a[mt2]=10001;
	}
	printf("%d",ans);
	return 0;
}