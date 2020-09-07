#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define MAX 10001
int min(int x,int y){
  if(x<y) return x;
  return y;
}
struct node{
  int u,e,w;//u起始，e终点，w钱
}edge[900010],temp;//表示路
int book[100100];//表示修码头的钱，费用为MAX可代表无法修码头
int pre[100100];
int flag=0;
void init(int n){
  int i;
  for(i=0;i<=n;i++){
    pre[i]=i;
  }//这里把第零个虚拟节点一起考虑在内，到时候如果最短路径连通图包含了零节点，就是需要码头参与，如果没有零节点照样联通，就是无码头参与
}
int find(int x){
  if(pre[x]==x) return x;
  else return pre[x]=find(pre[x]);//一路向上，寻找根节点，，用来看整张图是否连通
}
int f_first(int n,int m){//有码头参与
  init(n);
  int sum=0;//总价格计算
  flag=0;
  for(int i=0;i<m;i++){
    if(edge[i].w==MAX) continue;
    int t1=find(edge[i].u);
    int t2=find(edge[i].e);
    if(t1!=t2||edge[i].w<0){//如果起点终点还没与联通，就让他们联通，已经联通但是修路赚钱，不修白不修
    pre[t1]=t2;//让他们联通
    sum+=edge[i].w;
    if(edge[i].u==0) flag++;//如果用上了虚拟节点，说明用了码头
    }
  }
  return sum;
}
int f_second(int n,int m){//没有码头参与
  init(n);  
  int sum=0;  
  for(int i=0; i<m; i++){    
    if(edge[i].u==0) continue;    
    int t1=find(edge[i].u);    
    int t2=find(edge[i].e);    
    if(t1!=t2||edge[i].w<0){      
      pre[t1]=t2;    
      sum+=edge[i].w;
    }
  }
  return sum;
}
int main(){
  int m,n;
  scanf("%d%d",&m,&n);
  int i,j;
  for(i=0;i<m;i++){
    scanf("%d%d%d",&edge[i].u,&edge[i].e,&edge[i].w);
  }
  for(i=1;i<=n;i++){
    scanf("%d",&book[i]);
  }
  for(i=1;i<=n;i++){
    edge[m].u=0;
    edge[m].e=i;
    if(book[i]==-1) edge[m].w=MAX;
    else edge[m].w=book[i];
    m++;//把码头变成路，并且和第0个"虚拟"的节点联通
  }
  for(i=0;i<m-1;i++){
    for(j=i+1;j<m;j++){
      if(edge[i].w>edge[j].w){
        temp=edge[i];
        edge[i]=edge[j];
        edge[j]=temp;
      }
    }
  }//然后对路径长度做出排序就可以依次取最短路径进行城市建设了
  int a1=f_first(n,m);
  if(flag==1){//只修了一个码头，那么说明把零节点拿走以后图不能完全联通
    int a2=f_second(n,m);
    printf("%d",a2);
  }
  else printf("%d",min(a1,f_second(n,m)));
  return 0;
}