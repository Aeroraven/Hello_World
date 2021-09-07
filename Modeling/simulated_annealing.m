%todo
clc,clear;
points=[37	6
38	48
82	32
68	84
82	17
74	33
59	99
16	58
69	93
51	49
9	38
68	90
8	64
98	13
73	5
51	16
30	84
40	55
73	96
72	19
66	82
77	79
92	10
18	31
95	36
59	11
85	75
18	10
4	62
5	83];
%generating initial solution
SZ=size(points);
N=SZ(1);
A=randperm(N);
Acost=0;
for i=1:N-1
    Acost=Acost+dist(points(A(i),:),points(A(i+1),:)'); 
end
Acost=Acost+dist(points(A(N),:),points(A(1),:)');
for it=1:1000
    u=randi([1,N],1,1);
    v=randi([1,N],1,1);
    Bcost=0;
    B=A;
    temp=B(u);
    B(u)=B(v);
    B(v)=temp;
    for i=1:N-1
        Bcost=Bcost+dist(points(B(i),:),points(B(i+1),:)'); 
    end
    Bcost=Bcost+dist(points(B(N),:),points(B(1),:)');
    if Bcost<Acost
        Acost=Bcost;
        A=B;
    end
    fprintf("Initial Round %d, Dist=%f\n",it,Acost);
end
%solve
Ts=1500;
Te=0.1^10;
Tr=0.999;
C=A;
Ccost=Acost;
T=Ts;
Cnt=0;
while 1
    Cnt=Cnt+1;
    B=A;
    Bcost=0;
    u=randi([1,N],1,1);
    v=randi([1,N],1,1);
    B(1,u:v)=fliplr(B(1,u:v));
    for i=1:N-1
        Bcost=Bcost+dist(points(B(i),:),points(B(i+1),:)'); 
    end
    Bcost=Bcost+dist(points(B(N),:),points(B(1),:)');
    df=Bcost-Acost;
    if Bcost<Ccost
        C=B;
        Ccost=Bcost;
    end
    if df<0
        Acost=Bcost;
        A=B;
    else
        rp=rand();
        if rp<=exp(-df/T)
            Acost=Bcost;
            A=B;
        end
    end
    T=T*Tr;
    if T<Te
        break
    end
    fprintf("SA Round %d-%f, Dist=%f\n",Cnt,T,Ccost);
end
%output
sol=zeros(N+1,2);
for i=1:N
    sol(i,1)=points(C(i),1);
    sol(i,2)=points(C(i),2);
end
sol(N+1,1)=points(C(1),1);
sol(N+1,2)=points(C(1),2);
figure;
scatter(points(:,1)',points(:,2)');
hold on;
plot(sol(:,1)',sol(:,2)');
