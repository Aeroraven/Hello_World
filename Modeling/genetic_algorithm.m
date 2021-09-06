clc,clear;
%GA-based TSP Solver
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
%figure;
scatter(points(:,1)',points(:,2)');

%generate initial group
INIT = 2000;
ROUNDS = 1000;
MUTANT_RATE = 80; % OF 100
points_size=size(points);
initial_group=zeros(INIT,points_size(1));
initial_group_cost=zeros(INIT,1);
for it=1:INIT
    p=randperm(points_size(1));
    cur_cost=0;
    for i=1:(points_size(1)-1)
        cur_cost = cur_cost + sqrt((points(i,1)-points(i+1,1))^2+(points(i,2)-points(i+1,2))^2);
    end
    while(1)
        new_p=p;
        u=randi([1,points_size(1)],1,1);
        v=randi([1,points_size(1)],1,1);
        temp=new_p(u);
        new_p(u)=new_p(v);
        new_p(v)=temp;
        new_cost=0;
        for i=1:(points_size(1)-1)
            new_cost = new_cost + sqrt((points(new_p(i),1)-points(new_p(i+1),1))^2+(points(new_p(i),2)-points(new_p(i+1),2))^2);
        end
        if new_cost<=cur_cost
            p=new_p;
            cur_cost=new_cost;
        else
            break;
        end
    end
    for j=1:points_size(1)
        initial_group(it,j)=p(j);
    end
    initial_group_cost(it,1)=cur_cost;
end
%encode
initial_group = initial_group / points_size(1);
for g=1:ROUNDS
    op_group = initial_group;
    %crossover
    crossover_seq = randperm(INIT);
    for i=1:2:INIT
        crossover_pt=randi([1,points_size(1)-1],1,1);
        temp=op_group(crossover_seq(i),[crossover_pt:points_size(1)]);
        op_group(crossover_seq(i),[crossover_pt:points_size(1)])=op_group(crossover_seq(i+1),[crossover_pt:points_size(1)]);
        op_group(crossover_seq(i+1),[crossover_pt:points_size(1)])=temp;
    end
    %mutant
    for i=1:INIT
        u=randi([1,points_size(1)],1,1);
        v=randi([1,points_size(1)],1,1);
        temp=op_group(i,u);
        op_group(i,u)=op_group(i,v);
        op_group(i,v)=temp;
    end
    %mutant2
    for i=1:INIT
        u=randi([1,points_size(1)],1,1);
        op_group(i,u)=op_group(i,u)+0.05*(rand()-0.5);
    end
    %concatenate
    group = [initial_group;op_group];
    group_value = zeros(2*INIT,1);
    for i=1:2*INIT
        select_l=group(i,:);
        [select_s,ind]=sort(select_l);
        for j=1:points_size(1)-1
            group_value(i)=group_value(i)+sqrt((points(ind(j),1)-points(ind(j+1),1)).^2+(points(ind(j),2)-points(ind(j+1),2)).^2);
        end
        j=points_size(1);
        group_value(i)=group_value(i)+sqrt((points(ind(j),1)-points(ind(1),1)).^2+(points(ind(j),2)-points(ind(1),2)).^2);
    end
    group_concat=[group group_value];
    group_concat=sortrows(group_concat,points_size(1)+1);
    %select
    group_concat=group_concat(1:INIT,:);
    initial_group=group_concat(:,1:points_size(1));
    fprintf('Round %d Ends, Optimal Dist:%f\n',g,group_concat(1,points_size(1)+1));
end
%output final solution
final_sol=group_concat(1,1:points_size(1));
[xt,final_sol]=sort(final_sol);
plot_pts=zeros(points_size(1)+1,2);
for i=1:points_size(1)
    plot_pts(i,1)=points(final_sol(i),1);
    plot_pts(i,2)=points(final_sol(i),2);
end
plot_pts(points_size(1)+1,1)=points(final_sol(1),1);
plot_pts(points_size(1)+1,2)=points(final_sol(1),2);
pltx=plot_pts(:,1)';
plty=plot_pts(:,2)';
figure;
scatter(points(:,1)',points(:,2)');
hold on
plot(pltx,plty);
%output digit
fprintf('Optimal Distace:%f\n',group_concat(1,points_size(1)+1));