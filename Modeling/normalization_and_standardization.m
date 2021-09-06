%mapminmax
vector=[1,2,3,4,5];
[y,ps]=mapminmax(vector);
disp(y);
disp(ps);
%mapminmax-2: normalize in lines
vector=[1,2,3,4,5;
        6,7,8,9,10];
[y,ps]=mapminmax(vector);
disp(y);
disp(ps);