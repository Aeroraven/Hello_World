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

%using matrices - select cols
mat = rand(7,123);
ov = [29 36 45 52 82 87 99 100 101 102 103 107 108 109 111 ...
    112 113 114 115 116 117 118 119 120 121 122 123];
vp = mat(1:5,[1:25 ov(1:25)]);
disp(mat);