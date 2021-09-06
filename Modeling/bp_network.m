%using BPNN
clc,clear
boy_heights = (rand(1,200)-0.5)*7+175;
girl_heights = (rand(1,200)-0.5)*7+170;
boy_weights = (rand(1,200)-0.5)*7+75;
girl_weights = (rand(1,200)-0.5)*57+60;
boy_y = [ones(1,200);zeros(1,200)];
girl_y = [zeros(1,200);ones(1,200)];
boy_x = [boy_heights;boy_weights];
girl_x = [girl_heights;girl_weights];
x = [boy_x girl_x];
y = [boy_y girl_y];
net = feedforwardnet(4);
netx = train(net,x,y);
disp('train-done');
disp(netx([175;60]));
disp('prediction-done');