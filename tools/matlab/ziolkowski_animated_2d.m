%% load data
clear;
[filename, folder] = uigetfile('../../*.mat', 'Select result data');
load(fullfile(folder, filename));
%% init data
x = 0:GridPointSize(1):DeviceDimension(1);
y = 0:GridPointSize(2):DeviceDimension(2);
Z = 2; 
%y=0:1:2;
[X,Y]=meshgrid(x,y);
T=size(e_z,4);

dim_bez=['x';'y';'z'];
em_bez=['e','h'];
rec_num=size(records,1);

lim = zeros(3,rec_num*2);
data = zeros([size(e_z) rec_num]);

%% merge data
for n=1:rec_num
    data(:,:,:,:,n)=eval(cell2mat(records(n)));
end
for bu=1:2
    for d=1:3
        n=1;
        if any(strcmp(records,[em_bez(bu) '_' dim_bez(d)]))
            temp(:,:,:,:,n)=eval([em_bez(bu) '_' dim_bez(d)]);
            n=n+1;
        end
    end
    data(:,:,:,:,bu+rec_num)=vecnorm(temp,2,5);
end
data(:,:,:,:,3+rec_num)=vecnorm(data(:,:,:,:,(end-1):end),2,5);
records((rec_num+1):(rec_num+3))={'I_e','I_m','I_{em}'};
rec_num=rec_num+3;

%% init plot limits
lim(1,1)=0;
lim(1,2)=DeviceDimension(1);
lim(2,1)=0;
lim(2,2)=DeviceDimension(2);
for n=1:rec_num
        lim(3,(n-1)*2+1)=min(min(min(min(real(data(2,:,:,:,n))))))-0.1;
        lim(3,(n-1)*2+2)=max(max(max(max(real(data(2,:,:,:,n))))))+0.1;
end

clearvars n d bu dim_bez em_bez x y;
clearvars -regexp ^e_ ^h_ ^i_;

%% plot
x0=10;
y0=400;
width=1200;
height=500;
for t=1:2:(T-1)
    h=figure(1);
    h.Name=['time: ' num2str(t*TimeStepSize) ' [s]'];
    set(gcf,'units','points','position',[x0,y0,width,height])
    for n=1:rec_num
        subplot(ceil(rec_num/4),4,n);
        temp=data(Z,:,:,t,n);
        mesh(X,Y,reshape(temp,size(temp,1)*size(temp,2),size(temp,3)))
        xlim(lim(1,1:2));
        ylim(lim(2,1:2));
        zlim(lim(3,((n-1)*2+1):((n-1)*2+2)));
        title(cell2mat(records(n)));
        view([-5 50]);
    end
    pause(0.005)
end
clearvars temp n t h x0 y0 width height;
