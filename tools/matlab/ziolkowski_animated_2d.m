%% load data
clear;
[filename, folder] = uigetfile('../../*.mat', 'Select result data');
load(fullfile(folder, filename));
%% init data
x = 0:GridPointSize(1)*2:DeviceDimension(1);
if Dimension>1
    y = 0:GridPointSize(2)*2:DeviceDimension(2);
else
    y=1;
end
Z = round(size(e_z,1)/2); 

[X,Y]=meshgrid(x,y);
T=size(e_z,4);

dim_bez=['x';'y';'z'];
em_bez=['e','h'];
rec_num=size(records,1);

lim = zeros(3,rec_num*2);
data = zeros([size(e_z) rec_num]);
%ToDo: use complete eps and mu
mu_0=4*pi*1e-7;
eps_0=1/(mu_0*physconst('LightSpeed')^2); %8.8542e-12
const=[eps_0/2;mu_0/2];

%% merge data
for n=1:rec_num
    data(:,:,:,:,n)=eval(cell2mat(records(n)));
end
%% compute work-density %ToDo: take into account, that E and H have offset 
for bu=1:2
    n=1;
    for d=1:3
        if any(strcmp(records,[em_bez(bu) '_' dim_bez(d)]))
            temp(:,:,:,:,n)=eval([em_bez(bu) '_' dim_bez(d)]);
            n=n+1;
        end
    end
    data(:,:,:,:,bu+rec_num)=const(bu)*sum(temp.^2,5);
end
data(:,:,:,:,3+rec_num)=sum(data(:,:,:,:,(end-1):end),5);
records((rec_num+1):(rec_num+3))={'w_e','w_m','w_{em}'};
rec_num=rec_num+3;

%% init plot limits
lim(1,1)=0;
lim(1,2)=x(end);
lim(2,1)=0;
lim(2,2)=y(end);
for n=1:rec_num
        lim(Dimension+1,(n-1)*2+1)=min(min(min(min(real(data(Z,:,:,:,n))))))-0.1;
        lim(Dimension+1,(n-1)*2+2)=max(max(max(max(real(data(Z,:,:,:,n))))))+0.1;
end
%%
clearvars n d bu dim_bez em_bez x y const mu_0 eps_0;
clearvars -regexp ^e_ ^h_ ^inv\d{2}$ ^d\d{2}$;

%% plot
x0=10;
y0=400;
width=1200;
height=500;
plot_dim = 2;
plot_int = 75;

for t=1:1:(T-1)
    h=figure(1);
    h.Name=['time: ' num2str(t*TimeStepSize) ' [s]'];
    set(gcf,'units','points','position',[x0,y0,width,height])
    for n=1:rec_num
        subplot(ceil(rec_num/3),3,n);
        
        if Dimension == 1 
            temp=data(Z,:,:,t,n);
            plot(X(1,:),reshape(temp,size(temp,1)*size(temp,2),size(temp,3)));
            ylim(lim(2,((n-1)*2+1):((n-1)*2+2)));
        elseif size(data,2)==1 || plot_dim==1 
            temp=data(1:plot_int:size(data,1),1:plot_int:size(data,2),:,t,n);
            plot(X(1,:),reshape(temp,size(temp,1)*size(temp,2),size(temp,3)));
            ylim(lim(Dimension+1,((n-1)*2+1):((n-1)*2+2)));
        elseif Dimension == 2 || (Dimension == 3)% && Z == 1)
            temp=data(Z,:,:,t,n);
            mesh(X,Y(:,1),reshape(temp,size(temp,1)*size(temp,2),size(temp,3)));
            ylim(lim(2,1:2));
            zlim(lim(Dimension+1,((n-1)*2+1):((n-1)*2+2)));
            view([10 80]);
        else
            
        end
        xlim(lim(1,1:2));
        title(cell2mat(records(n)));
    end
    pause(0.005)
end
clearvars temp n t h x0 y0 width height;
