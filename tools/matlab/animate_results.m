%% load data
clear;
[filename, folder] = uigetfile('../../*.mat', 'Select result data');
load(fullfile(folder, filename));
%% init data
y = 1;
z = 1;
x = linspace(0,DeviceDimension(1),size(e_z,3));
if Dimension > 1
    y = linspace(0,DeviceDimension(2),size(e_z,2));
    if Dimension > 2
        z = linspace(0,DeviceDimension(3),size(e_z,1));
    end
end

[X,Y,Z]=meshgrid(x,y,z);
%%
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
% data(:,:,:,:,1+rec_num)=sum(data(:,:,:,:,1:3),5);
% records(rec_num+1)={'w'}; rec_num=rec_num+1;

% data(:,:,:,:,3+rec_num)=sum(data(:,:,:,:,(end-1):end),5);
% records((rec_num+1):(rec_num+3))={'w_e','w_m','w_{em}'};
% rec_num=rec_num+3;

%% init plot limits
lim(1,1)=0;
lim(1,2)=x(end);
lim(2,1)=0;
lim(2,2)=y(end);
for n=1:rec_num
        lim(Dimension+1,(n-1)*2+1)=1.1*min(min(min(min(min(real(data(:,:,:,:,n)))))));
        lim(Dimension+1,(n-1)*2+2)=1.1*max(max(max(max(max(real(data(:,:,:,:,n)))))))+1e-10;
end
%%
clearvars n d bu dim_bez em_bez x y const mu_0 eps_0;
clearvars -regexp ^e_ ^h_ ^inv\d{2}$ ^d\d{2}$;

%% plot
if 0
    filename = 'testAnimated.gif';
    
    x0=10;
    y0=400;
    width=1200;
    height=800;
    plot_dim = 2;
    plot_int = 5;
    for t=1:1:T-1%:250%(T-1)
        h=figure(1)
        h.Name=['time: ' num2str(t*SimEndTime/(T-1)) ' [s]'];
        set(gcf,'units','points','position',[x0,y0,width,height])
        for n=1:rec_num
            subplot(ceil(rec_num/3),3,n);
            if Dimension == 1 
                temp=data(1,:,:,t,n);
                plot(X(1,:),reshape(temp,size(temp,1)*size(temp,2),size(temp,3)));
                ylim(lim(2,((n-1)*2+1):((n-1)*2+2)));
            elseif size(data,2)==1 || plot_dim==1 
                for z = 1:plot_int:size(data,1)
                    temp=data(z,1:plot_int:size(data,2),:,t,n);
                    %temp=data(1:plot_int:size(data,2),z,:,t,n);
                    plot(X(1,:,1),reshape(temp,size(temp,1)*size(temp,2),size(temp,3)));  
                    hold on;
                end
                ylim(lim(Dimension+1,((n-1)*2+1):((n-1)*2+2)));
                hold off;
            elseif Dimension == 2 || ((Dimension == 3) && (size(data,1)== 1 || plot_dim==2))
                temp=data(round(size(data,1)/2),1:end,:,t,n);
                %temp=data(:,round(size(data,1)/2)-10,:,t,n);
                mesh(X(1:end,:,1),Y(1:end,:,1),reshape(temp,size(temp,1)*size(temp,2),size(temp,3)));
                ylim(lim(2,1:2));
                zlim(lim(Dimension+1,((n-1)*2+1):((n-1)*2+2)));
                view([45 45]);
            else
                isosurface(X,Y,Z,data(:,:,:,t,n),0.4*max(max(max(max(data(:,:,:,:,n))))))
                %contourslice(X,Y,Z,data(:,:,:,t,n),linspace(0,DeviceDimension(3),3),linspace(0,DeviceDimension(3),3),linspace(0,DeviceDimension(3),3))
                view([0 70]);
                xlim([0 max(max(max(X)))]);
                ylim([0 max(max(max(Y)))]);
                zlim([0 max(max(max(Z)))]);
                if t == 1
                    camroll(-90);
                end
            end
            xlim(lim(1,1:2));
            %xlim([50,70]*1e-6);
            title(cell2mat(records(n)));
        end
%         % Capture the plot as an image 
%         frame = getframe(h); 
%         im = frame2im(frame); 
%         [imind,cm] = rgb2ind(im,256); 
%         % Write to the GIF File 
%         if t == 1 
%           imwrite(imind,cm,filename,'gif','DelayTime',0.4, 'Loopcount',inf); 
%         elseif mod(t,4)==0
%           imwrite(imind,cm,filename,'gif','DelayTime',0.4,'WriteMode','append'); 
%         end 
        pause(0.005)
    end
end

clearvars temp n t h x0 y0 width height
