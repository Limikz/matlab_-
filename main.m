clear all;
clc;
data=load('..\matlab\diabetes1.csv');
data_y=data(1:end,9);
data_x=data(1:end,1:8);

%数据描述
data_max=max(data_x);
data_min=min(data_x);
data_mean=mean(data_x);
data_std=std(data_x);
data_d=quantile(data_x,0.25,1);
data_u=quantile(data_x,0.75,1);

%使用归一化
for i = 1:768
    data_x(i,:)=(data_x(i,:)-data_mean);
    for j=1:8
        data_x(i,j)=data_x(i,j)/data_std(1,j);
    end
end

% 将X变量进行扩充为X_exp
% 我们认为y的取值不光与X1，X2...X8有关，还和其次方项及交叉项有关
% 该过程就是将原本8项,扩充为包含常数项在内的44项
X_exp=data_x;
for i=1:8
    for j=i:8
        X_exp(:,end+1)=data_x(:,i).*data_x(:,j);
    end
end
alpha = 0.01;   %设置步长
len=length(data_y);
lambda=1;
theta = gradient_descent(X_exp, data_y, alpha, len,lambda);
pred=sigmoid(X_exp*theta);      %获得预测值
pred=pred>0.5;
right=length((find(pred==data_y)));
aw=right/len; 
conf_matrix = confusionmat(data_y,double(pred));      %获得混淆矩阵

% 混淆矩阵
mat = conf_matrix;
% 混淆矩阵主题颜色
maxcolor = [191,54,12]; % 最大值颜色
mincolor = [255,255,255]; % 最小值颜色

% 绘制坐标轴
m = length(mat);
imagesc(1:m,1:m,mat)
xticks=(1:m);
xlabel('Predict class','fontsize',10.5)
yticks=(1:m);
ylabel('Actual class','fontsize',10.5)
% 构造渐变色
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];

colormap(mymap)
colorbar()
% 色块填充数字
for i = 1:m
    for j = 1:m
        text(i,j,num2str(mat(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',10);
    end
end
% 图像坐标轴等宽
ax = gca;
ax.FontName = 'Times New Roman';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
axis square
saveas(gca,'conf_matrix.png');
close all;
plotroc(data_y',pred');     %roc曲线
frame = getframe(gcf);
ro = frame2im(frame);
imwrite(ro,'roc.png');
close all;
