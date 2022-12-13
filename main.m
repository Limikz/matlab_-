clear all;
clc;
data=load('..\matlab\diabetes1.csv');
data_y=data(1:end,9);
data_x=data(1:end,1:8);

%��������
data_max=max(data_x);
data_min=min(data_x);
data_mean=mean(data_x);
data_std=std(data_x);
data_d=quantile(data_x,0.25,1);
data_u=quantile(data_x,0.75,1);

%ʹ�ù�һ��
for i = 1:768
    data_x(i,:)=(data_x(i,:)-data_mean);
    for j=1:8
        data_x(i,j)=data_x(i,j)/data_std(1,j);
    end
end

% ��X������������ΪX_exp
% ������Ϊy��ȡֵ������X1��X2...X8�йأ�������η���������й�
% �ù��̾��ǽ�ԭ��8��,����Ϊ�������������ڵ�44��
X_exp=data_x;
for i=1:8
    for j=i:8
        X_exp(:,end+1)=data_x(:,i).*data_x(:,j);
    end
end
alpha = 0.01;   %���ò���
len=length(data_y);
lambda=1;
theta = gradient_descent(X_exp, data_y, alpha, len,lambda);
pred=sigmoid(X_exp*theta);      %���Ԥ��ֵ
pred=pred>0.5;
right=length((find(pred==data_y)));
aw=right/len; 
conf_matrix = confusionmat(data_y,double(pred));      %��û�������

% ��������
mat = conf_matrix;
% ��������������ɫ
maxcolor = [191,54,12]; % ���ֵ��ɫ
mincolor = [255,255,255]; % ��Сֵ��ɫ

% ����������
m = length(mat);
imagesc(1:m,1:m,mat)
xticks=(1:m);
xlabel('Predict class','fontsize',10.5)
yticks=(1:m);
ylabel('Actual class','fontsize',10.5)
% ���콥��ɫ
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];

colormap(mymap)
colorbar()
% ɫ���������
for i = 1:m
    for j = 1:m
        text(i,j,num2str(mat(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',10);
    end
end
% ͼ��������ȿ�
ax = gca;
ax.FontName = 'Times New Roman';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
axis square
saveas(gca,'conf_matrix.png');
close all;
plotroc(data_y',pred');     %roc����
frame = getframe(gcf);
ro = frame2im(frame);
imwrite(ro,'roc.png');
close all;
