Len = 100;
% 生成卷积层前的数据
data = zeros(Len*(12*12+6)+32,9);
x8 = [];
for i = 1:150
    x8(:,1) = reshape(x1(1:12,1:12,i)',[12*12,1]);x8(:,2) = reshape(x1(1:12,2:13,i)',[12*12,1]);x8(:,3) = reshape(x1(1:12,3:14,i)',[12*12,1]);
    x8(:,4) = reshape(x1(2:13,1:12,i)',[12*12,1]);x8(:,5) = reshape(x1(2:13,2:13,i)',[12*12,1]);x8(:,6) = reshape(x1(2:13,3:14,i)',[12*12,1]);
    x8(:,7) = reshape(x1(3:14,1:12,i)',[12*12,1]);x8(:,8) = reshape(x1(3:14,2:13,i)',[12*12,1]);x8(:,9) = reshape(x1(3:14,3:14,i)',[12*12,1]);
    data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),1) = x8(:,1);data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),2) = x8(:,2);data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),3) = x8(:,3);
    data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),4) = x8(:,4);data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),5) = x8(:,5);data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),6) = x8(:,6);
    data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),7) = x8(:,7);data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),8) = x8(:,8);data((i-1)*(12*12+6)+1:(i-1)*(12*12+6)+(12*12),9) = x8(:,9);
end
    
data_1 = (2*asin(2*data-1)+pi)/(2*pi);      % 反函数
data = round(data,2);data_1 = round(data_1,2);
data_1m = 1 - data_1;
path = 'C:\Users\30685\Desktop\documents\MATLAB\MNIST\result\data\MNIST_data\MNIST_data';
csvwrite([path,'_1.csv'],data_1(:,1));csvwrite([path,'_1m.csv'],data_1m(:,1));
csvwrite([path,'_2.csv'],data_1(:,2));csvwrite([path,'_2m.csv'],data_1m(:,2));
csvwrite([path,'_3.csv'],data_1(:,3));csvwrite([path,'_3m.csv'],data_1m(:,3));
csvwrite([path,'_4.csv'],data_1(:,4));csvwrite([path,'_4m.csv'],data_1m(:,4));
csvwrite([path,'_5.csv'],data_1(:,5));csvwrite([path,'_5m.csv'],data_1m(:,5));
csvwrite([path,'_6.csv'],data_1(:,6));csvwrite([path,'_6m.csv'],data_1m(:,6));
csvwrite([path,'_7.csv'],data_1(:,7));csvwrite([path,'_7m.csv'],data_1m(:,7));
csvwrite([path,'_8.csv'],data_1(:,8));csvwrite([path,'_8m.csv'],data_1m(:,8));
csvwrite([path,'_9.csv'],data_1(:,9));csvwrite([path,'_9m.csv'],data_1m(:,9));

% 生成平均池化层的数据
x9 = ones((20*Len)*L+512,9);
for i = 1:Len
    for j = 1:4
        for k = 1:4
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,1) = x2((j-1)*3+1,(k-1)*3+1,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,2) = x2((j-1)*3+1,(k-1)*3+2,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,3) = x2((j-1)*3+1,(k-1)*3+3,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,4) = x2((j-1)*3+2,(k-1)*3+1,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,5) = x2((j-1)*3+2,(k-1)*3+2,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,6) = x2((j-1)*3+2,(k-1)*3+3,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,7) = x2((j-1)*3+3,(k-1)*3+1,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,8) = x2((j-1)*3+3,(k-1)*3+2,i)/(3.1);
            x9((i-1)*20*L+(j-1)*4*L+(k-1)*L+1:(i-1)*20*L+(j-1)*4*L+k*L,9) = x2((j-1)*3+3,(k-1)*3+3,i)/(3.1);
        end
    end
end

x9 = (2*asin(2*x9-1)+pi)/(2*pi); 

path = 'C:\Users\30685\Desktop\documents\MATLAB\MNIST\result\data\avg_pool\avg_pool';
csvwrite([path,'_1.csv'],round(1-x9(:,1),3));csvwrite([path,'_2.csv'],round(1-x9(:,2),3));csvwrite([path,'_3.csv'],round(1-x9(:,3),3));
csvwrite([path,'_4.csv'],round(1-x9(:,4),3));csvwrite([path,'_5.csv'],round(1-x9(:,5),3));csvwrite([path,'_6.csv'],round(1-x9(:,6),3));
csvwrite([path,'_7.csv'],round(1-x9(:,7),3));csvwrite([path,'_8.csv'],round(1-x9(:,8),3));csvwrite([path,'_9.csv'],round(1-x9(:,9),3));

x10 = ones(Len*20*L+256,1);
for i = 1:Len
    for j = 1:16
        x10((i-1)*20*L+(j-1)*L+1:(i-1)*20*L+j*L,1) = x4(j,i)/2;
    end
end
x11 = (2*asin(2*x10-1)+pi)/(2*pi); 
csvwrite([path,'avg_pool.csv'],round(1-x10,3));
csvwrite([path,'avg_pool_0.csv'],round(1-x11,3));

x12 = ones(Len*16,1);
for i = 1:Len
    for j = 1:16
        x12((i-1)*16+(j-1)+1:(i-1)*16+j,1) = x4(j,i);
    end
end
x13 = reshape(x12,[16,Len]);
function d = avgpooling(data,m,n)
    shape = size(data);
    a = shape(1,1);b = shape(1,2);
    d = zeros(a/m,b/n);
    for i = 1:a/m
        for j = 1:b/n
            c = data((i-1)*m+1:(i-1)*m+m,(j-1)*n+1:(j-1)*n+n);
            d(i,j) = sum(c,'all')/(m*n);
        end
    end
end