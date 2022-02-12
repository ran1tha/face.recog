function [Var,DimvN,net,U_red,map,Z,mu] = trainNN3(fea,gnd,Dim,Neurons)

[m,n] = size(fea);  %Number of data and featuress
f = 15;             %Number of distinct faces
Var = zeros(1,length(Dim));                 %Initialize variance percentage array
DimvN = zeros(length(Dim),length(Neurons)); %Initialize True positives matrix
com = 0;                                    %For tracking progress
tot = length(Dim)*length(Neurons);          %For tracking progress

for idx1=1:length(Dim)
    
k = Dim(idx1);      %Dimension to be reduced to 
X = fea';           %Features x Examples matrix
gndY = gnd;         %Grounth truth

%Divide data into Test, Train and Validation sets
tr_data = X(:,3:10);   tr_y = gndY(3:10);
ts_data = X(:,1);      ts_y = gndY(1);
cv_data = X(:,2);      cv_y = gndY(2);
for divide=11:10:m
tr_data = [tr_data X(:,divide+2:divide+9)]; tr_y = [tr_y; gndY(divide+2:divide+9)];
ts_data = [ts_data X(:,divide)]; ts_y = [ts_y; gndY(divide)];
cv_data = [cv_data X(:,divide+1)]; cv_y = [cv_y; gndY(divide+1)];
end

%Randomly shuffle each data set
idx_tr = randperm(size(tr_data,2));
tr_data = tr_data(:,idx_tr); tr_y=tr_y(idx_tr);
idx_ts = randperm(size(ts_data,2));
ts_data = ts_data(:,idx_ts); ts_y=ts_y(idx_ts);
idx_cv = randperm(size(cv_data,2));
cv_data = cv_data(:,idx_cv); cv_y=cv_y(idx_cv);


%Concatenate all data sets together
X = [tr_data ts_data cv_data]; gndY = [tr_y; ts_y; cv_y];

%Make gnd into a boolean vector
Y = zeros(f,m); 
for i=1:m
    Y(gndY(i),i) = 1;
end

%Identify the Train, Test, Validation indices
trn=1:length(tr_y);                                                           %Train data
ts=length(tr_y)+1:length(tr_y)+length(ts_y);                                  %Test data
cv=length(tr_y)+length(ts_y)+1:length(tr_y)+length(ts_y)+length(cv_y);       %Validation data


%% Principle Component Analysis to reduce dimension from 1024 to k

%Mean normalization (Substract the mean face from the set of faces)
mu = mean(X');
X = X-mu';

m1=max(trn);                         %Number of training data (PCA IS DONE ONLY FOR TRAINING DATA)
sigma = (1/m1)*X(:,trn)*X(:,trn)';    %Covariance matrix of data
[U,S,~] = svd(sigma);               %Single Value Decomposition
U_red = U(:,1:k);                   %Dimensionality reduction
Z = U_red'*X;                       %Dimension reduced data

%Check the percentage of the variance retained
s_k=0;
for p=1:k
    s_k=s_k+S(p,p);                 %Variance of Dimension reduced data 
end
s_n=0;
s_n = sum(diag(S));                 %Variance of data before reducing the dimension

Var(idx1) = s_k/s_n;                %Percentage of the variance retained

for idx2=1:length(Neurons)
    
neuron = Neurons(idx2);             %Number of Neurons in the hidden layer    

%% Neural Network
net = patternnet([neuron neuron]);     %Initialize the neural net
net.trainFcn = 'trainscg';      %Define training fcn
net.divideFcn = 'divideind';    %Define how the data set is divided
net.divideParam.trainInd = trn; %Training data
net.divideParam.testInd = ts;   %Testing data
net.divideParam.valInd = cv;    %Validation data
[net,tr] = train(net,Z,Y);           %Train nnet
com = com+1;                    %For tracking progress
fprintf('Dimension: %d | Neurons: %d | Training... | %d/%d Completed!\n',k,neuron,com,tot);


%% Calculate the accuracy from test data

y=net(Z(:,ts));              %Feed only the test data back the trained NNet
map = zeros(f,f);            %Initialize heat map
c=0; Y_ts=Y(:,ts);           %Get test data ground truth
for i=1:length(ts)
    [d,c] = max(Y_ts(:,i));    %Actual face
    [p,q] = max(y(:,i));    %Predicted face
    map(c,q)=map(c,q)+1;    %Save to the heat map
 
end
tp=sum(diag(map));                   %Number of True positives are in the diagonal of the heat map
DimvN(idx1,idx2) = tp/length(ts);    %Update True positives matrix
% heatmap(map);

%% Generate regression plot
out = vec2ind(net(Z));
actual = vec2ind(Y);


trOut = out(tr.trainInd);
vOut = out(tr.valInd);
tsOut = out(tr.testInd);
trTarg = actual(tr.trainInd);
vTarg = actual(tr.valInd);
tsTarg = actual(tr.testInd);
%plotregression(trTarg,trOut,'Train',vTarg,vOut,'Validation',tsTarg,tsOut,'Testing')


end
end
end