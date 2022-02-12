%% Initialize
clc; close all; clear all;

%Load data file
load Yale_32x32.mat;
Dim = [48];      %Number of Dimensions after dimensionality reduction [Can be input as an array of values]
Neurons = [80];       %Number of neurons in the hidden layer [Can be input as an array of values]
bgt = 12;           %Brightness - For generating artificial data
sf = 1.2;           %Scale factor - For generating artificial data
Numitr = 100;      %Number of times to run the NN - Set to 1 to run only once
ltrmap = zeros(15,15);    %Initialize confusion matrix

 for ind1=1:length(Dim)
    D = Dim(ind1);
    for ind2=1:length(Neurons)
        N = Neurons(ind2);

if Numitr ~=1       %Generate the accuracy vs Numitr plot 
    figure
end
meanaccts=0;        %For plotting
meanaccrs=0;        %For plotting
for ind=1:Numitr            %For determining the accuracy of NN   
%% Setup NN and run it

% Shuffle faces of each person
for shuf=1:11:length(gnd)
    temp1 = fea(shuf:shuf+10,:);    temp2 = gnd(shuf:shuf+10);
    idx = randperm(11);
    temp1 = temp1(idx,:);           temp2 = temp2(idx);
    fea(shuf:shuf+10,:) = temp1;    gnd(shuf:shuf+10) = temp2;
end

%Reserve images for later use [Note that this data isn't fed to the neural network at all]
ltr = 15;                                       %One face of each 15 different faces is reserved
ltrX = fea(1,:);    ltrY = gnd(1);
fea1 = fea(2:11,:);    gnd1 = gnd(2:11);
for later=12:11:165
ltrX = [ltrX; fea(later,:)]; ltrY = [ltrY; gnd(later)];
fea1 = [fea1; fea(later+1:later+10,:)]; gnd1 = [gnd1; gnd(later+1:later+10)];
end
ltrX = ltrX';

fea = fea1; gnd = gnd1;             %Update fea and gnd 

%Generate Artificial Data from the set of 165 images
bright = fea+bgt;                   %Generate data by increasing brightness
dark = fea-bgt;                     %Generate data by reducing brightness
scaled = ((fea-128).*sf)+128;       %Generate data by scaling brightness to both sides
bright2 = fea+20;                   %Generate data by increasing brightness
dark2 = fea-20;                     %Generate data by reducing brightness
scaled2 = ((fea-128).*2)+128;       %Generate data by scaling brightness to both sides
bright3 = fea+30;                   %Generate data by increasing brightness
dark3 = fea-30;                     %Generate data by reducing brightness
scaled3 = ((fea-128).*1.8)+128;     %Generate data by scaling brightness to both sides
bright4 = fea+40;                   %Generate data by increasing brightness
dark4 = fea-40;                     %Generate data by reducing brightness
scaled4 = ((fea-128).*2.2)+128;     %Generate data by scaling brightness to both sides

features = [fea; bright; dark; scaled; bright2; dark2; scaled2; bright3; dark3; scaled3; bright4; dark4; scaled4];     %Concatenate all the data 
ground_truth = [gnd; gnd; gnd; gnd; gnd; gnd; gnd; gnd; gnd; gnd; gnd; gnd; gnd];        %Corresponding ground truth values


%Train the neural network
    %Outputs - Retained variance, heat map of reduced dimension vs number of
    %neurons, trained neural network, Eigen vectors upto the reduced
    %dimensions, heatmap for test data
    
    %Inputs - Features, Ground truth, Dimensions to be reduced to, Number
    %of neurons in the hidden layer

[Var,DimvN,net,U_red,testmap,Z,~] = trainNN(features,ground_truth,D,N);
fprintf('\nVariance Retained: %f%% \n',Var*100);
fprintf('Test Data Accuracy: %f%% \n\n',DimvN*100);

variance(ind1,ind2) = Var; %Variance Matrix
%% Results

%Feed the original 165 images back to the NN and check the accuracy 
load Yale_32x32.mat;    %Load fresh data again

%Preprocess data to feed into the NN
I165 = fea';            
mu = mean(I165');   
I165 = I165-mu';        %Mean Normalization
I165 = U_red'*I165;     %Dimensionality reduction

y=net(I165);            %Get predicted output
map1 = zeros(15,15);    %Initialize a heatmap
c=0;

Y = zeros(15,length(gnd)); %Make gnd into a boolean vector
for k=1:length(gnd)
    Y(gnd(k),k) = 1;
end

for j=1:length(gnd)
    [d,c] = max(Y(:,j));        %Actual face
    [p,q] = max(y(:,j));        %Predicted face
    map1(c,q)=map1(c,q)+1;      %Save to heatmap
 
end

tp=sum(diag(map1));                     %Total True positives
accuracy = (tp/length(gnd))*100;        %Accuracy of 165 Image data set
fprintf('Predicted images out of 165 images: %d | Accuracy: %f%% \n',tp,accuracy);

%Feed the reserved data into the NN and check the accuracy
mu = mean(ltrX');       
ltrXmu = ltrX-mu';        %Mean normalization
ltrXred = U_red'*ltrXmu;     %Dimensionality reduction

y=net(ltrXred);            %Get predicted output
map2 = zeros(15,15);     %Initialize heatmap

c=0;
Y = zeros(15,ltr); %Make gnd into a boolean vector
for k=1:ltr
    Y(ltrY(k),k) = 1;
end

for j=1:length(ltrY)
    [d,c] = max(Y(:,j));    %Actual face
    [p,q] = max(y(:,j));    %Predicted face
    map2(c,q)=map2(c,q)+1;    %Save to heatmap
 
end
ltrmap = ltrmap+map2;         %Update heatmap
tp=sum(diag(map2));                   %Total True positives
accuracy = (tp/length(ltrY))*100;    %Accuracy of reserved data set
fprintf('Predicted images out of %d images: %d | Accuracy: %f%% \n',ltr,tp,accuracy);

% figure;
% heatmap(testmap);            %Generate heatmap for test data

%% For Determining the accuracy of the NN

if Numitr~=1
% accts(ind) = DimvN*100;           %Accuracy array - Test data
% meanaccts(ind) = mean(accts);     %Mean accuracy upto now - Test data
% subplot(2,1,1)
% plot(1:ind,meanaccts,'LineWidth',2)
% xlabel('Number of Iterations'); ylabel('Mean Accuracy upto the iteration % - Test Data');
% title("The Accuracy of Test data set | Dimensions: "+D+" | Neurons: "+N+" | Accuracy: "+meanaccts(ind));
% grid on;
accrs(ind) = accuracy;            %Accuracy array - Test data
meanaccrs(ind) = mean(accrs);     %Mean accuracy upto now - Test data
% subplot(2,1,2)
plot(1:ind,meanaccrs,'LineWidth',2) 
xlabel('Number of runs'); ylabel('Mean Accuracy upto the iteration % - Reserved Data');
title("The Accuracy of Reserved data set | Dimensions: "+D+" | Neurons: "+N+" | Accuracy: "+meanaccrs(ind));
grid on;
end
end
end
 end

 %Plot confusion matrix
 figure; heatmap(ltrmap);