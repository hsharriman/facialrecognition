%% Eigenfaces
% Eigenface implentation of facial recognition 
% authors@ Corey Cochran-Lepiz & Hwei-Shin Harriman
% For Quantitative Engineering Analysis class at Olin College of
% Engineering
%% Setup
clear all
close all

tic %used for timing
n1 = 3;
n2 = 38; %used for first run, we're gonna rewrite it later in for loop

%%%% Load and format data %%%%

load('classdata.mat')
allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));
trainFaces = allFaces(:,1:2:end);   %pulls odds
testFaces = allFaces(:,2:2:end);    %pull evens
trainFaces_norm = (trainFaces-mean(trainFaces)).*(1/sqrt(size(trainFaces,1)));

%%%% Perform singular value decomposition on pull weights %%%%
[U,~,~] = svd(trainFaces_norm,'econ');
weightEig = U(:,n1:n2);
compTrain = weightEig\trainFaces; %projection
trident = sum(trainFaces,2)/size(trainFaces,2);
w = weightEig'*(testFaces-trident);

%%%% Gives results of current n1:n2 values %%%%
pos = 0;
for inde = 1:size(testFaces,2)    
    diff = sum((compTrain-w(:,inde)).^2,1);
    ind = find(diff==min(diff));
    if inde == ind
        pos = pos + 1;
    end
end
res = 100*(pos/size(compTrain,2))
toc
%% Trial One
tic
pers = randi(size(testFaces,2))
diff = sum((compTrain-w(:,pers)).^2,1);
ind = find(diff==min(diff));
toc

%%%% Plots random person along with the guessed face %%%%
figure
subplot(1,2,1)
imagesc(grayfaces(:,:,pers*2));colormap('gray');
subplot(1,2,2)
imagesc(grayfaces(:,:,ind*2-1));colormap('gray');
set(gcf,'Position', [400, 500, 1000, 500])


%% Optimization
res = zeros(1,40);
for n  = 1:40 %number of vectors we want to consider
%     tic; %for timing. 
    weightEig = U(:,1:n);
    compTrain = weightEig\trainFaces;

    trident = sum(trainFaces,2)/size(trainFaces,2);
    w = weightEig'*(testFaces-trident); %
    w = weightEig\testFaces;
    pos = 0; %keeps track of matches
    neg = 0; %keeps track of mismatches
    rt = zeros(1,size(compTrain,2));
    for x = 1:size(compTrain,2) %number of faces we have, in this case 66
        tic
        if test(x,compTrain,w) == 1;
            pos = pos + 1;
        else
            neg = neg + 1;
        end
        rt(1,x) = toc;
    end
    res(2,n) = mean(rt); %toc returns the time between last tic and toc
    res(1,n) = 1*(pos/(pos+neg)); %stores accuracy for each n
end

%%%% Plots results %%%%
close all
figure 
hold on
xlabel('Number of eigenvectors')
yyaxis left
ylim([0 1])
ylabel('Accuracy')
plot(res(1,:),'-o')
yyaxis right
ylabel('Runtime (sec)')
plot(res(2,:),'-o')

max(res(1,:))
ind = find(res(1,:)==max(res(1,:)))
res(2,ind)
%% Functions
function bool = test(inde,train,weight)
    diff = sum((train-weight(:,inde)).^2,1); %euclidian distance
    ind = find(diff==min(diff));%finds index
    if inde == ind
        bool = 1;
    else
        bool = 0;
    end
end

function res = test1(ran,train,weight)
    bools = zeros(1,size(train,2));
    for inde = 1:ran    
        diff = sum((train-weight(:,inde)).^2,1);
        ind = find(diff==min(diff));
        if inde == ind
            bools(inde) = 1;
        end
    end
    res = 100*(sum(bools)/size(train,2));
end