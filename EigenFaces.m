%% Setup
clear all
close all
tic

n = 28; %formality, we're gonna rewrite it later in for loop

load('classdata_demo.mat')
%allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));
allFaces = reshape(grayfaces,65536,122);
trainFaces = allFaces(:,1:2:end);   %pulls odds
testFaces = allFaces(:,2:2:end);    %pull evens

trainFaces_norm = (trainFaces-mean(trainFaces)).*(1/sqrt(size(trainFaces,1)));

[U,~,~] = svd(trainFaces_norm,'econ');
weightEig = U(:,1:n);
compTrain = weightEig\trainFaces;
trident = sum(trainFaces,2)/size(trainFaces,2);
w = weightEig'*(trainFaces-trident);

pos = 0;
for inde = 1:61    
    diff = sum((compTrain-w(:,inde)).^2,1);
    ind = find(diff==min(diff));
    if inde == ind
        pos = pos + 1;
    end
end
res = 100*(pos/size(compTrain,2))
toc 
tic
pers = randi(61)
diff = sum((compTrain-w(:,pers)).^2,1);
ind = find(diff==min(diff));
toc
subplot(1,2,1)
imagesc(grayfaces(:,:,pers*2));colormap('gray');
subplot(1,2,2)
imagesc(grayfaces(:,:,ind*2-1));colormap('gray');
set(gcf,'Position', [400, 500, 1000, 500])





% res = zeros(2,40);
% for n  = 1:40 %number of vectors we want to consider
%     tic %for timing. 
%     weightEig = U(:,1:n);
%     compTrain = weightEig\trainFaces;
%     trident = sum(trainFaces,2)/size(trainFaces,2);
%     w = weightEig'*(trainFaces-trident);
%     pos = 0; %keeps track of matches
%     neg = 0; %opposite of above
%     for x = 1:61 %number of faces we have, in this case 66
%         if test(x,compTrain,w) == 1;
%             pos = pos + 1;
%         else
%             neg = neg + 1;
%         end
%     end
%     res(2,n) = toc; %toc returns the time between last tic and toc
%     res(1,n) = 1*(pos/(pos+neg));
% end
% res
% figure
% plot(1:40,res(1,:)); hold on
% plot(1:40,res(2,:))

% ind = find(res(1,:)==max(res(1,:)))
% ind1 = find(res(2,:)==min(res(2,:)))
%% Functions

function res = test(ran,train,weight)
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