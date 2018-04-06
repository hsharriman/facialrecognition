%% Bayesian
%% Load and format data
clear all 
% tic

load('classdata_full.mat')
allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));

trainFaces = zeros(size(allFaces,1),65);
% tFaces = zeros(size(allFaces,1),65); %test faces
ind = 1; %bookeeping
ind1 = 1;
faceInfo = zeros(1,65);
for n = 1:222
    
   if y.picnum(n) == 3 %stores face indeces 2 and 3 for sample data
       trainFaces(:,ind*2-1) = allFaces(:,n-1);
       trainFaces(:,ind*2) = allFaces(:,n);
       ind = ind + 1;
   end
    %{
   if y.picnum(n) == 0
      trainFaces(:,ind*2-1) = allFaces(:,n);
      trainFaces(:,ind*2) = allFaces(:,n+1);
      ind = ind + 1;
   end
    %}
   if y.picnum(n) == 0 %stores face index of 0 for testing with 
       tFaces(:,ind1) = allFaces(:,n);
       faceInfo(ind1) = n;
       ind1 = ind1 + 1;
   end
end
odds = trainFaces(:,1:2:end);
evens = trainFaces(:,2:2:end);
%% Normalization of data

omegaFace = evens-odds; %DELTA
omega = eigWeights(omegaFace,32); %Defines face space
og = omega\omegaFace;
C = cov(og');%Sigma

oface = (1/256).*(omegaFace-mean(omegaFace));
me = mean(omega\oface,2); %normalize then find mena of function.

tFaces = (tFaces-mean(tFaces)).*(1/256);%normalize
tFaces = omega\tFaces;

den = 1./((2*pi)^(size(omega,2)/2)*norm(C)^.5);%denominator constand
allFaces = (allFaces-mean(allFaces)).*(1/256);%normalize
allFaces = omega\allFaces;
% allFaces = omega'*(allFaces-mean(omegaFace));

%% Main
tic
pers = 7;
probs = prob(allFaces(:,pers),tFaces,omega,C,me,den);
toc

data = zeros(1,65);
ind = 1;

%% Class and Data
load('classdata_demo.mat')
allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));
tFaces = allFaces(:,1:2:end);
allFaces = allFaces(:,2:2:end);
allFaces = (allFaces-mean(allFaces)).*(1/256);
tFaces = (tFaces-mean(tFaces)).*(1/256);


inde = 1;
res = zeros(1,40);
for n = 32
    allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));
    allFaces = (allFaces-mean(allFaces)).*(1/256);
    tFaces = allFaces(:,1:2:end);
    allFaces = allFaces(:,2:2:end);
    ind = 1;
    data = zeros(1,61);
    omega = eigWeights(omegaFace,n);
%     allFaces = omega'*(allFaces-mean(tFaces,2));
%     tFaces = omega\((tFaces-mean(tFaces)).*(1/256));
    og = omega\omegaFace;
    C = cov(og');
    tFaces = omega\tFaces;
    allFaces = omega\allFaces;
    oface = (1/256).*(omegaFace-mean(omegaFace));
    me = mean(omega\oface,2);
    tic 
    for pers = 1:61
        if true
            tic
            probs = prob(allFaces(:,pers),tFaces,omega,C,me,den);
            maxval = find(probs(1,:)==max(probs(1,:)));
            if pers  == maxval
                data(ind) = 1;
%                 pplot(grayfaces(:,:,pers*2-1),grayfaces(:,:,maxval*2),.5)
%             1
            else
                pplot(grayfaces(:,:,pers*2-1),grayfaces(:,:,maxval*2),.5)
                data(ind) = 0;
            end
            ind = ind + 1;
%             toc
        end
    end
    res(1,inde) = toc;
    res(2,inde) = (sum(data)/61);
    inde = inde + 1;
    
end

% q = 100*(sum(data)/61)

max(res(2,:))
find(res(2,:)==max(res(2,:))) 
figure 
hold on
xlabel('Number of eigenvetors')
yyaxis left
ylim([0 1])
ylabel('Accuracy')
plot(res(2,:),'-o')
yyaxis right
ylabel('Runtime (sec)')
plot(res(1,:),'-o')

tic
pers = 7;
probs = prob(allFaces(:,pers),tFaces,omega,C,me,den);
toc

%% Fun
% for n = 1:222
%     if y.picnum(n) == 0
%         subplot(1,2,1)
%         imagesc(grayfaces(:,:,n));colormap('gray');
%         subplot(1,2,2)
%         imagesc(grayfaces(:,:,n+1));colormap('gray');
%         set(gcf,'Position', [400, 500, 1000, 500])
%         pause(.2)
%     end
% end
% num

%% Functions

function pplot(face1,face2,t)
    subplot(1,2,1)
    imagesc(face1);colormap('gray');
    subplot(1,2,2)
    imagesc(face2);colormap('gray');
    set(gcf,'Position', [400, 500, 1000, 500])
    pause(t)
end

%did not ultimately use this function
function pe = prob1(tface,faces,C) %method 2 using white shifting
    
    pe = zeros(1,size(faces,2));
    den = 1./((2*pi)^(size(C,2)/2)*norm(C)^.5)
    for n = 1:size(faces,2)
       pe(1,n) = den*exp(-.5*(sum(faces(n)-tface))^2);
       if n == 8
           (sum(faces(n)-tface))^2
           (faces(n)-tface)
           pe(1,1:8)
       end
    end
end

%method 1 using normal distribution
function pe = prob(tproj,proj,omega,C,me,den) %tface is face we're examining
   pe = zeros(1,size(proj,2));
   for n = 1:size(proj,2)
       tic
       delta = (proj(:,n)-tproj);
       pe(1,n) = den*exp(-.5*(delta-me)'*inv(C)*(delta-me));
   end
   
end

function w = eigWeights(faceSet,n)
    
    meanLess = faceSet - mean(faceSet);
    norm = meanLess.*(1/sqrt(size(meanLess,1)));

    [U,~,~] = svd(norm,'econ');
    w = U(:,1:n); %returns weight vectors
end

function v = eigValues(faceSet,n)
    meanLess = faceSet - mean(faceSet);
    norm = meanLess.*(1/sqrt(size(meanLess,1)));

    [~,V,~] = svd(norm,'econ');
    v = (V(1:n,1:n)); %returns weight vectors

end