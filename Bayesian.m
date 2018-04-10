%% Bayesian
% Bayesian Implentation of Facial Recognition
% authors@ Corey Cochran-Lepiz & Hwei-Shin Harriman
% For Quantitative Engineering Analysis class at Olin College of
% Engineering
%% Load and format data
clear all
close all

%%%% Load and 'reshape' data from grid to vector %%%%
load('classdata_full.mat')
allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));

%%%% Zeros for optimization %%%%
trainFaces = zeros(size(allFaces,1),86);
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
%% Mean-centering of data

omegaFace = odds-evens;                 %DELTA
omega = eigWeights(omegaFace,1,32);     %Defines face space
og = omega\omegaFace;                   %project DELTA unto face space
C = cov(og');                           %Sigma

oface = (1/256).*(omegaFace-mean(omegaFace));
me = mean(omega\oface,2);               %maen-center then find mean of function.

tFaces = (tFaces-mean(tFaces)).*(1/256);%mean-center
tFaces = omega\tFaces;                  %projection

den = 1./((2*pi)^(size(omega,2)/2)*norm(C)^.5);%denominator constant
allFaces = (allFaces-mean(allFaces)).*(1/256);  %mean-center
allFaces = omega\allFaces;

%% Trial One
% tic and toc meant for timing the duration of what runs in between
% probs is a function defines below.

tic
pers = 7;
probs = prob(allFaces(:,pers),tFaces,C,me,den);
toc

%% Class and Data
load('classdata_demo.mat')              %loads demo day class data
allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));
tFaces = allFaces(:,1:2:end);           %split into two groups
allFaces = allFaces(:,2:2:end);
allFaces = (allFaces-mean(allFaces)).*(1/256);
tFaces = (tFaces-mean(tFaces)).*(1/256);

%5:32
%6:25

res = zeros(2,40);
data = zeros(1,size(allFaces,2));
OMEGA = eigWeights(omegaFace,1,size(omegaFace,2)); 
ind = 1;
inde = 1;

for m = 1 %unused for these purposes
    for n = 1:40
        % Reformat and mean-center data
        allFaces = reshape(grayfaces,size(grayfaces,1)*size(grayfaces,2),size(grayfaces,3));
        tFaces = allFaces(:,1:2:end);
        allFaces = allFaces(:,2:2:end);
        ind = 1;
        data = zeros(1,size(allFaces,2));
        % Pull from n amount of eigenvectors
        omega = OMEGA(:,m:n);
        og = omega\omegaFace;
        C = cov(og');
        tFaces = omega\tFaces;
        allFaces = omega\allFaces;
        oface = (1/256).*(omegaFace-mean(omegaFace));
        me = mean(omega\oface,2);
        tic 
        for pers = 1:size(allFaces,2)
            
            tic
            probs = prob(allFaces(:,pers),tFaces,C,me,den);
            maxval = find(probs(1,:)==max(probs(1,:)));
            if pers  == maxval
                data(ind) = 1;
    %             pplot(grayfaces(:,:,pers*2-1),grayfaces(:,:,maxval*2),.5)
            else
%                 pplot(grayfaces(:,:,pers*2-1),grayfaces(:,:,maxval*2),.5)
                data(ind) = 0;
            end
            ind = ind + 1;
        end
        res(1,inde) = toc;
        res(2,inde) = (sum(data)/size(allFaces,2));
        inde = inde + 1;
    end
end

% q = 100*(sum(data)/size(data,2))

% Graphs results
max(res(2,:))
find(res(2,:)==max(res(2,:))) 
figure 
hold on
xlabel('Number of eigenvectors')
yyaxis left
ylim([0 1])
ylabel('Accuracy')
plot(res(2,:),'-o')
yyaxis right
ylabel('Runtime (sec)')
plot(res(1,:),'-o')


%% Fun
% % Used to visualize faces and check data
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

%pretty plot! Used to plot two faces against each other in a neat fashion
function pplot(face1,face2,t)
    subplot(1,2,1)
    imagesc(face1);colormap('gray');
    subplot(1,2,2)
    imagesc(face2);colormap('gray');
    set(gcf,'Position', [400, 500, 1000, 500])
    pause(t)
end

%did not ultimately use this function, meant for whitening
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
function pe = prob(tproj,proj,C,me,den) %tface is face we're examining
   pe = zeros(1,size(proj,2));
   for n = 1:size(proj,2)
       tic
        delta = (tproj-proj(:,n));
       pe(1,n) = den*exp(-.5*(delta-me)'*inv(C)*(delta-me));
   end
   
end

%return n1:n2 of eigenvectors pulled from covariance of face set
function w = eigWeights(faceSet,n1,n2)
    
    meanLess = faceSet - mean(faceSet);
    norm = meanLess.*(1/sqrt(size(meanLess,1)));

    [U,~,~] = svd(norm,'econ');
    w = U(:,n1:n2); %returns weight vectors
end

%Similar to one right above but with eigenvalues instead.
function v = eigValues(faceSet,n1,n2)
    meanLess = faceSet - mean(faceSet);
    norm = meanLess.*(1/sqrt(size(meanLess,1)));

    [~,V,~] = svd(norm,'econ');
    v = (V(1:n2,n1:n2)); %returns weight vectors

end