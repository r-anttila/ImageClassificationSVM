%% Load data
N = 28*28; % The number of features
P = 5000; % The number of training samples
PTest = 1250;  % The number of test samples

load('MNISTData.mat');

%Reshape in such a way that each row contains one sample
XTrain = reshape(XTrain, [N,P]); XTrain = XTrain';
XTest = reshape(XTest, [N,PTest]); XTest = XTest';


%% Display example images
figure()
subplot(221)
imshow(reshape(XTrain(200,:),[28,28]));
title("Example images from class 3")

subplot(222)
imshow(reshape(XTrain(2800,:),[28,28]));
title("Example images from class 8")


subplot(223)
imshow(reshape(XTrain(140,:),[28,28]));

subplot(224)
imshow(reshape(XTrain(3400,:),[28,28]));
%% SVM with no kernel
gamma = 1;

fprintf("Solving the Linear SVM...\n");
cvx_begin quiet
    variable a(N);
    variable u(P);
    variable b;
    
    minimize (1/2*a'*a+gamma*norm(u,1));
    subject to 
        YTrain.*(XTrain*a+b)>=1-u;
        u>=0;
cvx_end
pred = @(z) sign(z*a+b);
fprintf("DONE\n\n");

%% Accuracy

miss = nnz(sign(XTrain*a+b)-YTrain);

fprintf("The number of missclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n", P-miss);
fprintf("Accuracy: %f\n\n", (P-miss)/P);

%% Validation

miss =nnz(sign(XTest*a+b)-YTest);

fprintf("The number of misclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n", PTest-miss);
fprintf("Accuracy: %f\n", (PTest-miss)/PTest);


%% SVM dual with RBF kernel
sigma = 4;

% The RBF (or gaussian) kernel
RBF = @(x1,x2) exp(-vecnorm((x1-x2),2,2).^2/(2*sigma^2));

fprintf("Constructing the RBF Kernel matrix...\n")
% Construct the kernel matrix
K = kerMat(RBF,XTrain);

H = YTrain.*K.*YTrain';
fprintf("DONE\n\n");

%% Solve the dual SVM with RBF kernel problem
fprintf("Solving the Non-Linear SVM...\n");
gamma = 1;

f = -ones(1,P);
z = zeros(1,P);

c = quadprog(H,f,[],[],YTrain',0,z,gamma*-f);

b_ind = find(c>0.001,1);
bd = YTrain(b_ind)-kerSum(RBF,c,YTrain,XTrain,XTrain(b_ind,:));

% Function handle to make predictions
pred = @(z) sign(kerSum(RBF,c,YTrain,XTrain,z)+bd);
fprintf("DONE\n\n");


%% Accuracy

TrainPred = pred(XTrain);
miss = nnz(TrainPred-YTrain);
fprintf("The number of misclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n", P-miss);
fprintf("Accuracy: %f\n\n", (P-miss)/P);

%% Validation

TestPred = pred(XTest);
miss = nnz(TestPred-YTest);
fprintf("The number of misclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n", PTest-miss);
fprintf("Accuracy: %f\n\n", (PTest-miss)/PTest);

%% Show missclassifications
preds = TestPred;
predictedClass1Inds = find(preds==-1);
predictedClass2Inds = find(preds==1);

misses = find(TestPred~=YTest);

class1Misses = misses(ismember(misses,predictedClass1Inds));
class2Misses = misses(ismember(misses,predictedClass2Inds));

figure()
subplot(221)
imshow(reshape(XTest(class1Misses(2),:),[28,28]));
title("Predicted: 3, True: 8")

subplot(223)
imshow(reshape(XTest(class1Misses(3),:),[28,28]));

subplot(222)
imshow(reshape(XTest(class2Misses(2),:),[28,28]));
title("Predicted: 8, True: 3")

subplot(224)
imshow(reshape(XTest(class2Misses(1),:),[28,28]));