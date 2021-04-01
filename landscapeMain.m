%% Reading the data
load('LandscapeData.mat');

vals = [valForest;valBuildings]; % Combine to full dataset

%% Feature extraction
% RGB values

rgb_forest = RGB_features(forest);
rgb_buildings = RGB_features(buildings);

rgb_valForest = RGB_features(valForest);
rgb_valBuildings = RGB_features(valBuildings);

% Connected edges

edge_forest = edge_features(forest);
edge_buildings = edge_features(buildings);

edge_valForest = edge_features(valForest);
edge_valBuildings = edge_features(valBuildings);


%% Visualization

scatter3(rgb_forest(1,:),rgb_forest(2,:),rgb_forest(3,:),'filled'), hold on
scatter3(rgb_buildings(1,:),rgb_buildings(2,:),rgb_buildings(3,:));
legend("Forest", "Buildings");

%% SVM classification with no kernel

% Combine all the images into the training set

forestFeatures = [rgb_forest;edge_forest];
buildingFeatures = [rgb_buildings;edge_buildings];

X = [forestFeatures,buildingFeatures]';

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(forest),1);ones(length(buildings),1)];

% The number of samples
P = size(X,1);

% The number of weights
N = size(X,2);
gamma = 1e-2;

cvx_begin quiet
    variable a(N);
    variable u(P);
    variable b;
    
    minimize (1/2*a'*a+gamma*norm(u,1));
    subject to 
        Y.*(X*a+b)>=1-u;
        u>=0;
cvx_end
pred = @(z) sign(z*a+b);

%% Accuracy

miss = nnz(sign(X*a+b)-Y);

fprintf("The number of missclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n", P-miss);
fprintf("Accuracy: %f\n", (P-miss)/P);


%% Validation
forestVals = [rgb_valForest;edge_valForest];
buildingVals = [rgb_valBuildings;edge_valBuildings];

Xval = [forestVals,buildingVals]';

Yval = [-1*ones(length(valForest),1);ones(length(valBuildings),1)];

Pp = length(Xval);

miss =nnz(sign(Xval*a+b)-Yval);
fprintf("The number of misclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n", Pp-miss);
fprintf("Accuracy: %f\n", (Pp-miss)/Pp);


%% SVM dual representation with no kernel

% Combine all the images into the training set

forestFeatures = [rgb_forest;edge_forest];
buildingFeatures = [rgb_buildings;edge_buildings];

X = [forestFeatures,buildingFeatures]';

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(forest),1);ones(length(buildings),1)];

D = Y'.*(X*X').*Y;

% The number of samples
P = size(X,1);

%The number of weights
N = size(X,2);
gamma = 1e-2;

f = -ones(1,length(Y));
z = zeros(1,length(Y));

% Solve the quadratic SVM problem
c = quadprog(D,f,[],[],Y',0,z,gamma*-f);

ad = sum(c.*Y.*X)';
b_ind = find(c>0.001,1);
bd = Y(b_ind)-X(b_ind,:)*ad;

% Function for predicting the class of a sample
pred = @(z) sign(z*ad+bd);

%% Accuracy

fprintf("The number of misclassifications: %i\n", nnz(sign(X*ad+bd)-Y));
fprintf("Accuracy: %f\n", (P-nnz(sign(X*ad+bd)-Y))/P);

%% Validation
forestVals = [rgb_valForest;edge_valForest];
buildingVals = [rgb_valBuildings;edge_valBuildings];

Xval = [forestVals,buildingVals]';

Yval = [-1*ones(length(valForest),1);ones(length(valBuildings),1)];

fprintf("The number of misclassifications: %i\n", nnz(sign(Xval*ad+bd)-Yval));

%% SVM dual with RBF kernel
% Combine all the images into the training set

forestFeatures = [rgb_forest;edge_forest];
buildingFeatures = [rgb_buildings;edge_buildings];

X = [forestFeatures,buildingFeatures]';

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(forest),1);ones(length(buildings),1)];

sigma = 125;
gamma = 10;

% The RBF (or gaussian) kernel
RBF = @(x1,x2) exp(-vecnorm((x1-x2),2,2).^2/(2*sigma^2));

% Construct the kernel matrix
K = kerMat(RBF,X);

H = Y.*K.*Y';

% The number of samples
P = size(X,1);

%The number of weights
N = size(X,2);

f = -ones(1,length(Y));
z = zeros(1,length(Y));

c = quadprog(H,f,[],[],Y',0,z,gamma*-f);

b_ind = find(c>0.001,1);
bd = Y(b_ind)-kerSum(RBF,c,Y,X,X(b_ind,:));

% Function handle to make predictions
pred = @(z) sign(kerSum(RBF,c,Y,X,z)+bd);

%% Accuracy

miss = nnz(pred(X)-Y);
fprintf("The number of misclassifications: %i\n", miss);
fprintf("Correctly classified: %i\n",P-miss);
fprintf("Accuracy: %f\n", (P-miss)/P);

%% Validation
forestVals = [rgb_valForest;edge_valForest];
buildingVals = [rgb_valBuildings;edge_valBuildings];

Xval = [forestVals,buildingVals]';

Yval = [-1*ones(length(valForest),1);ones(length(valBuildings),1)];

miss = nnz(pred(Xval)-Yval);
fprintf("The number of misclassifications: %i\n", miss);
fprintf("Accuracy: %f\n", (Pp-miss)/Pp);

%% Show missclassifications
preds = pred(Xval);
predictedForestInds = find(preds==-1);
predictedBuildingInds = find(preds==1);

misses = find(pred(Xval)~=Yval);

missedForest = ismember(misses,predictedForestInds);
missedBuildings = ismember(misses,predictedBuildingInds);

buildingMisses = misses(missedBuildings);
forestMisses = misses(missedForest);

figure()
subplot(221)
imshow(vals{buildingMisses(13)});
title("Predicted: Buildings, True: Forest")

subplot(223)
imshow(vals{buildingMisses(6)});

subplot(222)
imshow(vals{forestMisses(7)});
title("Predicted: Forest, True: Buildings")

subplot(224)
imshow(vals{forestMisses(3)});

