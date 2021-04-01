%% Toy demonstration primal hard

% Combine all the images into the training set

class1x = [2,3,4,5,5,5,6]';
class1y = [6,6,4,6,5,4,3]';

class2x = [1,2,2,3,4,4]';
class2y = [3,4,2,3,2,1]';

X = [class1x,class1y;class2x,class2y];

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [ones(length(class1x),1);-ones(length(class2x),1)];

% The number of samples
P = size(X,1);

% The number of weights
N = size(X,2);
lambda = 1e-1;

cvx_begin
    variable a(N);
    variable b;

    minimize (1/2*a'*a);
    subject to 
        Y.*(X*a+b)>=1;
cvx_end

%% Toy demonstration primal soft

% Combine all the images into the training set

class1x = [1,3,4,5,5,5,6]';
class1y = [4,6,4,6,5,4,3]';

class2x = [1,2,2,3,4,4]';
class2y = [3,4,2,3,2,1]';

X = [class1x,class1y;class2x,class2y];

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(class1x),1);ones(length(class2x),1)];

% The number of samples
P = size(X,1);

% The number of weights
N = size(X,2);
lambda = 1e-1;

cvx_begin
    variable a(N);
    variable u(P);
    variable b;

    minimize (1/2*a'*a+lambda*norm(u,1));
    subject to 
        Y.*(X*a+b)>=1-u;
        u>=0;
cvx_end

%% Results

scatter(class1x,class1y), hold on;
scatter(class2x,class2y), hold on;

plot([0,-b/a(1)],[-b/a(2),0])

%% Toy demonstration dual hard

class1x = [2,3,4,5,5,5,6]';
class1y = [7,6,4,6,5,4,3]';

class2x = [1,2,2,3,4,4]';
class2y = [3,4,2,3,2,1]';

X = [class1x,class1y;class2x,class2y];

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(class1x),1);ones(length(class2x),1)];

D = Y'.*(X*X').*Y;

% The number of samples
P = size(X,1);

%The number of weights
N = size(X,2);
lambda = 0.1;

cvx_begin
    variable c(P);

    maximize (sum(c)-1/2*c'*D*c);
    subject to 
        c'*Y == 0;
        c>=0;
cvx_end

ad = sum(c.*Y.*X);
b_ind = find(c>0.1);
bd = Y(b_ind(2))-ad*X(b_ind(2),:)';

%% Results

scatter(class1x,class1y), hold on;
scatter(class2x,class2y), hold on;

plot([0,-bd/ad(1)],[-bd/ad(2),0])

%% Toy demonstration dual soft

class1x = [1,3,4,5,5,5,6]';
class1y = [4,6,4,6,5,4,3]';

class2x = [1,2,2,3,4,4]';
class2y = [3,4,2,3,2,1]';

X = [class1x,class1y;class2x,class2y];

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(class1x),1);ones(length(class2x),1)];

D = Y'.*(X*X').*Y;

% The number of samples
P = size(X,1);

%The number of weights
N = size(X,2);
lambda = 0.7;

cvx_begin quiet
    variable c(P);
    variable r(P);
    variable delta;

    maximize (sum(c)-(1/2*c'*D*c+delta^2/(2*lambda)*(1-1/2)));
    subject to 
        c'*Y == 0;
        c+r == delta;
        0<=c<=delta;
        r>=0;
cvx_end

ad = sum(c.*Y.*X);
b_ind = find(c>0.1);
bd = Y(b_ind(2))-ad*X(b_ind(2),:)';

%% Results

scatter(class1x,class1y), hold on;
scatter(class2x,class2y), hold on;

plot([0,-bd/ad(1)],[-bd/ad(2),0])

%% Toy demonstration non-linear dual hard

% Combine all the images into the training set

class1x = [2,3,3,3,4,4,5]';
class1y = [4,5,4,3,4,3,4]';

class2x = [1,2,3,5,5,5,6]';
class2y = [4,6,2,6,5,2,3]';

X = [class1x,class1y;class2x,class2y];

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(class1x),1);ones(length(class2x),1)];

% The number of samples
P = size(X,1);

%The number of weights
N = size(X,2);
gamma = 5;

sigma = 1;

% The RBF (or gaussian) kernel
RBF = @(x1,x2) exp(-vecnorm((x1-x2),2,2).^2/(2*sigma^2));

% Construct the kernel matrix
K = kerMat(RBF,X);

H = Y.*K.*Y';

cvx_begin quiet
    variable c(P) nonnegative;

    maximize (sum(c)-1/2*c'*H*c);
    subject to 
        c'*Y == 0;
        0 <= c <= gamma;
cvx_end

b_ind = find(c>0.001,1);
bd = Y(b_ind)-kerSum(RBF,c,Y,X,X(b_ind,:));

% Function handle to make predictions
pred = @(z) sign(kerSum(RBF,c,Y,X,z)+bd);

%% Visualization 
scatter(class1x,class1y,[],[0.9290 0.6940 0.1250],'filled'), hold on;
scatter(class2x,class2y,[],[0.4940 0.1840 0.5560],'filled'), hold on;
ylim([0,7]);
xlim([0,7]);
print("demo_nonlin.png","-dpng");
%% Results

x1range = 0:0.01:7;
x2range = 0:0.01:7;
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

prediction = pred(XGrid);


gscatter(xx1(:),xx2(:),prediction,[],[],[],'off'), hold on;
scatter(class1x,class1y,'filled'), hold on;
scatter(class2x,class2y,'filled'), hold on;
ylim([0,7]);
xlim([0,7]);
print("demo_classified", "-dpng")

%% Toy demonstration non-linear dual soft

% Combine all the images into the training set

class1x = [2,1,3,3,4,4,5]';
class1y = [4,6,4,6,4,3,4]';

class2x = [1,2,3,5,5,5,6]';
class2y = [4,6,2,6,5,2,3]';

X = [class1x,class1y;class2x,class2y];

% Y is the vector containing the labels for the training set, -1 for forest
% and +1 for buildings.
Y = [-1*ones(length(class1x),1);ones(length(class2x),1)];

% The number of samples
P = size(X,1);

%The number of weights
N = size(X,2);
lambda = 0.1;

sigma = 1;

% The RBF (or gaussian) kernel
RBF = @(x1,x2) exp(-vecnorm((x1-x2),2,2).^2/(2*sigma^2));

% Construct the kernel matrix
K = kerMat(RBF,X);

H = Y.*K.*Y';

cvx_begin quiet
    variable c(P) nonnegative;

    maximize (sum(c)-1/2*c'*H*c);
    subject to 
        c'*Y == 0;
        0 <= c ;
cvx_end

b_ind = find(c>0.001,1);
bd = Y(b_ind)-kerSum(RBF,c,Y,X,X(b_ind,:));

% Function handle to make predictions
pred = @(z) sign(kerSum(RBF,c,Y,X,z)+bd);

%% Results

x1range = 0:0.01:7;
x2range = 0:0.01:7;
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

prediction = pred(XGrid);


gscatter(xx1(:),xx2(:),prediction), hold on;
scatter(class1x,class1y), hold on;
scatter(class2x,class2y), hold on;


