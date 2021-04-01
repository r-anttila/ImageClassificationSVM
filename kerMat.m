function K = kerMat(func,X)
    % Construct the Kernel matrix
    P = size(X,1);
    K = zeros(P);
    for ii = 1:P
        for jj = 1:P
           K(ii,jj) = func(X(ii,:),X(jj,:));
        end
    end
end