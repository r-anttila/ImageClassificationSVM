function s = kerSum(func,c,Y,X,z)
    P = length(X);
    s=0;
    for ii=1:P
        s = s + c(ii)*Y(ii)*func(X(ii,:),z);
    end
end