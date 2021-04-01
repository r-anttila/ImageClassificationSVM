function edges = edge_features(data)

    edges = zeros(2,length(data));

    for ii=1:length(data)
        pic = data{ii};
        edgev = edge(rgb2gray(pic),'Sobel','vertical');
        edgeh = edge(rgb2gray(pic),'Sobel','horizontal');
        edges(1,ii) = bwconncomp(edgev).NumObjects; 
        edges(2,ii) = bwconncomp(edgeh).NumObjects; 
    end
end