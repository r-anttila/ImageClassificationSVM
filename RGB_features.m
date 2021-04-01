function rgb = RGB_features(data)
    rgb = zeros(3,length(data));
    
    for ii=1:length(data)
        pic = data{ii};
        rgb(:,ii) = [mean(pic(:,:,1),'all');mean(pic(:,:,2),'all');mean(pic(:,:,3),'all')]; 
    end

end