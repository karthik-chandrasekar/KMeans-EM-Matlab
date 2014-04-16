fileID = fopen('data.txt','r');
    formatSpec = '%f %f';
    sizeA = [1600 2];
    data = fscanf(fileID,formatSpec,sizeA);
    %% Step 1 : Initial guess
    K=2;
    [rows,dim] = size(data);
    temp=randperm(length(data),K);
    
    meucap = zeros(K,dim);
    sigmacap = zeros(K,dim);
    meucap(1,:)=data(temp(1),:);
    meucap(2,:)=data(temp(2),:);
    sigmacap(1,:)=var(data);
    sigmacap(2,:)=var(data);
    piecap = [0.3,0.7];
    for i = 1:20
        for z=1:K
            
            %% Step 2 : Expectation Step; computes the responsibilities
            %Qq1=gauss_dist(data,meucap1(i,:),sigmacap1)
            Qq= mvnpdf(data,meucap(z,:),sigmacap(z,:))
            %Qq2=gauss_dist(data,meucap2(i, :),sigmacap2)
            log_likelihood(i)=sum(log(((1-piecap(z))*Qq) + (piecap(z)*Qq)));

            responsibilities(z,:)=(piecap(z)*Qq)./((piecap(z)*Qq)+(piecap(z)*Qq))
            %responsibilities(2,:)=(piecap(i)*Qq2)./(((1-piecap(i))*Qq1)+(piecap(i)*Qq2));
            %% Step 3 : Maximization Step; compute the weighted means and variances 
        
            meansum = 0;
            sigmaSum = 0;
            for j=1:rows
                meanssum = meansum + ((1-responsibilities(z,j)).*data(j,:));
            end
            meucap(z,:)= meansum/sum(1-responsibilities(z,:));
            for j=1:rows
                sigmaSum = sigmaSum + ((1-responsibilities(z,j)).*((data(j,:) - meucap(z,:)).^2));          
            end
            sigmacap(z,:)=sigmaSum/sum(1-responsibilities(z,:));
            piecap(z)=sum(responsibilities(z,:))/length(data);
        end
    end
    figure
    plot(log_likelihood)
    xlabel('Iteration');
    ylabel('Observed Data Log-likelihood');
    grid minor
