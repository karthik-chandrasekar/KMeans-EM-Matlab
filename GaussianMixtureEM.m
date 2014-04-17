function [] = GaussianMixtureEM(clusters, r, filename, seedSelection)

delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);

maxLogLikeliVal = -100000000;
finalLogLikeli = zeros(1, Mrow);

%The following entire procedure is repeadted r times 

for rCount = 1:r

    y = datasample(1:Mrow,clusters,'Replace',false);

    gMean = zeros(clusters, Mcol);
    gCovar = zeros(clusters, Mcol);
    gPhi = rand(clusters, 1)';
    
    gGamma = zeros(clusters, Mrow);
    newLogLikeli = zeros(Mrow,1);
    sumLogLikeli = zeros(1,Mrow);   
 
    if (seedSelection == 'R')
        %Initial values are selected randomly
        for i=1:clusters
            gMean(i,:) = M(y(i),:);
            gCovar(i,:) = i*var(M(y(i),:));
        end

    else     
        %Centroids returned by K-Means will be used as initial values
        gMean = abs(KMeans(clusters, r, filename, 'N'));
        gCovar = 2 * gMean;

    end
    
    convergenceCount = 0;
    maxDiff = 1;
    
    
    while maxDiff > 0.00001    
        convergenceCount = convergenceCount +1;
        
        %Following procedure has to be repeated number of cluster times
               
        totalPhi = zeros(Mrow, 1);
        logLikeli = zeros(Mrow,1);

        for i = 1:clusters
        
         %E-Step - Find log likeli hood value        
            gDist = mvnpdf(M, gMean(i,:), gCovar(i,:));
            logLikeli(:,1) = logLikeli(:, 1) + log(gPhi(i)*gDist);
            totalPhi = totalPhi + gPhi(i)*gDist;
        end  
        
        %Finding gamma for  each component 
        for i = 1:clusters
            gGamma(i,:) = (gPhi(i) * gDist) ./ totalPhi;
        end
        
        %M-Step - Find new mean, covariance and phi values
     
        for i = 1:clusters
            %Updating new gMean
            gMeanNum  = 0;
            for j=1:Mrow
                gMeanNum = gMeanNum + ((gGamma(i,j)).*M(j,:));
            end
            gMean(i,:) = gMeanNum/sum(gGamma(i,:));

            %Updating new sigma
            covarNum = 0;
            for j = 1:Mrow
                covarNum = covarNum + ((gGamma(i,j)).*((M(j,:) - gMean(i,:)).^2));
            end
            gCovar(i,:) = abs(covarNum/sum(gGamma(i,:)));

            %Updating new gPhi
            gPhi(i) = sum(gGamma(i,:)/Mrow);
            
        end
            
        %Checking for convergence
        diffLogLikeli = abs(logLikeli - newLogLikeli);
        maxDiff = max(diffLogLikeli);
        
        %Updating log likelihood
        newLogLikeli = logLikeli;
        sumLogLikeli(convergenceCount) = sum(logLikeli);       
 
    end   
    
    %Picking the maximum log likeli values
    if (maxLogLikeliVal<sum(sumLogLikeli))
        finalLogLikeli = sumLogLikeli;
    end
    
end

%Plotting figure
finalLogLikeli = finalLogLikeli(finalLogLikeli~=0)
figure
plot(finalLogLikeli)
xlabel('Iteration');
ylabel('Observed Data Log-likelihood');
grid minor
