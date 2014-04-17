function [] = GaussianMixtureEM(clusters, r, filename, seedSelection)

%filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);

%The following entire procedure is repeadted r times with different intial
%seeds everytime

for rCount = 1:3

    %Randomly pick initial gMean and covriance and phi.
    y = datasample(1:Mrow,clusters,'Replace',false);

    gMean = zeros(clusters, Mcol);
    gCovar = zeros(clusters, Mcol);
    gPhi = rand(clusters, 1)';
    
    gGamma = zeros(clusters, Mrow);
    newLogLikeli = zeros(Mrow,1);
    sumLogLikeli = zeros(1,Mrow);   
 
    if (seedSelection == 'R')
        %Random mean and variance selection
        for i=1:clusters
            gMean(i,:) = M(y(i),:);
            gCovar(i,:) = i*var(M(y(i),:));
        end

    else     
        %Pick best seeds from Kmeans
        gMean = abs(KMeans(clusters, r, filename, 'N'));
        gCovar = 2 * gMean;

    end
    
    convergenceCount = 0;
    maxDiff = 1;
    
    %Every run is iterated till the log likelihood is converged. 
    %while maxDiff > 0.001
    
    for iterCount = 1:20
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
            gGamma(i,:) = (gPhi(i) * gDist) ./ totalPhi
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
end
sumLogLikeli = sumLogLikeli(sumLogLikeli~=0)
figure
plot(sumLogLikeli)
xlabel('Iteration');
ylabel('Observed Data Log-likelihood');
grid minor
