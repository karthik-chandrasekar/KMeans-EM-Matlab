function [] = GaussianMixtureEM(clusters, r, filename)

%filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);

%The following entire procedure is repeadted r times with different intial
%seeds everytime

for rCount = 1:1

    %Randomly pick initial mean and variance 
    y = datasample(1:Mrow,clusters,'Replace',false);

    mean = zeros(clusters, Mcol);
    covari = zeros(clusters, Mcol);
    phi = [0.3, 0.1, 0.6]
    resp = zeros(clusters, Mrow);
    newLogLikeli = zeros(Mrow,1);
    
    for i=1:clusters
        mean(i,:) = M(y(i),:);
        covari(i,:) = i*var(M(y(i),:));
    end
      
    convergenceCount = 0;
    maxDiff = 1;
    %Every run is iterated till the log likelihood is converged. 
    while maxDiff > 0.001
    %for index = 1:200
    
        convergenceCount = convergenceCount +1;
        
        %Following procedure has to be repeated clusters times
               
        totalPhi = zeros(Mrow, 1);
        logLikeli = zeros(Mrow,1);

        for i = 1:clusters
        
         %E-Step            
            gDist = mvnpdf(M, mean(i,:), covari(i,:));
            logLikeli(:,1) = logLikeli(:, 1) + log(phi(i)*gDist);
            totalPhi = totalPhi + phi(i)*gDist;
        end  
        %Finding responsibility of each component 
        for i = 1:clusters
         
            resp(i,:) = (phi(i) * gDist) ./ totalPhi;
        end
        
        
        %M-Step - Find new mean and variance
        
        %Updating new mean
        meanNum  = 0;
        for j=1:Mrow
                meanNum = meanNum + ((1-resp(i,j)).*M(j,:));
        end
        mean(i,:) = meanNum/sum(1-resp(i,:));
     
        
        %Updating new sigma
        covarNum = 0;
        for j = 1:Mrow
            covarNum = covarNum + ((1-resp(i,j)).*((M(j,:) - mean(i,:)).^2));
        end
        covari(i,:) = covarNum/sum(1-resp(i,:));
        
        %Updating new phi
        phi(i) = sum(resp(i,:)/Mrow);
                
        %Checking for convergence
        diffLogLikeli = abs(logLikeli) - abs(newLogLikeli);
        maxDiff = abs(max(diffLogLikeli));
        
        maxDiff
        %Updating log likelihood
        newLogLikeli = logLikeli;
        
    end   
end
figure
plot(logLikeli)
xlabel('Iteration');
ylabel('Observed Data Log-likelihood');
grid minor

