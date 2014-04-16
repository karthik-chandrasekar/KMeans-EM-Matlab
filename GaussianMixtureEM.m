function finalSSE = GMEM(clusters, r, filename)

%filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);


%The following entire procedure is repeadted r times with different intial
%seeds everytime

for rCount = 1:r

    %Randomly pick initial mean and variance 
    y = datasample(1:Mrow,clusters,'Replace',false);

    mean = zeros(clusters, Mcol);
    covari = zeros(clusters, Mcol);
    phi = zeros(1, clusters);
    logLikeli = zeros(Mrow,1);
    resp = zeros(clusters, Mcol);
    
    for i=1:clusters
        mean(i,:) = M(y(i),:);
        covari(i,:) = i*var(M(y(i),:));
    end
      
    %Every run is iterated till the log likelihood is converged. 
    while maxDiff > 0.001
    
        convergenceCount = convergenceCount +1;
        
        %Following procedure has to be repeated clusters times
               
        totalPhi = 0;
        for i = 1:clusters
        
         %E-Step
            gDist = mvnpdf(M, mean(i,:), covari(i,:));
            logLikeli(convergenceCount) = logLikeli(convergenceCount) + log(phi(i)*gDist);
            totalPhi = totalPhi + phi(i)*gDist;              
        end  
        
        %Finding responsibility of each component 
        for i = 1:clusters
            resp(i,:) = (phi(i) * gDist) ./ totalPhi;
        end
        
        %M-Step - Find new mean and variance
        
    end   
end


