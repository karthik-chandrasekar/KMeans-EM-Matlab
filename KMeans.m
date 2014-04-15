function finalSSE = KMeans(clusters, r, filename)

%filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);
centroidMatrix = zeros(clusters, Mcol);
AllInstanceLabels = zeros(Mrow, r);
AllsseMap = zeros(r,1);
 
%The following entire procedure is repeadted r times with different intial
%seeds everytime

for rCount = 1:r
    
    %The following procedure has to be repeated for r times

    %Initially picking k seeds randomly
    for k=1:clusters
        centroidMatrix(k,:) = M(randi([1, Mrow], 1, 1), :);
    end

    convergenceCount = 0;
    maxDiff = 1;

    while maxDiff > 0.001
    
        convergenceCount = convergenceCount +1;
    
        %CentroidMap which maps the instance to a cluster
        centroidMap = zeros(Mrow,1);

        %Updating centroid computing data structures
        centroidInstanceCount = zeros(clusters, 1);
        newCentroid = zeros(clusters, Mcol);

        %Cluster instances
        for index = 1:Mrow
            inputVect = M(index, :);
            minVal = 1000000;
            minIndex = 0;
        
            for kVal = 1:clusters
                centroid = centroidMatrix(kVal, :); 
                %Gives euclidean distance between two vectors
                dist = norm(centroid - inputVect);
                if dist == min([dist minVal])
                    minVal = dist;
                    minIndex = kVal;
                end
            end
            centroidMap(index) = minIndex;
    
            %Concurrently updating next centroid computation values
            centroidInstanceCount(minIndex) = centroidInstanceCount(minIndex)+1;
            newCentroid(minIndex, :) = newCentroid(minIndex, :) + M(index, :);
        end
        %One iteration is over

        %Compute new centroid
        for index = 1:clusters
            newCentroid(index, :) = newCentroid(index, :)/centroidInstanceCount(index); 
        end
    
        disp('Old centroid  - ');
        disp(centroidMatrix);

        disp('New centroid - ');
        disp(newCentroid);

        diffCentroid = abs(centroidMatrix - newCentroid);
        disp('Diff centroid');
        disp(diffCentroid);

        disp('Max difference ');
        maxDiff = max(max(diffCentroid));
        disp(maxDiff);
    
        %copy new centroid to the old centroid
        oldCentroidMatrix = centroidMatrix;
        centroidMatrix = newCentroid;
        disp(convergenceCount);
    end    

    %Finding SSE - Sum of squared errors of prediction
    sseMap = zeros(clusters, 1);
    disp('SSE Map');
    
    for index = 1:Mrow
        inputVect = M(index, :);
        clusterIndex = centroidMap(index);
        sseMap(clusterIndex) = sseMap(clusterIndex) + norm(inputVect - oldCentroidMatrix(clusterIndex));
    end

    disp('SSE Map');
    disp(sseMap);

    disp('Sum SSE map');
    disp(sum(sseMap));
    
    disp('rCount');
    disp(rCount);
    
   AllInstanceLabels(:,rCount) = centroidMap; 
   AllsseMap(rCount) = sum(sseMap);
   
end

disp('All instance labels');
%disp(AllInstanceLabels);

disp('AllsseMap');
disp(AllsseMap);

%Find the cluster with the minimum SSE and select it
minSSE = 10000;

for k = 1:r
    if(minSSE > AllsseMap(k))
     minSSE = AllsseMap(k);
     %minSSEIndex = k;
    end    
end

%finalMap = AllInstanceLabels(:,minSSEIndex);
finalSSE = minSSE;
