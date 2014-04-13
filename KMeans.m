filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);
clusters = 3;
centroidMatrix = zeros(clusters, Mcol);

%Initially picking seeds randomly
for k=1:clusters
    centroidMatrix(k,:) = M(randi([1, Mrow], 1,1), :);
end
disp(centroidMatrix);

%The following procedure has to be repeated for r times

convergenceCount = 0;
maxDiff = 1

while maxDiff > 0.001
    
    convergenceCount = convergenceCount +1;
    %CentroidMap which maps the instance to a cluster
    centroidMap = zeros(Mrow,1);

    %Updating centroid
    centroidInstanceCount = zeros(clusters, 1);
    newCentroid = zeros(clusters, Mcol);

    %Single iteration to cluster instances
    for index = 1:Mrow
        inputVect = M(index, :);
        minVal = 1000000;
        mindIndex = 0;
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
    
        %Concurrently updating next centroid support values
        centroidInstanceCount(minIndex) = centroidInstanceCount(minIndex)+1;
        newCentroid(minIndex, :) = newCentroid(minIndex, :) + M(index, :);
    end

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
    centroidMatrix = newCentroid;
    disp(convergenceCount);
end    

