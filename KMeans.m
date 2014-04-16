function finalSSE = KMeans(clusters, r, filename)

%filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);
centroidMatrix = zeros(clusters, Mcol);
AllInstanceLabels = zeros(Mrow, r);
AllsseMap = zeros(r,1);

curSSEMap = zeros(Mrow, 1);
minSSEMap = zeros(Mrow, 1);
minSSE = 1000000;

 
%The following entire procedure is repeadted r times with different intial
%seeds everytime

for rCount = 1:r
    
    %The following procedure has to be repeated for r times

    %Initially picking k seeds randomly
    
    y = datasample(1:Mrow,clusters,'Replace',false);
    for k=1:clusters
        centroidMatrix(k,:) = M(y(k), :);
    end

    convergenceCount = 0;
    maxDiff = 1;

    while maxDiff > 0.001
    
        convergenceCount = convergenceCount +1;
    
        %CentroidMap which maps the instance to a cluster
        centroidMap = zeros(Mrow,1);

        %Updating next centroid values
        centroidInstanceCount = zeros(clusters, 1);
        newCentroid = zeros(clusters, Mcol);

        %Iterating over the input data points
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
        
        %Find the SSE after every iteration of single K Mean run
        
        curSSE = 0;
        for index = 1:Mrow
            inputVect = M(index, :);
            clusterIndex = centroidMap(index);
            instanceNorm =  norm(inputVect - oldCentroidMatrix(clusterIndex));
            curSSE = curSSE + instanceNorm * instanceNorm;
        end
        
        curSSEMap(convergenceCount) = curSSE;
       
    end    

    %Finding SSE - Sum of squared errors of prediction after every run
    sseMap = zeros(clusters, 1);
    disp('SSE Map');
    
    curSSE = 0;
    for index = 1:Mrow
        inputVect = M(index, :);
        clusterIndex = centroidMap(index);
        instanceNorm =  norm(inputVect - oldCentroidMatrix(clusterIndex));
        curSSE = curSSE + instanceNorm * instanceNorm;
    end
    
    disp('rCount');
    disp(rCount);
    
    AllInstanceLabels(:,rCount) = centroidMap; 
    AllsseMap(rCount) = sum(sseMap);
    
    %Manintaining only the maximum sse results at any point of time
    if(curSSE < minSSE)
        minSSE = curSSE;        
        minSSEMap = curSSEMap;          
    end
   
end

%finalMap = AllInstanceLabels(:,minSSEIndex);
finalSSE = minSSEMap;



%Plot the figure for SSE per iteration vs Iteration

nonZeroMinSSEMap = minSSEMap(minSSEMap~=0)';
[nzRow, nzCol] = size(nonZeroMinSSEMap);
iterationXaxis = 1:nzCol;

figure
plot(iterationXaxis , nonZeroMinSSEMap);
xlabel('Iteration');
ylabel('Observed SSE per iteration');
grid minor
