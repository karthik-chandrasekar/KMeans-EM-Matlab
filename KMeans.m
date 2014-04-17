function finalSSE = KMeans(clusters, r, filename, figure)

%Load data from the given input file
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);
centroidMatrix = zeros(clusters, Mcol);

curSSEMap = zeros(Mrow, 1);
minSSEMap = zeros(Mrow, 1);
minSSE = 1000000;
 
%The following entire procedure is repeadted r times with different intial seeds everytime

for rCount = 1:r
       
    y = datasample(1:Mrow,clusters,'Replace',false);
    for k=1:clusters
        centroidMatrix(k,:) = M(y(k), :);
    end

    convergenceCount = 0;
    maxDiff = 1;

    while maxDiff > 0.001
    
        convergenceCount = convergenceCount +1;
    
        %CentroidMap which maps the instance to a cluster it belongs to.
        centroidMap = zeros(Mrow,1);

        %Updating next centroid values
        centroidInstanceCount = zeros(clusters, 1);
        newCentroid = zeros(clusters, Mcol);

        %Iterating over all the input data points
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

        %Compute new centroid
        for index = 1:clusters
            newCentroid(index, :) = newCentroid(index, :)/centroidInstanceCount(index); 
        end
   
        diffCentroid = abs(centroidMatrix - newCentroid);
        maxDiff = max(max(diffCentroid));
    
        %copy new centroid to the old centroid
        oldCentroidMatrix = centroidMatrix;
        centroidMatrix = newCentroid;
        
        %Find the SSE after every iteration of single K-Means run      
        curSSE = 0;
        for index = 1:Mrow
            inputVect = M(index, :);
            clusterIndex = centroidMap(index);
            instanceNorm =  norm(inputVect - oldCentroidMatrix(clusterIndex));
            curSSE = curSSE + instanceNorm * instanceNorm;
        end   
        curSSEMap(convergenceCount) = curSSE/Mrow;       
    end    

    %Finding SSE at the end of single K-Mean run        
    curSSE = 0;
    for index = 1:Mrow
        inputVect = M(index, :);
        clusterIndex = centroidMap(index);
        instanceNorm =  norm(inputVect - oldCentroidMatrix(clusterIndex));
        curSSE = curSSE + instanceNorm * instanceNorm;
    end
            
    %Manintaining only the maximum SSE results. 
    if(curSSE < minSSE)
        minSSE = curSSE;        
        minSSEMap = curSSEMap;          
    end
   
end


%Plot the figure for SSE per iteration vs Iteration number 

nonZeroMinSSEMap = minSSEMap(minSSEMap~=0)';
[nzRow, nzCol] = size(nonZeroMinSSEMap);
iterationXaxis = 1:nzCol;

if (figure == 'Y')
    figure
    plot(iterationXaxis , nonZeroMinSSEMap);
    xlabel('Iteration');
    ylabel('Observed SSE per iteration');
    grid minor
end

finalSSE = oldCentroidMatrix;
