filename = '/Users/karthikchandrasekar/Desktop/SecondSem/SML/ProgrammingAssignment/dataset1.txt';
delimiter = '';
M = dlmread(filename, delimiter);
[Mrow, Mcol] = size(M);
clusters = 3;

centroidIndex = zeros(clusters, 1);
for k=1:clusters
    centroidIndex(k) = randi([1, Mrow], 1,1);
end

inputCentroidMap = zeros(Mrow,1);

for index = 1:Mrow
    inputVect = M(index, :);
    minVal = 1000000;
    mindIndex = 0;
    for kVal = 1:clusters
        disp(kVal);
        centroid = M(centroidIndex(kVal), :); 
        result = inputVect - centroid;
        result = result.*result;
        dist = sum(result);
        if dist == min([dist minVal])
            minIndex = centroidIndex(k);
        end
    end
    inputCentroidMap(index) = minIndex;    
end
%disp(inputCentroidMap);
disp(centroidIndex);


