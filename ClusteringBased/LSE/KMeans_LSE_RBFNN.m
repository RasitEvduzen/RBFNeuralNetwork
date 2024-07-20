clc, clear, close all, warning off;
%% LSE Based 3D RBF Model Fitting
% Written By: Rasit
% 20-Jul-2024

%% RBF Fonksiyonu
rbfFunc = @(x, y, cx, cy, gamma, weight) weight * exp(-((x - cx).^2 + (y - cy).^2) / (2 * gamma^2));

%% Create Data
NoD = 2e3;
CenterX = [1 0 1 0];
CenterY = [0 1 1 0];
Sigma = [.25 .25 .25 .25];
Weight = [1 1 1 1];
ModelOrder = 4;
x = 3 * rand(NoD, 1) - 1;
y = 3 * rand(NoD, 1) - 1;
z = 0;
for i = 1:ModelOrder
    z = z + rbfFunc(x, y, CenterX(i), CenterY(i), Sigma(i), Weight(i));
end

%% K-Means Clustering And Nonlinear Regression
data = [x, y, z]; 
numClusters = ModelOrder; 
maxIterations = 100;

rng("default");
initialCenters = data(randperm(size(data, 1), numClusters), :);
centers = initialCenters;

figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')
for iter = 1:maxIterations
    distances = pdist2(data, centers);
    [~, idx] = min(distances, [], 2);

    newCenters = zeros(numClusters, 3);
    for i = 1:numClusters
        newCenters(i, :) = mean(data(idx == i, :), 1);
    end

    clf
    subplot(121);
    scatter3(x, y, z, 'red', 'LineWidth', 1), hold on
    scatter3(newCenters(:, 1), newCenters(:, 2), newCenters(:, 3), 100, 'kx', 'LineWidth', 3);
    title(['3D K-Means & LSE RBF Neural Network']);
    xlabel('X'), ylabel('Y'), zlabel('Z');
    grid on;

    clusterCenters = centers(:, 1:2);
    clusterWidths = zeros(numClusters, 1);
    for i = 1:numClusters
        clusterData = data(idx == i, :);
        clusterWidths(i) = std(clusterData(:, 3));
    end

    A = [];
    b = z;
    for i = 1:ModelOrder
        A = [A, rbfFunc(x, y, clusterCenters(i, 1), clusterCenters(i, 2), clusterWidths(i), 1)];
    end
    xlse = A\b;   % LSE Solution for RBF_NN

    Range = [-1 2];
    [X, Y] = meshgrid(linspace(Range(1), Range(2), 1e2));
    Z = 0;
    for i = 1:ModelOrder
        Z = Z + rbfFunc(X, Y, clusterCenters(i, 1), clusterCenters(i, 2), clusterWidths(i), xlse(i));
    end
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', .75), axis([Range(1) Range(2) Range(1) Range(2) -.2 1.2])

    subplot(122);
    scatter(x, y, 'red', 'LineWidth', 1), hold on
    scatter(newCenters(:, 1), newCenters(:, 2), 100, 'kx', 'LineWidth', 3);
    xlabel('X'), ylabel('Y'),grid on,axis equal;
    drawnow
    if isequal(newCenters, centers)
        break;
    end
    centers = newCenters;
end


