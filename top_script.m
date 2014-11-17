clc
clear all
dbstop if error

training_ratio = 0.5;
sample_length = 25501;
load NormalizedFeaturesSet2
feature_set = NormalizedFeaturesSet2;
params.classifier_type = 2;
% 1 - KNN
% 2 - GMM
% 3 - NN
% 4 - SVM

if (1 == params.classifier_type)
% knn params
params.num_neighbours = 5;
params.distance_type = 'cityblock';
elseif (2 -- params.classifier_type)
% GMM params
params.num_components = 3;
end

% randomizing the order of input
rand_order = randperm(sample_length);
training_length = round(training_ratio*sample_length);

training_data = feature_set(rand_order(1:training_length),:);
testing_data = feature_set(rand_order(training_length+1:end),:);

train.features = training_data(:,1:end-1);
train.class = training_data(:,end);

testing.features = testing_data(:,1:end-1);
testing.class = testing_data(:,end);

model = training(train,params);

exp_class = classify(testing.features);

p1 = sum(feature_set(:,end) == 1)/sample_length;
p2 = sum(feature_set(:,end) == 2)/sample_length;
p3 = sum(feature_set(:,end) == 3)/sample_length;
confusion_mat = zeros(3,3);
for i = 1:3
    for j = 1:3
        confusion_mat(i,j) = sum((exp_class == i).*(testing.class == j))/sum(testing.class == j);
    end
end
e1 = confusion_mat(2,1) + confusion_mat(3,1);
e2 = confusion_mat(1,2) + confusion_mat(3,2);
e3 = confusion_mat(2,3) + confusion_mat(2,3);

total_err = p1*e1 + p2*e2 + p3*e3
confusion_mat
