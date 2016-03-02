% clc;
% clear;
% close all;

tic;
train_s = importdata('./Arcene_Dataset/Train_data.txt');
train_l = importdata('./Arcene_Dataset/Train_labels.txt');
test_s = importdata('./Arcene_Dataset/Valid_data.txt');
test_l = importdata('./Arcene_Dataset/Valid_labels.txt');

ks = [10 100];
for i = 1:2
    k = ks(1,i)
    
    randorder = randperm(size(train_s,1));
    rand_samples = train_s(randorder, :);
    rand_labels = train_l(randorder, :);
    
    train_s = rand_samples;
    train_l = rand_labels;
    
    randorder = randperm(size(test_s,1));
    rand_samples = test_s(randorder, :);
    rand_labels = test_l(randorder, :);
    
    test_s = rand_samples;
    test_l = rand_labels;
    
    % PCA
    train_s_new' = kernelpca(train_s',k);
    test_s_new' = kernelpca(test_s',k);
    
    [predict_lbls, acc_pca] = svm_classifier(train_s_new, train_l, test_s_new, test_l);
    acc_pca;
    accu_pca(1,i) = acc_pca;
    
end

% Fisher LDA
train_s_n = kernellda(train_s',train_l);
test_s_n = kernellda(test_s',test_l);

[predict_lbls1, acc_lda] = svm_classifier(train_s_n, train_l, test_s_n, test_l);
acc_lda;
accu_lda(1,1) = acc_lda;

mean_accu = mean(accu_pca)
std = std(accu_pca)

accu_lda

toc