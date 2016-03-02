function [predict_lbl, accuracy] = svm_classifier(trn_ftr,trn_lbl,tst_ftr,tst_lbl)

% TRAIN
bestcv = 0;
for log2c = -5:5,
  for log2g = -5:5,
      
      %Linear
      cmd = ['-t 0 -v 5 -c ', num2str(10^log2c), ' -g ', num2str(10^log2g)];
      
      %RBF
      cmd = ['-t 2 -v 5 -c ', num2str(10^log2c), ' -g ', num2str(10^log2g)];
      
      cv = svmtrain(trn_lbl,trn_ftr, cmd);
      if (cv >= bestcv),
        bestcv = cv; 
        bestc = 10^log2c; 
        bestg = 10^log2g;
      end
   end
end

% Linear 
cmd = ['-t 0 -c ', num2str(bestc), ' -g ', num2str(bestg)];

%RBF
cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg)];

model = svmtrain(trn_lbl,trn_ftr, cmd);

 
% TEST

[predict_lbl, accuracy, dec_val] = svmpredict(tst_lbl, tst_ftr, model);
