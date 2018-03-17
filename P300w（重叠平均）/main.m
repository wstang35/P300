%
% This program predicts the first character in session 12, run 01, using a very simple classification method
% This classification method uses only one sample (at 310ms) and one channel (Cz) for classification
% 
% (C) Gerwin Schalk; Dec 2002
clc;
clear all;
fprintf(1, '2nd Wadsworth Dataset for Data Competition:\n');
fprintf(1, 'Data from a P300-Spelling Paradigm\n');
fprintf(1, '-------------------------------------------\n');
fprintf(1, '(C) Gerwin Schalk 2002\n\n');

% load data file
fprintf(1, 'Loading data file\n');
files = { 'AAS010R01' , 'AAS010R02', 'AAS010R03' , 'AAS010R04'...
    'AAS010R05' , 'AAS011R01' , 'AAS011R02' , 'AAS011R03', ...
    'AAS011R04' , 'AAS011R05' , 'AAS011R06' , 'AAS012R01' , ...
    'AAS012R02' , 'AAS012R03' , 'AAS012R04' , 'AAS012R05' , ...
    'AAS012R06' , 'AAS012R07' , 'AAS012R08' };

%基本参数
samplefreq=240;
triallength=round(600*samplefreq/1000);     % samples in one evoked response
max_stimuluscode=12;
channelselect= [9 11 13 34 49 51 53 56 60 62];     %通道选择
titlechar='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-';
downsample=12; %下采样

%% ============ 构造training set特征矩阵 ===================================

%空特征矩阵
filedata={};
filey={};
fileCode={};
%读取所有样本（11次run中每次trail）的特征矩阵
for filesnumber=1:11
    load( files{ filesnumber } );
% 0.1-20Hz bandpass filter on the signal
    signal = passband_filter(signal);

    starttrial=min(trialnr)+1;    % intensification to start is the first intensification
    endtrial=max(trialnr);        % the last intensification is the last one for the last character
        
    %构造特征矩阵
    % go through all intensifications and calculate classification results after each
    fprintf(1, 'Going through all intensifications in %d \n',filesnumber);
    trialdata=zeros((endtrial-starttrial+1),(length(channelselect)*triallength/downsample)); %构造矩阵，用于存放每次trail中144*10个特征数据
    trialy=zeros((endtrial-starttrial+1),1);
    trialCode=zeros((endtrial-starttrial+1),1);
    for cur_trial=starttrial:endtrial
        % get the indeces of the samples of the current intensification
        trialidx=find(trialnr == cur_trial);
        % get the data for these samples (i.e., starting at time of stimulation and triallength samples
        signaldata=signal(min(trialidx):min(trialidx)+triallength-1, channelselect);
        signaldata=signaldata(1:downsample:end,:);
        signaldata=signaldata(:)';
        trialdata(cur_trial-starttrial+1,:)=signaldata;
        trialy(cur_trial-starttrial+1)=StimulusType(min(trialidx));
        trialCode(cur_trial-starttrial+1)=StimulusCode(min(trialidx));
    end % session
    filedata{filesnumber}=trialdata;  %构造特征矩阵
    filey{filesnumber} = trialy;
    fileCode{filesnumber} = trialCode;
end

%合并各文件数据
traindata=cat(1,filedata{:});
trainy=cat(1,filey{:});
trainCode=cat(1,fileCode{:});
traindata(7119,:)=[];
trainy(7119,:)=[];
trainCode(7119,:)=[];
%重叠平均
lastdata=[];
lasty=[];
for charnum=1:(size(traindata,1)/180)
    avgtraindata=zeros(12,size(traindata,2));
    avgtrainy=zeros(12,1);
    for i=1:12
        tempdata=traindata(((charnum-1)*180+1):(charnum*180),:);
        tempCode=trainCode(((charnum-1)*180+1):(charnum*180),:);
        tempy=trainy(((charnum-1)*180+1):(charnum*180),:);
        idx=find(tempCode==i);
        avgtraindata(i,:)=mean(tempdata(idx,:));
        avgtrainy(i)=tempy(min(idx));
    end
    lastdata=[lastdata;avgtraindata];
    lasty=[lasty;avgtrainy];
end
    

%% ============ 从0样本中选取适量训练样本，零一样本比为1. ===================================
oddidx = find(lasty == 1);
disoddidx = find(lasty == 0);
disoddidxRandSelect =disoddidx;%disoddidx(randperm(length(disoddidx),1*length(oddidx)));%从0样本中选取适量训练样本，零一样本比为1.  
lastdata = lastdata([oddidx;disoddidxRandSelect],:);%训练数据
lasty = lasty([oddidx;disoddidxRandSelect],:);


%% ============ 读取test set 数据 ===================================
%空特征矩阵
filedata={};
fileCode={};
%读取所有样本（11次run中每次trail）的特征矩阵
for filesnumber=12:19
    load( files{ filesnumber } );
% 0.1-20Hz bandpass filter on the signal
    signal = passband_filter(signal);

    starttrial=min(trialnr)+1;    % intensification to start is the first intensification
    endtrial=max(trialnr);        % the last intensification is the last one for the last character
        
    %构造特征矩阵
    % go through all intensifications and calculate classification results after each
    fprintf(1, 'Going through all intensifications in %d \n',filesnumber);
    trialdata=zeros((endtrial-starttrial+1),(length(channelselect)*triallength/downsample)); %构造矩阵，用于存放每次trail中144*10个特征数据
    trialy=zeros((endtrial-starttrial+1),1);
    for cur_trial=starttrial:endtrial
        % get the indeces of the samples of the current intensification
        trialidx=find(trialnr == cur_trial);
        % get the data for these samples (i.e., starting at time of stimulation and triallength samples
        signaldata=signal(min(trialidx):min(trialidx)+triallength-1, channelselect);
        signaldata=signaldata(1:downsample:end,:);
        signaldata=signaldata(:)';
        trialdata(cur_trial-starttrial+1,:)=signaldata;
        trialy(cur_trial-starttrial+1,:)=StimulusCode(min(trialidx));
    end % session
    filedata{filesnumber-12+1}=trialdata;  %构造特征矩阵
    fileCode{filesnumber-12+1} = trialy;
end

%合并各文件数据
testdata=cat(1,filedata{:});
testCode=cat(1,fileCode{:});

% ttesty=testCode;  %逐个刺激分类
% ttestdata=testdata;

%重叠平均
ttestdata=[];
ttesty=[];
for charnum=1:(size(testdata,1)/180)
    avgtestdata=zeros(12,size(testdata,2));
    avgtesty=zeros(12,1);
    for i=1:12
        tempdata=testdata(((charnum-1)*180+1):(charnum*180),:);
        tempCode=testCode(((charnum-1)*180+1):(charnum*180),:);
        idx=find(tempCode==i);
        avgtestdata(i,:)=mean(tempdata(idx,:));
        avgtesty(i)=i;
    end
    ttestdata=[ttestdata;avgtestdata];
    ttesty=[ttesty;avgtesty];
end

%% ============ 采用PCA降维 ===================================
[train_scale,test_scale]=scaleForSVM(lastdata,ttestdata);%先归一
[train_pca,test_pca] = pcaForSVM(train_scale,test_scale);

% %% ============ 采用learning Curve ===================================
% X = train_pca(1:150,:);
% y = trainy(1:150,:);
% Xval = train_pca(150:end,:);
% yval = trainy(150:end,:);
% lambda = 1;
% [error_train, error_val] = ...
% learningCurve([ones(size(X, 1), 1) X], y, ...
%                   [ones(size(Xval, 1), 1) Xval], yval, ...
%                   lambda);
% 
% plot(1:size(X, 1), error_train, 1:size(X, 1), error_val);
% title('Learning curve for linear regression')
% legend('Train', 'Cross Validation')
% xlabel('Number of training examples')
% ylabel('Error')
% axis([0 150])
% 
% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:size(X, 1)
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% 
%% ============ 寻找SVM参数C,g(K-fold CV) ===================================
fprintf(1, '寻找最优C,g参数\n');
C = 0;
g = 0;
[bestacc,bestc,bestg] = SVMcgForClass(lasty,train_pca);
fprintf(1, '寻参完成\n');
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),]; %' -w1 3 -w0 1'

%% ============ 训练SVM ===================================
model = svmtrain(lasty,train_pca,cmd);
fprintf(1, 'SVM训练完成\n');

%% ============ 训练结果===================================
CR = ClassResult(lasty, train_pca, model, 1);
%% ============ 逐字符预测(每180个trial) ===================================
predictchar='';
for charnum=1:(size(test_pca,1)/12)
    testlabel=zeros(12,1);
    [testlabel,accuracy,score]=svmpredict(testlabel,test_pca(((charnum-1)*12+1):(charnum*12),:),model);
     %哪一行，哪一列的1标签多，则判断结果为该行，该列
    [~,targetcolumn]=max(score(1:6));
     [~,targetrow] = max(score(7:12));
     predictchar(charnum)=titlechar((targetrow-1)*6+targetcolumn);
end
fprintf(1,predictchar);
fprintf(1,'\nFOODMOOTHAMPIECAKETUNAZYGOT4567');
accuracy=mean(predictchar=='FOODMOOTHAMPIECAKETUNAZYGOT4567')*100;
fprintf(1,'\n结果推算完成\n');
fprintf(1,'字符预测准确率为%.2f%%\n',accuracy);



