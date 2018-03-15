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
channelselect=[9 11 13 34 49 51 53 56 60 62];     %通道选择
titlechar='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-';
%% ============ 构造training set特征矩阵 ===================================

%空特征矩阵
stimulusdata=[];
stimulusy=[];
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
    for cur_trial=starttrial:endtrial
        % get the indeces of the samples of the current intensification
        trialidx=find(trialnr == cur_trial);
        % get the data for these samples (i.e., starting at time of stimulation and triallength samples
        trialdata=signal(min(trialidx):min(trialidx)+triallength-1, :);
        trialdata=trialdata(:,channelselect);%选择通道
        trialdata=trialdata(:)'; %构造行式特征向量(1X1440)
        stimulusdata=[stimulusdata;trialdata];  %构造特征矩阵
        stimulusy = [stimulusy;StimulusType(min(trialidx))];
    end % session
end

%从0样本中选取适量训练样本，零一样本比为1.
oddidx = find(stimulusy == 1);
disoddidx = find(stimulusy == 0);
[stimulusdata,mu,sigma] = featureNormalize(stimulusdata);%归一化
disoddidxRandSelect = disoddidx(randperm(length(disoddidx),1*length(oddidx)));%从0样本中选取适量训练样本，零一样本比为1.
train_data = stimulusdata([oddidx;disoddidxRandSelect],:);%训练数据
train_y = stimulusy([oddidx;disoddidxRandSelect],:);
stimulusdata = train_data;
stimulusy = train_y;

%% ============ 寻找SVM参数C,g(K-fold CV) ===================================
% fprintf(1, '寻找最优C,g参数\n');
% C = 0;
% g = 0;
% [bestCVaccuracy,C,g] = SVMcgForClass(stimulusy,stimulusdata);
% fprintf(1, '寻参完成\n');

%% ============ 训练SVM ===================================
model = svmtrain(stimulusy,stimulusdata,'-s 0 -t 2 -c 1.7411 -g 0.0039');
fprintf(1, 'SVM训练完成\n');

%% ============ 读取test set(SESSION 12)数据并判断字符===================================

%读取session12各个run的特征矩阵
for filesnumber=12:19
    load( files{ filesnumber } );
    % 0.1-20Hz bandpass filter on the signal
    signal = passband_filter(signal);
    % get a list of the samples that divide one character from the other
    idx=find(PhaseInSequence == 3);                                % get all samples where PhaseInSequence == 3 (end period after one character)
    charsamples=idx(find(PhaseInSequence(idx(1:end-1)+1) == 1));   % get exactly the samples at which the trials end (i.e., the ones where the next value of PhaseInSequence equals 1 (starting period of next character))
    %防止有的3到1，trailnr==0
    selectcharsamples=[];
    for ct=1:(length(charsamples))
        if(trialnr(charsamples(ct))~=0)
            selectcharsamples=[selectcharsamples,ct];
        end
    end
    charsamples=charsamples(selectcharsamples,:);
    charsamples=[1;charsamples];
    
    %逐个字符判断
    predictchar='';
    for cs=1:(length(charsamples)-1)
        %空特征矩阵
        testdata=[];
        testCode=[];
        % this determines the first and last intensification to be used here
    	% in this example, this results in evaluation of intensification 1...180 (180 = 15 sequences x 12 stimuli)
        starttrial=trialnr(charsamples(cs))+1;                                     % intensification to start is the first intensification
        endtrial=max(trialnr(find(samplenr < charsamples(cs+1))));         % the last intensification is the last one for the first character

        %构造特征矩阵
        % go through all intensifications and calculate classification results after each
        fprintf(1, 'Going through all intensifications in %d \n',filesnumber);
        for cur_trial=starttrial:endtrial
            % get the indeces of the samples of the current intensification
            trialidx=find(trialnr == cur_trial);
            % get the data for these samples (i.e., starting at time of stimulation and triallength samples
            trialdata=signal(min(trialidx):min(trialidx)+triallength-1, :);
            trialdata=trialdata(:,channelselect);%选择通道
            trialdata=trialdata(:)'; %构造行式特征向量(1X1440)
            testdata=[testdata;trialdata];  %构造特征矩阵
            testCode=[testCode;StimulusCode(min(trialidx))];
        end % session
        %归一化，用训练样本的mu和sigma
        testdata = bsxfun(@minus, testdata, mu);
        testdata = bsxfun(@rdivide, testdata, sigma);
        %对归一化之后的数据进行分类
        testlabel = zeros(size(testdata,1),1);
        testlabel = svmpredict(testlabel,testdata,model);
        %哪一行，哪一列的1标签多，则判断结果为该行，该列
        sumlabel = zeros(6,1);
        for Code=1:6
            sumlabel(Code)=sum(testlabel(find(testCode==Code)));
        end
        [~,targetcolumn] = max(sumlabel);
        for Code=7:12
            sumlabel(Code-6)=sum(testlabel(find(testCode==Code)));
        end
        [~,targetrow] = max(sumlabel);
        predictchar(cs)=titlechar((targetrow-1)*6+targetcolumn);
    end
    fprintf(1, predictchar);%输出预测单词
    fprintf(1, '\n'); %输出下个单词前换行
end


