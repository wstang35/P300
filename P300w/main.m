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

%��������
samplefreq=240;
triallength=round(600*samplefreq/1000);     % samples in one evoked response
max_stimuluscode=12;
channelselect=[9 11 13 34 49 51 53 56 60 62];     %ͨ��ѡ��
titlechar='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-';
%% ============ ����training set�������� ===================================

%����������
filedata={};
filey={};
%��ȡ����������11��run��ÿ��trail������������
for filesnumber=1:11
    load( files{ filesnumber } );
% 0.1-20Hz bandpass filter on the signal
    signal = passband_filter(signal);

    starttrial=min(trialnr)+1;    % intensification to start is the first intensification
    endtrial=max(trialnr);        % the last intensification is the last one for the last character
        
    %������������
    % go through all intensifications and calculate classification results after each
    fprintf(1, 'Going through all intensifications in %d \n',filesnumber);
    trialdata=zeros((endtrial-starttrial+1),(length(channelselect)*triallength)); %����������ڴ��ÿ��trail��144*10����������
    trialy=zeros((endtrial-starttrial+1),1);
    for cur_trial=starttrial:endtrial
        % get the indeces of the samples of the current intensification
        trialidx=find(trialnr == cur_trial);
        % get the data for these samples (i.e., starting at time of stimulation and triallength samples
        signaldata=signal(min(trialidx):min(trialidx)+triallength-1, channelselect);
        signaldata=signaldata(:)';
        trialdata(cur_trial-starttrial+1,:)=signaldata;
        trialy(cur_trial-starttrial+1)=StimulusType(min(trialidx));
    end % session
    filedata{filesnumber}=trialdata;  %������������
    filey{filesnumber} = trialy;
end

%�ϲ����ļ�����
traindata=cat(1,filedata{:});
trainy=cat(1,filey{:});
%% ============ ��0������ѡȡ����ѵ����������һ������Ϊ1. ===================================
oddidx = find(trainy == 1);
disoddidx = find(trainy == 0);
disoddidxRandSelect = disoddidx(randperm(length(disoddidx),1*length(oddidx)));%��0������ѡȡ����ѵ����������һ������Ϊ1.
traindata = traindata([oddidx;disoddidxRandSelect],:);%ѵ������
trainy = trainy([oddidx;disoddidxRandSelect],:);


%% ============ ��ȡtest set ���� ===================================
%����������
filedata={};
fileCode={};
%��ȡ����������11��run��ÿ��trail������������
for filesnumber=12:19
    load( files{ filesnumber } );
% 0.1-20Hz bandpass filter on the signal
    signal = passband_filter(signal);

    starttrial=min(trialnr)+1;    % intensification to start is the first intensification
    endtrial=max(trialnr);        % the last intensification is the last one for the last character
        
    %������������
    % go through all intensifications and calculate classification results after each
    fprintf(1, 'Going through all intensifications in %d \n',filesnumber);
    trialdata=zeros((endtrial-starttrial+1),(length(channelselect)*triallength)); %����������ڴ��ÿ��trail��144*10����������
    trialy=zeros((endtrial-starttrial+1),1);
    for cur_trial=starttrial:endtrial
        % get the indeces of the samples of the current intensification
        trialidx=find(trialnr == cur_trial);
        % get the data for these samples (i.e., starting at time of stimulation and triallength samples
        signaldata=signal(min(trialidx):min(trialidx)+triallength-1, channelselect);
        signaldata=signaldata(:)';
        trialdata(cur_trial-starttrial+1,:)=signaldata;
        trialy(cur_trial-starttrial+1,:)=StimulusCode(min(trialidx));
    end % session
    filedata{filesnumber-12+1}=trialdata;  %������������
    fileCode{filesnumber-12+1} = trialy;
end

%�ϲ����ļ�����
testdata=cat(1,filedata{:});
testCode=cat(1,fileCode{:});

%% ============ ����PCA��ά ===================================
[train_scale,test_scale]=scaleForSVM(traindata,testdata);%�ȹ�һ
[train_pca,test_pca] = pcaForSVM(train_scale,test_scale);

% %% ============ ����learning Curve ===================================
% X = train_pca(1:1800,:);
% y = trainy(1:1800,:);
% Xval = train_pca(1800:end,:);
% yval = trainy(1800:end,:);
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
% axis([0 1800 0 150])
% 
% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:size(X, 1)
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;

% % ============ Ѱ��SVM����C,g(K-fold CV) ===================================
% fprintf(1, 'Ѱ������C,g����\n');
% C = 0;
% g = 0;
% [bestCVaccuracy,C,g] = SVMcgForClass(trainy,train_pca);
% fprintf(1, 'Ѱ�����\n');

%% ============ ѵ��SVM ===================================
model = svmtrain(trainy,train_pca,'-s 0 -t 2 -c 1.7411 -g 0.0039');
fprintf(1, 'SVMѵ�����\n');

%% ============ ���ַ�Ԥ��(ÿ180��trial) ===================================
predictchar='';
for charnum=1:(size(test_pca,1)/180)
    testlabel=zeros(180,1);
    testlabel=svmpredict(testlabel,test_pca(((charnum-1)*180+1):(charnum*180),:),model);
     %��һ�У���һ�е�1��ǩ�࣬���жϽ��Ϊ���У�����
     sumlabel = zeros(6,1);
     for Code=1:6
        sumlabel(Code)=sum(testlabel(find(testCode(((charnum-1)*180+1):(charnum*180),:)==Code)));
     end
     [~,targetcolumn] = max(sumlabel);
     for Code=7:12
        sumlabel(Code-6)=sum(testlabel(find(testCode(((charnum-1)*180+1):(charnum*180),:)==Code)));
     end
     [~,targetrow] = max(sumlabel);
     predictchar(charnum)=titlechar((targetrow-1)*6+targetcolumn);
end
fprintf(1,predictchar);
fprintf(1,'\n����������\n');



