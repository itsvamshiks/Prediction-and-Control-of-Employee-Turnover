
a = csvread('HR_DATA.csv',1,0);
labels = a(:,7);

train1=a(1:1500,:);
train1_label=labels(1:1500,:);
train2=a(2001:3500,:);
train2_label=labels(2001:3500,:);
test1=a(1501:2000,:);
test1_label=labels(1501:2000,:);
test2=a(3501:4000,:);
test2_label=labels(3501:4000,:);



train=vertcat(train1,train2);
train_label=vertcat(train1_label,train2_label);
test=vertcat(test1,test2);
test_label=vertcat(test1_label,test2_label);



rtrain1=train(:,1:6);
rtrain2=train(:,8:9);

rtrain=horzcat(rtrain1,rtrain2);


rtrain1=test(:,1:6);
rtrain2=test(:,8:9);

rtest=horzcat(rtrain1,rtrain2);


weights=zeros(1,8);
b=0;


rand('seed', 1);
inds = randperm(size(rtrain, 1));
rtrain = rtrain(inds, :);
train_label = train_label(inds, :);

for i=1:size(rtrain,1)
    if train_label(i,1)==0
        train_label(i,1)=-1;
    end
end



rand('seed', 1);
inds = randperm(size(rtest, 1));
rtest = rtest(inds, :);
test_label = test_label(inds, :);


for i=1:size(rtest,1)
    if test_label(i,1)==0
        test_label(i,1)=-1;
    end
end



for ep=1:1000
    
for i=1:size(rtrain,1) 
    pred=dot(rtrain(i,:),weights)+b;
    if pred*train_label(i,1)<=0
        weights=weights+rtrain(i,:)*train_label(i,1); 
        b=b+train_label(i,1);
    end
    
end

end

err=0;

for i=1:size(rtest,1)
    pred=dot(rtest(i,:),weights)+b;
    preds(i,1)=pred;
    preds(i,2)=test_label(i,:);
    preds(i,3)=sign(pred)*sign(test_label(i,1));
    if preds(i,3)<=0
        err=err+1;
    end
end


err_arr=((size(rtest,1)-err)/size(rtest,1))*100;








