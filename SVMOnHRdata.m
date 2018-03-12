
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


weights=zeros(1,8);
bias_w=0;
grad=zeros(1,8);
bias_g=0;  

for ep=1:50  
    
for i=1:size(rtrain,1)
      grad=zeros(1,8);
         bias_g=0; 
         pred=dot(rtrain(i,:),weights)+bias_w;         
        if pred*train_label(i,1)<=1
            grad=grad+rtrain(i,:)*train_label(i,1); 
            bias_g=bias_g+train_label(i,1);
        end    
    r=1/i;    
    grad=grad-0.1*weights;
    weights=weights+r*grad;
    bias_w=bias_w+r*bias_g;   
end

end

succ=0;
err=0;

 for j=1:size(rtest,1)
        pred=dot(rtest(j,:),weights)+bias_w;
        preds(i,2)=test_label(j,1);
        preds(i,1)=pred;
        preds(i,3)=pred*test_label(j,1);
        if pred*test_label(j,1)>=1
            succ=succ+1;   
        else
            err=err+1;       
        end   
 end

accuracy = 100*(succ/(succ+err));
display(accuracy);







