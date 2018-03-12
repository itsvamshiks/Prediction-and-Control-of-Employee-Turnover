
a = csvread('HR_DATA.csv',1,0);
labels = a(:,7);
j=1;
k=1;
left_data=[];


for i=1:14999
    if eq(labels(i,1),1)
        left_data(j,:)=a(i,:);
        left_data_labels(j,:)=1;
        j=j+1;
    else    
        work_data(k,:)=a(i,:);
        work_data_labels(k,:)=-1;
        k=k+1;
    end
end


train1=left_data(1:3571,:);
train1_label=left_data_labels(1:3571,:);
train2=work_data(1:3571,:);
train2_label=work_data_labels(1:3571,:);


train=vertcat(train1,train2);
train_label=vertcat(train1_label,train2_label);


rtrain1=train(:,1:6);
rtrain2=train(:,8:9);

rtrain=horzcat(rtrain1,rtrain2);


rtrain1=work_data(:,1:6);
rtrain2=work_data(:,8:9);

rtest=horzcat(rtrain1,rtrain2);


weights=zeros(1,8);
b=0;


rand('seed', 1);
inds = randperm(size(rtrain, 1));
rtrain = rtrain(inds, :);
train_label = train_label(inds, :);


for ep=1:1
    
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
    preds(i,2)= -1;
    preds(i,3)=sign(pred)*sign(-1);
    preds(i,4)=i;
    if preds(i,3)<=0
        err=err+1;
    end
end

NTL=sortrows(preds,1);

NTL1(1:10,:)=NTL(11419:11428,4);


for i=1:10
   
    display(a(NTL1(i,1),:));
    
end


err_arr=((size(rtest,1)-err)/size(rtest,1))*100;








