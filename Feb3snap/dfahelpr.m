function intdata = int_data(data)

%	 intdata = int_data(data)
%
%	 int_data integrates 
%
%        data    ... input data
%
%        intdata ... integrated data

sum = 0;
data = data - mean(data);

for i = 1:length(data)
    sum = sum+data(i);
    intdata(i) = sum;
end




