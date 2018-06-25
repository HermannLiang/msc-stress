% --------------- Arduino2Matlab serial communication receiver ------------
% Description:
% This program is used to receive the transmition from Arduino. The
% transmitted data are physiological signals such as ECG and Respiration.
% -------------------------------------------------------------------------
% Other Information:
% Programmer: Youqian ZHANG
% Company: Imperial College London
% Date: 12 Jul 2017
% -------------------------------------------------------------------------
close all;
clc;
clear all;
delete(instrfindall); % Close serial port

% serial communication
s = serial('COM3'); % Change this to the real COM port
set(s, 'BaudRate', 115200);
fopen(s);
flushinput(s);

% initialize
fs = 50;

data_ECG = zeros(1, 1);
data_RES = zeros(1, 1);
data_TEM = zeros(1, 1);
stop_time = 10; % unit:seconds

tic;
% while toc < stop_time
 while 1
    try
        new_data = str2num(fscanf(s));
        
        switch new_data(1)
            case 1 % ECG
                data_ECG = [data_ECG, new_data(2)];
            case 2 % Airflow
                data_RES = [data_RES, new_data(2)];
%             case 3 % Temperature
%                 data_TEM = [data_TEM, new_data(2)];
        end;
    catch
        continue;
    end;
    
     
end;
fclose(s);
figure
subplot(2,2,1)
plot((1:length(data_ECG))./50,data_ECG)
title('ECG')
subplot(2,2,2)
plot((1:length(data_RES))./50,data_RES)
title('RES')
subplot(2,2,4)
plot((1:length(data_RES))./50,movmean(data_RES,50))
title('moving average RES')
subplot(2,2,3)
plot((1:length(data_TEM))./2,data_TEM)
title('Temperature')

data={data_ECG,data_RES};
