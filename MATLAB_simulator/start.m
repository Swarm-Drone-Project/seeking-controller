close all
clear
% clc
Ix = 0.00527;       % kg-m^2
k2 = 2.5e-6;     % N-s^2
m = 1.7;         % kg
g = 9.81;        % m/s^2
l = 0.5;         % m
Kp = 50.9e4;
Kd = 80.5e3;
Ki = 1.1e4;
Kl = 0.0; % Kl = 0 means no lateral velocity compensation.

% rho_air = 0;
% C_drag = 0;

rho_air = 1.225;                    % kg/m^3
C_drag = 0.8;                       % Drag coefficient

u0_min = 1.0*m*g/(2*k2);
u0_max = 1.5*1.0*m*g/(2*k2);
T_boost = 0.01; % s

H = 100;
D = -300;
velocity_H = 12.0;
velocity_D = -12.0;
velocity_magnitude = 12;
headingRateMax = 5;
trajectory_class = 1;
Amplitude = 50;
target_trigger = 5;
model_output = sim("quadcopter_dynamics_wo_gp.slx");
disp("without_gp")
% run("Data_Animation.m")
% Call the function with your data
quadcopter_visualization_with_controls(model_output, H, D, l);