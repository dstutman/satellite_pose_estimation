% MATLAB version
% This code use EPnP code for Microsat Pose Estimation Assignment

clc; clear all; close all;

% addpath data; 
% addpath error;
addpath EPnP; 
addpath Images_Test/annotations; % Add path where you store .json file


fprintf('\n---------EPnP--------------\n');

% Camera intrinsic paramethers matrix A (perform calibration before)

fx = 800;   fy = 800;   f = 0.05;
u0 = 4032/2;   v0 = 3042/2;
% m = fx/f;
m = 1;

A = [ fx/m   0     u0/m;     % Intrinsic Paramethers
       0    fy/m   v0/m;
       0     0       1 ];

% Cooperative target (we know its dimentions)

point.model  = [ 0,  0,   0; % Point 1
                 0,  0, -10; % Point 2
                 0, 10, -10; % Point 3
                 0, 10,   0; % Point 4
                30,  0,   0; % Point 5
                30,  0, -10; % Point 6
                30, 10, -10; % Point 7
                30, 10,   0];% Point 8
            
first_image = 2936; % ID of the first image to be analyzed
last_image  = 2942; % ID of the last  image to be analyzed

num_images = (last_image - first_image) +1; % number of images to be analyzed

for index_fig = 1:num_images
    
    fileName = ['IMG_',num2str(index_fig+2935),'.json'];  % Filename in JSON extension
    disp(fileName)                                        % Print the name of the current image
    fid = fopen(fileName);                                % Opening the file
    raw = fread(fid,inf);                                 % Reading the contents
    str = char(raw');                                     % Transformation
    fclose(fid);                                          % Closing the file
    data = jsondecode(str);                               % Using the jsondecode function to parse JSON from string
    
    index = 0;
    for i = 0:1
        for j = 1:4
            point.Ximg(j+i*4,:,index_fig) = data.keypoints(i+1,j,:); % Re-organize data as EPnP needed
            if point.Ximg(j+i*4,end,index_fig) == 1
                index = index+1;
                point_detected(index) = j+i*4; % ID of the point detected in the current figure
            end
        end
    end
    Xworld = point.model(point_detected,:);  % Detected Point coordinates in World Reference Frame
    Ximg   = point.Ximg(point_detected,1:2); % Detected Point coordinates in Image Plane
    
    n = index; % number of point detected
    
    if n > 3 % Control number of points detected
    
    % EPnP (suppressed) ----------------------------------------------------
    
    %     x3d_h = [Xworld,ones(n,1)];
    %     x2d_h = [Ximg,  ones(n,1)];
    %
    %     [Rp,Tp,Xc,sol]=efficient_pnp(x3d_h,x2d_h,A);
    %     fprintf('\n--------- Rotation Matrix --------------\n');
    %     Rp
    %     fprintf('\n--------- Traslation vector --------------\n');
    %     Tp
    % draw Results
    %     for i=1:n
    %         point.Xcam_est(i,:)=Xc(i,:)';
    %     end
    
    % EPnP_GAUSS_NEWTON (currently used) ----------------------------------------------------
    
    x3d_h = [Xworld,ones(n,1)]; % homogeneous coordinates of the points in world reference
    x2d_h = [Ximg,  ones(n,1)]; % homogeneous position of the points in the image plane
    
    [R,T,Xc,sol] = efficient_pnp_gauss(x3d_h,x2d_h,A); % PnP solver
    
    fprintf('\n--------- Rotation Matrix (Gauss Newton optimization) --------------\n');
    disp(R)
    fprintf('--------- Traslation vector (Gauss Newton optimization) --------------\n');
    disp(T')
    % draw Results
    %     for i=1:n
    %         point.Xcam_est(i,:)=Xc(i,:)';
    %     end
    
    else
        fprintf('\n Less then 4 points detected. Pose estimation not possible. \n');
    end
    clear point_detected Xworld Ximg
end

