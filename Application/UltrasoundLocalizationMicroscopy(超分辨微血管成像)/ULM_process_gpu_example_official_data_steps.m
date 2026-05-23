%% IMPORTANT NOTE
% Before running this script:
% Download the ULM dataset from: https://zenodo.org/records/4343435
% (you can choose one or more of the following):
%   - PALA_data_InVivoMouseTumor
%   - PALA_data_InVivoRatBrain_part1
%   - PALA_data_InVivoRatBrain_part2
%   - PALA_data_InVivoRatBrainBolus_part1
%   - PALA_data_InVivoRatBrainBolus_part2
%   - PALA_data_InVivoRatKidney_part1
%   - PALA_data_InVivoRatKidney_part2


%% ========================================================================
% ENVIRONMENT SETUP
% =========================================================================
clear; clc; close all;

% Add paths for the PALA_SRUS toolbox and parent directories
currentPath = pwd;
parentDir = fileparts(fileparts(currentPath));
addpath(genpath(parentDir));

%% ========================================================================
% RENDERING FUNCTION CONTROL FLAGS
% =========================================================================
% Set to 1 to calculate and render specific velocity maps
flag_vel_norm = 1; % Absolute velocity magnitude map
flag_vel_z    = 1; % Axial directional velocity map (Z-axis)

%% ========================================================================
% LOAD DATA & DIRECTORY SETUP
% =========================================================================
dataFolder = 'D:\software_matlab\exampledata\ULM\PALA_data_InVivoRatKidney\IQ';
resultFolder = fullfile(dataFolder, 'results');

% Create the result directory if it does not exist
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

% Create a list of all .mat files in the data folder
IQdataList = {};
IQnum = 1;
fileList = dir([dataFolder, filesep, '*.mat']);
for i = 1:numel(fileList)
    IQdataList{IQnum} = [fileList(i).folder, filesep, fileList(i).name];
    IQnum = IQnum + 1;
end
IQnum = IQnum - 1; % Total number of data blocks


%% ========================================================================
% PARAMETERS CONFIGURATION
% =========================================================================
% Load the first file to extract dimensions
load(IQdataList{1});
bfiq = IQ;

% If micro bubbles are too dense, choose ROI to get better tracking
ROI_z = 1:size(IQ,1);
ROI_x = 1:size(IQ,2);
lamdba = 1540/15e6; % m
x_axis = ROI_x*lamdba;
z_axis = ROI_z*lamdba;

NbFrames = size(bfiq,3);
framerate = UF.FrameRateUF;  % compounded frame rate
res = 10;

% Define the ULM parameters structure
ULM = struct( ...                       
    'numberOfParticles', 100,...         % Number of particles extracted per frame (30-100)
    'res', res,...                       % Resolution factor (typically 10)
    'SVD_cutoff', [50, NbFrames - 50],...% SVD cutoff limits for tissue clutter filtering
    'max_linking_distance', 2,...        % Max allowed linking distance between frames (pixels)
    'min_length', 15,...                 % Minimum valid track length (rejects noise)
    'fwhm', [1 1]*3,...                  % Mask size for sub-pixel localization [z, x]
    'max_gap_closing', 0,...             % Max allowed missing frames in a single track
    'size', [size(bfiq(ROI_z,ROI_x,:),1), size(bfiq(ROI_z,ROI_x,:),2), NbFrames],...
    'scale', [lamdba*1000 lamdba*1000 1/framerate],...     % Physical scale [z, x, dt]
    'numberOfFramesProcessed', NbFrames,... 
    'interp_factor', 1/res,...           % Interpolation factor for track smoothing
    'butter_CuttofFreq', [50 300],...    % Butterworth filter cutoff frequencies
    'framerate', framerate...            % Frame rate (Hz)
    );

%% ========================================================================
% LOAD DLL & INITIALIZE GPU
% =========================================================================
% Check if the C++ CUDA DLL is loaded; if not, load it
if ~libisloaded('ULM')
    loadlibrary('ULM.dll', 'ULM.h');
end

% Initialize the GPU processing handle with ULM parameters
disp("------ Initializing ULM GPU ------")
gpu_handle = calllib('ULM', 'initializeULMGPU', ...
    ULM.numberOfParticles, ... 
    ULM.res, ...
    ULM.SVD_cutoff(1), ULM.SVD_cutoff(2),...
    ULM.max_linking_distance,...
    ULM.min_length,...
    ULM.fwhm(1), ULM.fwhm(2),...
    ULM.max_gap_closing,...
    ULM.size(1), ULM.size(2), ULM.size(3),...
    ULM.scale(1), ULM.scale(2), ULM.scale(3),...
    ULM.butter_CuttofFreq(1), ULM.butter_CuttofFreq(2),...
    ULM.framerate,...
    flag_vel_norm, flag_vel_z);
disp("------ Initialization Finished ------")

%% ========================================================================
% PROCESS EACH DATA BLOCK AND ACCUMULATE RESULTS
% =========================================================================
% Initialize accumulation matrices for the high-resolution grid
Density_result = zeros(length(ROI_z)*ULM.res, length(ROI_x)*ULM.res);
if(flag_vel_norm)
    Velocity_result_norm = zeros(length(ROI_z)*ULM.res, length(ROI_x)*ULM.res);
end
if(flag_vel_z)
    Velocity_result_z = zeros(length(ROI_z)*ULM.res, length(ROI_x)*ULM.res);
end

disp("------ Processing Start ------")

idx = 1;

load(IQdataList{idx});
disp("Processing " + IQdataList{idx})
bfiq = IQ(ROI_z, ROI_x, :); % Crop to ROI

% Extract real part and create a C-style single-precision pointer
single_data_real = single(real(bfiq));
single_data_real = reshape(single_data_real, [], 1);
single_data_real_ptr = libpointer('singlePtr', single_data_real);

% Extract imaginary part and create a C-style pointer
single_data_imag = single(imag(bfiq));
single_data_imag = reshape(single_data_imag, [], 1);
single_data_imag_ptr = libpointer('singlePtr', single_data_imag);

% Prepare a pointer with enough memory to receive the output from GPU
single_rev = zeros(2 * size(single_data_imag, 1), 1);
single_data_rev_ptr = libpointer('singlePtr', single_rev);


%% Call GPU DLL (Step 1: SVD)
ret = calllib('ULM', 'processULMGPU', gpu_handle, ...
    single_data_real_ptr, single_data_imag_ptr, single_data_rev_ptr, 1);

aa = single_data_rev_ptr.Value;

svd_real = aa(1:2:2*ULM.size(1)*ULM.size(2)*ULM.size(3));
svd_imag = aa(2:2:2*ULM.size(1)*ULM.size(2)*ULM.size(3));
svd_result = svd_real + 1i*svd_imag;
svd_result = reshape(svd_result,ULM.size(1),ULM.size(2),ULM.size(3));
figure(1)
for idx = 1:ULM.size(3)
    img = log_compressed(abs(svd_result(:,:,idx)));
    imagesc(img,[-30,0])
    colormap(gray)
    pause(0.01)
end


%% Call GPU DLL (Step 2: localization)
ret = calllib('ULM', 'processULMGPU', gpu_handle, ...
    single_data_real_ptr, single_data_imag_ptr, single_data_rev_ptr, 2);

aa = single_data_rev_ptr.Value;
aa = aa(1:4*ULM.size(3)*ULM.numberOfParticles);
MatTracking = reshape(aa,4,[]);

figure(2);
for i = 1:size(svd_result,3)
    img_envelope = abs(svd_result(:,:,i));
    img_log = log_compressed(img_envelope);
    imagesc(img_log,[-30 0]);
    colormap("gray");axis tight
    
    % 2. Hold the current axes to overlay plots on the image
    hold on;
    
    % 3. Filter microbubbles belonging to the current frame (frame i)
    % Use logical indexing: find all columns in MatTracking where the 4th row equals i
    current_frame_idx = (MatTracking(4, :) == i); 
    
    % 4. Extract x and y coordinates for the current frame
    x_coords = MatTracking(3,current_frame_idx); % Row 3: x-coordinate
    y_coords = MatTracking(2,current_frame_idx); % Row 2: y-coordinate
    
    % 5. Plot red crosses ('r+' represents red cross)
    % MarkerSize controls the size, LineWidth controls the thickness
    plot(x_coords, y_coords, 'r+', 'MarkerSize', 5, 'LineWidth', 1);
    
    % 6. Release hold to allow refreshing the background and markers in the next loop
    hold off;
    title(sprintf('Localized Points - Frame: %d / %d', i, size(svd_result,3)));
    pause(0.01);
end

%%
% Call GPU DLL (Step 3: tracking)
ret = calllib('ULM', 'processULMGPU', gpu_handle, ...
    single_data_real_ptr, single_data_imag_ptr, single_data_rev_ptr, 3);

aa = single_data_rev_ptr.Value;
num_tracks = aa(1); % Read the total number of tracks
idx = 2;            % Set the read index/pointer

% Preallocate with a cell array to improve execution speed
tracked_cells = cell(num_tracks, 1);

for t = 1:num_tracks
    % Read the number of points in the current track
    num_points = aa(idx); 
    idx = idx + 1;
    
    if num_points > 0
        % Extract sequentially stored [Z, X, Frame], chunk length is 3 * num_points
        chunk = aa(idx : idx + 3*num_points - 1);
        
        % Convert 1D chunk to an N x 3 matrix, where each row is [Z, X, Frame]
        % Note: MATLAB is column-major, so reshape to 3 rows first, then transpose
        pts = reshape(chunk, 3, [])'; 
        
        % Add a 4th column as Track ID (useful for drawing trajectory lines later)
        track_ids = repmat(t, num_points, 1);
        tracked_cells{t} = [pts, track_ids];
        
        % Update the read index
        idx = idx + 3*num_points;
    end
end

% Concatenate into the final N x 4 matrix [Z, X, Frame, TrackID]
all_tracked_points = cell2mat(tracked_cells);


figure(3);
for i = 1:size(svd_result,3)
    % Draw background image (pass x_axis and z_axis if you want to use physical units for coordinates)
    img_envelope = abs(svd_result(:,:,i));
    img_log = log_compressed(img_envelope);
    imagesc(img_log, [-30 0]);
    colormap("gray"); axis tight;

    hold on;

    % Find all microbubble points belonging to the current frame i
    % The 3rd column of all_tracked_points is the frame number (iFrame)
    current_frame_idx = (all_tracked_points(:, 3) == i);

    % Extract X and Z coordinates
    % (Note: The format is Column 1 = Z/Depth, Column 2 = X/Width)
    z_coords = all_tracked_points(current_frame_idx, 1);
    x_coords = all_tracked_points(current_frame_idx, 2);

    % Plot red crosses
    plot(x_coords, z_coords, 'r+', 'MarkerSize', 5, 'LineWidth', 1);

    hold off;
    title(sprintf('Tracked Points - Frame: %d / %d', i, size(IQ,3)));

    pause(0.01);
end


%% ========================================================================
% UNLOAD DLL AND CLEANUP
% =========================================================================
% Release GPU memory handles and unload the dynamic library
calllib('ULM', 'deleteULMGPUHandle', gpu_handle);
unloadlibrary('ULM')
disp("------ ULM Processing Completely Finished ------")