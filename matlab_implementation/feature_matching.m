I1 = rgb2gray(imread("Images Series/Preliminary Series/IMG_5810.JPG"));
option = 4;

if option == 1
    rot_angle = 60;
    delta_x = 0.0;
    delta_y = 0.0;
    scaling = 1.0;
    save_file = 'case_1.mat'
elseif option == 2
    rot_angle = 0;
    delta_x = 25;
    delta_y = 50;
    scaling = 1.0;
    save_file = 'case_2.mat'
elseif option == 3
    rot_angle = 0;
    delta_x = 0.0;
    delta_y = 0.0;
    scaling = 1.5;
    save_file = 'case_3.mat'
elseif option == 4
    rot_angle = 0;
    delta_x = 0.0;
    delta_y = 0.0;
    scaling = 1.0;
    save_file = 'case_4.mat'
end

I2 = imtranslate(imresize(imrotate(I1,rot_angle),scaling),[delta_x, delta_y], 'FillValues', 255);


points1 = detectSURFFeatures(I1,'MetricThreshold',20000);
points2 = detectSURFFeatures(I2,'MetricThreshold',20000);

[f1, vpts1] = extractFeatures(I1, points1);
[f2, vpts2] = extractFeatures(I2, points2);

indexPairs = matchFeatures(f1,f2);
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
legend("matched points 1", "matched points 2");

matchedCoords1 = matchedPoints1.Location;
matchedCoords2 = matchedPoints2.Location;

save(save_file, "matchedCoords1", "matchedCoords2");



