DeLTA_mat_file_dir = "/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/DeLTA_data/mother_machine/evaluation/postprocessed/"
mat_file_names = dir(strcat(DeLTA_mat_file_dir,"*.mat"));
output_dir = strcat(DeLTA_mat_file_dir,"masks/")
f = waitbar(0, 'Starting');
for x=1:length(mat_file_names)
    current_dir = strcat(mat_file_names(x).folder,"/",mat_file_names(x).name);
    data = load(current_dir);
    chambers = length(data.res);
    for y=1:chambers
        timepoints = table(size(data.res(y).labelsstack)).Var1(3);
        for z=1:timepoints
        current_image_name = strcat("Position",num2str(x,"%02.f"),"_Chamber",num2str(y,"%02.f"),"_Frame",num2str(z,"%03.f"),".png");
        imwrite(data.res(y).labelsstack(:,:,z),strcat(output_dir,current_image_name))
        end
    end
    waitbar(x/length(mat_file_names), f, sprintf('Progress: %d %%', floor(x/length(mat_file_names)*100)))
end

