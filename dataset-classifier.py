import os
import shutil

folder_path = './raw_dataset'  # Replace with the actual folder path
# 0 = crop 1 = weed
# Get the list of files in the folder
files = os.listdir(folder_path)
crop = './dataset/crop/'
# weed = 'D:\ML_Data\Project5_Ag_Crop and weed detection\Actual_data\Weed\\'
weed='./dataset/weed'
# Iterate over each file
for file_name in files:
    if file_name.endswith('.txt'):  # Check if the file has a .txt extension
        file_path = os.path.join(folder_path, file_name)  # Get the full file path
        name=file_name[:-4]
        img_name= name+ '.jpeg'
        img_path = os.path.join(folder_path,img_name)


        with open(file_path, 'r') as file:
            contents = file.read()  # Read the entire contents of the file
            # print(img_name)
            # print(contents[0])
            # print(img_path)
            if (contents[0]== '0'):
                shutil.copy(img_path, crop)
                # os.system(f'copy {img_path} {crop}')
            else :
                # os.system(f'copy {img_path} {weed}')
                shutil.copy(img_path, weed)
