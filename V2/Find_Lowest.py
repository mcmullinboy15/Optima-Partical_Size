import os
import shutil

import torch

counter = 16
for branch in ['Branch_A', 'Branch_B']:
    sum = 0
    for batch in [16, 32, 64, 128, 256, 512]:
        data_folder = f"{branch}/models/{batch}/"

        for data_filename in os.listdir(data_folder):
            data_path = f"{data_folder}{data_filename}"
            loaded_dict = torch.load(data_path, map_location=torch.device('cpu'))
            lowest_val_loss = loaded_dict['lowest_val_loss']
            if lowest_val_loss < 13:
                sum += lowest_val_loss
                print(data_path, lowest_val_loss)

                # get and convert to Andrews_v2_*th for the Wet mill computer
                shutil.copy(data_path, f"WetMillModels/Andrews_v2_{counter}.pth")
                counter += 1
    print(sum)
