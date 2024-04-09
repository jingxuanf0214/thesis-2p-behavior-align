import os
import shutil

def find_folders_and_create_directory(pathname1):
    # Directory where the selected folders will be placed
    target_directory = os.path.join(pathname1, 'qualified_sessions')
    
    # Check if the target directory already exists; if not, create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created directory: {target_directory}")
    else:
        print(f"Directory already exists: {target_directory}")
    
    # Look for folders containing 'hdeltab' (case-insensitive)
    for folder in os.listdir(pathname1):
        folder_path = os.path.join(pathname1, folder)
        if os.path.isdir(folder_path) and "hdeltab" in folder.lower():
            # Instead of moving, we just print out what we would move as an example
            print(f"Folder found: {folder}")
            dest_path = os.path.join(target_directory, folder)
            shutil.move(folder_path, dest_path)
            print(f"Moved: {folder_path} to {dest_path}")

# Example usage
#pathname1 = "..\preprocessed data"y
#find_folders_and_create_directory(pathname1)

def move_folders_if_no_match(pathname1, pathname2):
    # List all items in pathname1
    items_in_pathname1 = os.listdir(pathname1)
    
    for item in items_in_pathname1:
        item_path = os.path.join(pathname1, item)
        
        # Check if the current item is a directory
        if os.path.isdir(item_path):
            # Construct the path to a potentially matching folder in pathname2
            matching_path_in_pathname2 = os.path.join(pathname2, item)
            
            # Check if the matching folder exists in pathname2
            if not os.path.exists(matching_path_in_pathname2):
                # If not, move the folder from pathname1 to one layer above
                destination_path = os.path.abspath(os.path.join(pathname1, os.pardir, item))
                
                # Move the folder
                shutil.move(item_path, destination_path)
                
                print(f"Moved {item} to {destination_path}")

# Example usage:
#pathname1 = "..\preprocessed data\qualified_sessions"
#pathname2 = "Z:\hDeltaB_imaging\qualified_sessions"
#move_folders_if_no_match(pathname1, pathname2)

def move_roiData_struct_files(pathname1, pathname2):
    # Ensure pathname1 and pathname2 are directories
    if not os.path.isdir(pathname1) or not os.path.isdir(pathname2):
        print("One of the paths is not a directory.")
        return
    
    # Loop through folders in pathname2
    for folder_name in os.listdir(pathname2):
        folder_path = os.path.join(pathname2, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            processed_data_path = os.path.join(folder_path, 'processed data')
            
            # Check if 'processed data' subfolder exists
            if os.path.exists(processed_data_path):
                # Look for 'roiData_struct' files
                for file_name in os.listdir(processed_data_path):
                    if 'roiData_struct' in file_name:
                        source_file_path = os.path.join(processed_data_path, file_name)
                        
                        # Construct destination path under matching folder in pathname1
                        dest_folder_path = os.path.join(pathname1, folder_name)
                        dest_file_path = os.path.join(dest_folder_path, file_name)
                        
                        # Ensure the destination folder exists
                        if not os.path.exists(dest_folder_path):
                            os.makedirs(dest_folder_path)
                        
                        # Move the file
                        shutil.move(source_file_path, dest_file_path)
                        print(f"Moved {source_file_path} to {dest_file_path}")


# Example usage
#pathname1 = r'C:\Users\example\pathname1'
#pathname2 = r'C:\Users\example\pathname2'
#move_roiData_struct_files(pathname1, pathname2)

def move_one_trial_sessions(pathname1,target_name):
    target_folder = os.path.join(pathname1, target_name)
    
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Loop over all folders in pathname1
    for folder_name in os.listdir(pathname1):
        folder_path = os.path.join(pathname1, folder_name)
        
        # Skip if it's not a directory or it's the target directory
        if not os.path.isdir(folder_path) or folder_name == target_name:
            continue
        
        # List all data files in the folder
        data_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # Check for files with "trial1" and exclude folders with other trial numbers
        has_trial1 = any("trial1" in f for f in data_files)
        has_other_trials = any("trial" + str(i) in f for i in range(2, 10) for f in data_files)  # Adjust range as needed
        
        # If the folder contains only "trial1" files, move it to the target folder
        if has_trial1 and not has_other_trials:
            dest_path = os.path.join(target_folder, folder_name)
            shutil.move(folder_path, dest_path)
            print(f"Moved {folder_path} to {dest_path}")

# Example usage
#pathname1 = "..\preprocessed data\qualified_sessions"
#move_one_trial_sessions(pathname1,'one_trial_sessions')

def organize_matlab_files(pathname):
    # Loop over all items in the given pathname
    for folder_name in os.listdir(pathname):
        folder_path = os.path.join(pathname, folder_name)
        
        # Check if the current item is a directory
        if os.path.isdir(folder_path):
            # Define paths for the 'data' and 'results' subfolders
            data_folder_path = os.path.join(folder_path, 'data')
            results_folder_path = os.path.join(folder_path, 'results')
            
            # Check if both 'data' and 'results' folders exist
            if os.path.exists(data_folder_path) and os.path.exists(results_folder_path):
                print(f"Skipping folder as 'data' and 'results' already exist: {folder_path}")
                continue  # Skip the rest of the loop for this folder
            
            # Create 'data' folder if it doesn't exist
            if not os.path.exists(data_folder_path):
                os.makedirs(data_folder_path)
                print(f"Created folder: {data_folder_path}")
            
            # Create 'results' folder if it doesn't exist
            if not os.path.exists(results_folder_path):
                os.makedirs(results_folder_path)
                print(f"Created folder: {results_folder_path}")
            
            # Move all .mat files to the 'data' folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.mat'):
                    source_file_path = os.path.join(folder_path, file_name)
                    dest_file_path = os.path.join(data_folder_path, file_name)
                    shutil.move(source_file_path, dest_file_path)
                    print(f"Moved {source_file_path} to {dest_file_path}")

# Example usage
#pathname = "..\..\preprocessed data\FR1_sessions"
#organize_matlab_files(pathname)


def move_struct_files(pathname1):
    # Loop through all folders under pathname1
    for folder in os.listdir(pathname1):
        folder_path = os.path.join(pathname1, folder)
        
        # Check if this item is indeed a directory
        if os.path.isdir(folder_path):
            processed_data_path = os.path.join(folder_path, 'processed data')
            
            # Check if 'processed data' subfolder exists
            if os.path.exists(processed_data_path) and os.path.isdir(processed_data_path):
                # Loop through files in 'processed data'
                for file in os.listdir(processed_data_path):
                    if '_struct' in file and file.endswith('.mat'):
                        # Define source and destination paths
                        source_file_path = os.path.join(processed_data_path, file)
                        data_folder_path = os.path.join(folder_path, 'data')
                        
                        # Ensure 'data' subfolder exists, create if not
                        if not os.path.exists(data_folder_path):
                            os.makedirs(data_folder_path)
                        
                        destination_file_path = os.path.join(data_folder_path, file)
                        
                        # Move the file
                        shutil.move(source_file_path, destination_file_path)

# Example usage
move_struct_files('//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/FR1_imaging')
