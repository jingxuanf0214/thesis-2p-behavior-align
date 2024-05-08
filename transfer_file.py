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
move_struct_files('//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/FB4R_imaging')

def move_odor_folders(pathname1):
    # Define the path for the 'odor_trials' folder
    odor_trials_path = os.path.join(pathname1, "odor_trials")
    
    # Check if 'odor_trials' folder exists, create if not
    if not os.path.exists(odor_trials_path):
        os.makedirs(odor_trials_path)
    
    # Loop through all items in pathname1
    for item in os.listdir(pathname1):
        item_path = os.path.join(pathname1, item)
        
        # Check if the item is a directory and its name contains 'odor'
        if os.path.isdir(item_path) and 'odor' in item.lower():
            # Define the destination path within 'odor_trials'
            destination_path = os.path.join(odor_trials_path, item)
            
            # Move the folder
            shutil.move(item_path, destination_path)
            print(f"Moved '{item}' to '{odor_trials_path}'")

# Example usage
#move_odor_folders('//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/dan_imaging')


def process_folders(pathname1):
    # Step 1: Create 'to_processed' folder if it doesn't exist
    to_processed_path = os.path.join(pathname1, 'to_processed')
    if not os.path.exists(to_processed_path):
        os.makedirs(to_processed_path)
    
    # Step 2: Loop through folders under pathname1
    for folder_name in os.listdir(pathname1):
        folder_path = os.path.join(pathname1, folder_name)
        
        # Ensure it's a directory
        if not os.path.isdir(folder_path):
            continue
        
        data_path = os.path.join(folder_path, 'data')
        intermediate_path = os.path.join(folder_path, 'intermediate analysis output')
        
        # Step 3: Check for 'data' subfolder
        if os.path.exists(data_path):
            continue  # Skip this folder
        
        # Step 4: Check for 'intermediate analysis output' subfolder
        if os.path.exists(intermediate_path):
            # Rename 'intermediate analysis output' to 'data'
            new_data_path = os.path.join(folder_path, 'data')
            os.rename(intermediate_path, new_data_path)
            # Create 'results' subfolder
            results_path = os.path.join(folder_path, 'results')
            os.makedirs(results_path)
        else:
            # Step 5: If neither subfolder exists, move to 'to_processed'
            shutil.move(folder_path, to_processed_path)

# Example usage
#process_folders('//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/hDeltaB_imaging')


def delete_matching_folders(pathname1, pathname2):
    # Ensure pathname1 and pathname2 are directories
    if not os.path.isdir(pathname1) or not os.path.isdir(pathname2):
        print("One of the provided paths is not a directory.")
        return

    # Loop through folders under pathname1
    for folder_name in os.listdir(pathname1):
        folder_path1 = os.path.join(pathname1, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path1):
            matching_folder_path2 = os.path.join(pathname2, folder_name)
            
            # If a matching folder exists under pathname2, delete it
            if os.path.exists(matching_folder_path2) and os.path.isdir(matching_folder_path2):
                try:
                    shutil.rmtree(matching_folder_path2)
                    print(f"Deleted folder: {matching_folder_path2}")
                except Exception as e:
                    print(f"Error deleting folder {matching_folder_path2}: {e}")

# Example usage
#delete_matching_folders('//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/hDeltaB_imaging/low_quality', 'F:/Wilson lab/hDeltaB_imaging')


def move_and_organize_folders(pathname1, pathname2):
    # Loop through each folder in pathname1
    for folder_name in os.listdir(pathname1):
        folder_path1 = os.path.join(pathname1, folder_name)
        
        # Ensure it's a directory
        if os.path.isdir(folder_path1):
            matching_folder_path2 = os.path.join(pathname2, folder_name)
            
            # Check if the matching folder exists under pathname2
            if os.path.exists(matching_folder_path2) and os.path.isdir(matching_folder_path2):
                # Create 'data' and 'results' subfolders in the matching pathname2 folder
                data_path = os.path.join(matching_folder_path2, 'data')
                results_path = os.path.join(matching_folder_path2, 'results')
                os.makedirs(data_path, exist_ok=True)
                os.makedirs(results_path, exist_ok=True)
                
                # Move all contents from the pathname1 folder to the 'data' folder in pathname2
                for item in os.listdir(folder_path1):
                    source_item_path = os.path.join(folder_path1, item)
                    destination_item_path = os.path.join(data_path, item)
                    shutil.move(source_item_path, destination_item_path)
                
                # Delete the original folder from pathname1
                #shutil.rmtree(folder_path1)

# Example usage
#move_and_organize_folders('C:/Users/wilson/OneDrive - Harvard University/Thesis - Wilson lab/2P imaging/preprocessed data/qualified_sessions/multi_trial_sessions/no_results', '//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/hDeltaB_imaging/low_quality')


def move_folders_with_empty_results(pathname):
    # Create the 'no_results' directory if it doesn't exist
    no_results_path = os.path.join(pathname, 'no_results')
    if not os.path.exists(no_results_path):
        os.makedirs(no_results_path)
        print(f"Created directory: {no_results_path}")

    # Loop through all items in pathname
    for folder_name in os.listdir(pathname):
        folder_path = os.path.join(pathname, folder_name)
        
        # Skip the 'no_results' folder itself
        if folder_name == 'no_results' or folder_name == 'to_check':
            continue
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # Construct the pathname for the 'results' subfolder
            results_folder_path = os.path.join(folder_path, 'results')
            
            # Check if 'results' subfolder exists and is empty
            if os.path.exists(results_folder_path) and not os.listdir(results_folder_path):
                # Move the whole folder to 'no_results'
                dest_path = os.path.join(no_results_path, folder_name)
                shutil.move(folder_path, dest_path)
                print(f"Moved {folder_path} to {dest_path}")



#move_folders_with_empty_results(base_path)
                
def move_folders_containing_odor(pathname):
    # Create the 'with_odor' directory if it doesn't exist
    with_odor_path = os.path.join(pathname, 'with_odor')
    if not os.path.exists(with_odor_path):
        os.makedirs(with_odor_path)
        print(f"Created directory: {with_odor_path}")

    # Loop through all items in pathname
    for folder_name in os.listdir(pathname):
        folder_path = os.path.join(pathname, folder_name)
        
        # Skip the 'with_odor' folder itself
        if folder_name.lower() == 'with_odor':
            continue
        
        # Check if the item is a directory and its name contains 'odor'
        if os.path.isdir(folder_path) and 'odor' in folder_name.lower():
            # Move the folder to 'with_odor'
            dest_path = os.path.join(with_odor_path, folder_name)
            shutil.move(folder_path, dest_path)
            print(f"Moved {folder_path} to {dest_path}")

#move_folders_containing_odor(base_path)