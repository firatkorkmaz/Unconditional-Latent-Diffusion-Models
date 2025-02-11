import os
import shutil
from tqdm import tqdm
from zipfile import ZipFile


# Function to zip latent files
def zip_folder(folder_path, zipfile_path):
    # Collect all files to be zipped
    files_to_zip = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_to_zip.append(file_path)

    # Create a ZIP file with a progress bar
    with ZipFile(zipfile_path, 'w') as zipf:
        for file_path in tqdm(files_to_zip, desc=f"Zipping files", unit="file", leave=True):
            # Add each file to the ZIP archive
            zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(folder_path)))
    
    print(f"Latents dataset saved to: '{zipfile_path}'")


# Function to extract files from .zip file
def unzip(zipfile_path, extract_to, filetype):
    # Define extensions
    if filetype.lower() == "image":
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff',)
    elif filetype.lower() == "latent":
        extensions = ('.npy',)
    else:
        raise ValueError("Filetype must be either image or latent!")
        
    # Define the target directory for files
    os.makedirs(extract_to, exist_ok=True)
    
    # Start extraction process
    with ZipFile(zipfile_path, "r") as zip_file:
        # Get file list with defined extensions
        file_list = zip_file.namelist()
        files = [f for f in file_list if not f.endswith('/') and f.endswith(extensions)]
        
        if not files:
            raise ValueError("No files found in the ZIP archive.")

        # Display progress bar
        with tqdm(total=len(files), desc=f"Extracting {os.path.basename(zipfile_path)}") as progress_bar:
            for file_path in files:
                # Get only the file name (ignoring folder structure)
                file_name = os.path.basename(file_path)
                try:
                    # Extract file with streaming
                    with zip_file.open(file_path) as source, open(os.path.join(extract_to, file_name), 'wb') as target:
                        shutil.copyfileobj(source, target)
                except Exception as e:
                    print(f"Error extracting {file_path}: {e}")
                progress_bar.update(1)
    
    # Display number of extracted files
    print(f"Extraction complete!")
    for subdirectory, _, files in os.walk(extract_to):
        if (filecount := sum(1 for file in files if os.path.splitext(file)[-1] in ['.png', '.jpg', '.npy'])) > 0:
            print(f"{subdirectory}: {filecount} files.")
