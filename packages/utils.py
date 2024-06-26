from packages.common_packages import tf, np, os


# ---------------------------------------------------------------------------
# Function to print the directory  labels
# ---------------------------------------------------------------------------
def list_directory_contents(directory, label):
    contents = np.array(tf.io.gfile.listdir(str(directory)))
    print(f'{label} commands labels: {contents}')
    return contents


# ---------------------------------------------------------------------------
# Function to get the Model size in KB or MB
# ---------------------------------------------------------------------------
def get_and_convert_file_size(file_path, unit=None):
    size = os.path.getsize(file_path)
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')
