# Radiology AI Project

This code designed to process the RadImagenet and convert to stratified and refined organization.

## Folder Structure

```
correction_masks/
data/
output/
source/
    radimagenet.tar.gz
    RadiologyAI_test.csv
    RadiologyAI_train.csv
    RadiologyAI_val.csv
process.py
```

### Directories

- **correction_masks/**: Contains correction masks for the images.
- **data/**: Contains the extracted radiology images.
- **output/**: Directory for output files.
- **source/**: Contains source files and datasets.
  - **radimagenet.tar.gz**: the original compressed RadImagenet file.
  - **RadiologyAI_test.csv**: CSV file for test dataset.
  - **RadiologyAI_train.csv**: CSV file for training dataset.
  - **RadiologyAI_val.csv**: CSV file for validation dataset.

### Files

- **process.py**: Main script to process and organize the RadImagenet files.

## Usage

1. **Extract the Dataset**:
```sh
   python process.py
```
Ensure the dataset tar file is located at:
```
source
```
The script will automatically extract to:
```
data
```

2. **Process the Images**:
   The script will read the CSV files, refined the images, and organize accordingly.

## Dependencies

- Python 3.9+
- pandas
- OpenCV
- tarfile
- tqdm
- numpy

Install the required packages using pip:

```sh
pip install pandas opencv-python tarfile tqdm numpy
```

## License

This project is licensed under the MIT License.
