# Refined RadImagenet - Conversion Tools

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

To create a GitHub README file with the instructions for using Git to clone the Hugging Face repository `ogrenenmakine/Refined-RadImagenet`, you can format it as follows:

# Refined-RadImagenet Repository

This repository contains files from the Hugging Face repository `ogrenenmakine/Refined-RadImagenet`. Follow the instructions below to clone the repository using Git.

## Prerequisites

Ensure that you have Git and Git LFS (Large File Storage) installed on your system. Git LFS is necessary for handling large files in the repository.

### Install Git LFS

If you haven't installed Git LFS yet, you can do so using the following command:

```bash
git lfs install
```

## Cloning the Repository

To clone the entire repository into your local machine, use the following command:

```bash
git clone https://huggingface.co/ogrenenmakine/Refined-RadImagenet source
```

This command will clone all files from the repository into a directory named `source`.

## Notes

- Make sure you have sufficient storage space for large files.
- For more information about this dataset, visit the [Hugging Face page](https://huggingface.co/ogrenenmakine/Refined-RadImagenet).

Feel free to contribute or raise issues if you encounter any problems.

## Usage

1. **Extract the Dataset**:
```sh
python process.py
```
Ensure the dataset tar file is located at:
```
source/
```
The script will automatically extract to:
```
data/
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
