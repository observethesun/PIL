[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org)

## Machine Learning for Reconstruction of Polarity Inversion Lines from Solar Filaments

This repository contains the source code for the paper "Machine Learning for Reconstruction of Polarity Inversion Lines from Solar Filaments". It includes datasets, neural network models, and scripts used in the study, aiming to reconstruct the Sun's magnetic neutral lines using machine learning techniques.

## Repository Structure

- **McIntosh_data/**: This directory is intended for storing FITS images or other image formats to be processed by the neural network.
- **results/**: This directory contains four archives with the averaged predictions, tables of errors, and the tensors of mean predictions in `mean_predictions_*.pt`, which are described in our article.
  - `results_1.zip`, `results_32.zip` and `results_64.zip` contain the results for models trained using grid-based reference points with corresponding step sizes. Here, the number in the archive name indicates the step size of the grid.
  - `results_None.zip` contains the results for models trained without grid-based reference points.
- **scripts/**: Includes various Python scripts essential for running and managing the neural network model:
  - `config.py`: Contains global parameters and settings for the model.
  - `helper.py`: Auxiliary functions for various tasks within the project.
  - `imagedata.py`: A class that instances are made based on the input image for processing.
  - `neutralliner.py`: The core neural network model code.

- **demo.ipynb**: A Jupyter notebook that provides an example of how to use the model with a step-by-step guide. The `demo.ipynb` notebook provides a comprehensive example of how to use the neural network for predicting polarity inversion lines from solar filaments. Follow the steps and code cells within the notebook for a guided experience.

- **requirements.txt**: A text file listing the necessary Python modules.

## Getting Started

1. **Clone the Repository**: To get started with the code and models in this repository, first clone it using git:

    ```bash
    git clone https://github.com/observethesun/PIL.git
    ```

2. **Install Dependencies**: Navigate to the cloned repository's directory and install the required Python modules:

    ```bash
    cd PIL
    pip install -r requirements.txt
    ```

3. **Prepare Your Data**: Place your FITS images or other appropriate files into the `McIntosh_data/` directory.

4. **Check config**: Ensure to review `config.py` for any configurations specific to your environment or data.

5. **Model Evaluation and Usage**: Use `demo.ipynb` as a guided example of how to use the model for your specific needs. It will walk you through the process of loading data, running the model, and interpreting the results.

### Citing this work

```
Kisielius, V., Illarionov, E. Machine Learning for Reconstruction of Polarity Inversion Lines from Solar Filaments. Sol Phys 299, 69 (2024). https://doi.org/10.1007/s11207-024-02324-9
```
