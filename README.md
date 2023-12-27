# Machine Learning for Reconstruction of Polarity Inversion Lines from Solar Filaments

This repository contains the code for the scientific paper "Machine Learning for Reconstruction of Polarity Inversion Lines from Solar Filaments". It includes datasets, neural network models, and scripts used in the study, aiming to reconstruct the Sun's magnetic neutral lines using machine learning techniques.

## Repository Structure

- **McIntosh_data/**: This directory is intended for storing FITS images or other image formats to be processed by the neural network. Ensure that the images are named and formatted correctly before processing.
- **results/**: This directory contains four archives with the averaged predictions, tables of errors, and the tensors of mean predictions in `mean_predictions_*.pt` format. 
- **scripts/**: Includes various Python scripts essential for running and managing the neural network model:
  - `config.py`: Contains global parameters and settings for the model.
  - `helper.py`: Auxiliary functions for various tasks within the project.
  - `imagedata.py`: A class that instances are made based on the input image for processing.
  - `neutralliner.py`: The core neural network model code.

- **demo.ipynb**: A Jupyter notebook that provides an example of how to use the model with a step-by-step guide.

- **requirements.txt**: A text file listing the necessary Python modules. Install them using the command below:

```bash
pip install -r requirements.txt
```

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

3. **Prepare Your Data**: Place your FITS images or other appropriate files into the `McIntosh_data/` directory, ensuring they adhere to the expected naming conventions and formats.

4. **Run the Scripts**: Execute the scripts located in the `scripts/` directory to preprocess your data, train the model, or generate predictions. Ensure to review `config.py` for any configurations specific to your environment or data.

5. **Model Evaluation and Usage**: Use `demo.ipynb` as a guided example of how to use the model for your specific needs. It will walk you through the process of loading data, running the model, and interpreting the results.

6. **Review Results**: Check the `results/` directory to see the outputs, including averaged predictions, error metrics, and mean prediction tensors, described in our article.

## Usage Example

The `demo.ipynb` notebook provides a comprehensive example of how to use the neural network for predicting polarity inversion lines from solar filaments. Follow the steps and code cells within the notebook for a guided experience.

## Citation

If you use the code or models from this repository in your research, please cite the accompanying scientific paper:
