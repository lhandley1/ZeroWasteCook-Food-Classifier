# ZeroWasteCook 🍳 - Food Image Classification with InceptionV3
<div align="center">
  <img alt="ZeroWasteCook Logo" src="https://github.com/lhandley1/assets/blob/main/DarkZeroWasteCookLogo.png" width="150" />
</div>

## Description

This project demonstrates the process of fine-tuning the InceptionV3 deep learning model for food image classification using a custom dataset. The primary objective is to develop a model capable of accurately identifying various food ingredients from images. This capability is a fundamental functionality of the **ZeroWasteCook** application, which aims to help users reduce food waste by identifying ingredients they have and suggesting recipes. InceptionV3, a state-of-the-art convolutional neural network pre-trained on the massive ImageNet dataset, requires input images to be preprocessed to a standard size of 299x299 pixels.

## Dataset 🍎🥦🥕

The dataset used in this project is the "Food Classification Dataset" provided by Bjorn on Kaggle. It contains a collection of images featuring various food items. We will leverage this data to train our model to accurately classify different food ingredients.

**Dataset Link:** https://www.kaggle.com/datasets/bjoernjostein/food-classification

## Contents 📋

1.  **Import Libraries** 📚: Imports necessary libraries for PyTorch, data manipulation, and visualisation.
2.  **Upload and Extract Dataset** 📁: Handles the uploading and extraction of the dataset from a ZIP file.
3.  **Load and Summarise Data** 📊: Reads the dataset labels and image directory, and provides a summary of the data.
4.  **Custom Dataset Class** 🏗️: Defines a custom PyTorch `Dataset` class (`IngredientsDataset`) to handle loading and preprocessing of the food images and their labels.
5.  **Preprocessing/Transform Function** ✨: Defines the image transformations required for the InceptionV3 model, including resizing, cropping, converting to tensor, and normalisation.
6.  **Train/Validation Split** 📏: Splits the dataset into training and validation sets.
7.  **Create DataLoader** 🔄: Creates PyTorch `DataLoader` instances for efficient batching and shuffling of the training and validation data.
8.  **Display Image Sample** 🖼️: Visualises a batch of images from the training data.
9.  **Assign Hyperparameters** ⚙️: Sets the hyperparameters for the model training process, such as the number of classes, learning rate, and number of epochs.
10. **Create InceptionV3 Model Class** 🧠: Defines a custom PyTorch `nn.Module` class (`InceptionV3Network`) that loads the pre-trained InceptionV3 model and modifies the final layer for the specific number of food classes.
11. **Train Model Function** 💪: Implements the training loop for the InceptionV3 model, including forward and backward passes, loss calculation, and optimizer updates. It also includes plotting of training losses.
12. **Evaluate Model Function** ✅: Implements the evaluation of the trained model on the validation set, calculating overall accuracy and providing a classification report with per-class metrics.
13. **Train, Visualise, and Evaluate** 🚀: Executes the training and evaluation functions.
14. **Save Model** 💾: Saves the trained model's state dictionary to a file.

## Getting Started

1.  Upload the `archive.zip` dataset file when prompted.
2.  Run the cells sequentially to preprocess the data, define and train the model, and evaluate its performance.

## Results

The project provides the overall accuracy of the fine-tuned InceptionV3 model on the validation set and a detailed classification report showing precision, recall, and f1-score for each food class.

## Future Improvements

-   **Data Augmentation:** Explore techniques to enhance the dataset and improve model generalisation. 📈
-   **Hyperparameter Tuning:** Experiment with different settings to optimise model performance. 🔧

## Acknowledgement 🙏

We would like to acknowledge **Bjorn** for providing the "Food Classification Dataset" on Kaggle, which was essential for training and evaluating the model in this project.
