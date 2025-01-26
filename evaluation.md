# Project Evaluation Checklist

## Problem Description
- [x] Problem is described in README with enough context, so it's clear what the problem is and how the solution will be used

## EDA
- [x] Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis) 
    - For images: analyzing the content of the images.
    - For texts: frequent words, word clouds, etc.

## Model Training
- [x] Trained multiple models and tuned their parameters. For neural networks: same as previous, but also with tuning (adjusting learning rate, dropout rate, size of the inner layer, etc.)

## Exporting Notebook to Script
- [x] The logic for training the model is exported to a separate script

## Reproducibility
- [x] It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data

## Model Deployment 
- [x] Model is deployed (with Flask, BentoML, or a similar framework)

## Dependency and Environment Management
- [x] Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to activate the environment

## Containerization
- [x] The application is containerized and the README describes how to build a container and how to run it

## Cloud Deployment 
- [x] There's code for deployment to the cloud or Kubernetes cluster (local or remote). There's a URL for testing or a video/screenshot of testing it


