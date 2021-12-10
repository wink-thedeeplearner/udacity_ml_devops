# Project: Customer Churn Predictor

- Machine Learning DevOps Engineer Nanodegree

## Project Description
Below is the script to build and create models for predicting **customer churn**.
The original source code is in `churn_notebook.html`, which was used for this project.

To avoid dependencies and versioning issues, the code in this repo is implemented using a `Docker image`.

| **File Name** | **Description** |
|---| ----|
| `Dockerfile` | All the commands/instructions to assemble the Docker image. |
| `features.py` | Features used in data exploration and model building. |
| `churn_predictor.py` | The main python script, which contains methods for data exploration and visualization, model building, and classification report generation. |
| `churn_predictor_test.py` | Unit test via Pytest. |


## Implementation
1. Copy the `churn_predictor` folder and change directory into `churn_predictor`:

```
cd churn_predictor
```

2. Build the Docker image in the same directory using:

```
docker build -t churn_predictor .
```

After the image is built successfully, launch the entire procedure with:

```
docker run churn_predictor
```

The unit tests (`churn_predictor_test.py`) will automatically launch. If no tests are needed, launch the `churn_predictor.py` file:

```
docker run churn_predictor churn_predictor.py
```

Result of tests are available in `logs` folder.

The results for plotted figures and models are stored in the following folders: 
- `plot_figures/eda`
- `plot_figures/result`
- `models`

The above folders are created as Docker volumes.
