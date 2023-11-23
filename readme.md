# Farm Ownership Prediction using Decision Trees

## Requirements

Before running the code, ensure that you have the following R packages installed:

- tidyverse
- rpart
- rpart.plot

You can install these packages using the following R commands:

```R
install.packages("tidyverse")
install.packages("rpart")
install.packages("rpart.plot")
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/farm-ownership-prediction.git
   ```

2. Navigate to the repository directory:

   ```bash
   cd farm-ownership-prediction
   ```


## Description

The code performs the following tasks:

1. Loads the required libraries, including `tidyverse`, `rpart`, and `rpart.plot`.

2. Reads the dataset, "IndonesianRiceFarms.csv," into a tibble called `riceFarms`.

3. Randomly splits the dataset into training (75% of records) and testing (25% of records) using a random seed of 370.

4. Generates a decision tree model to predict `FarmOwnership` based on the other variables in the dataset. It uses a complexity parameter (`cp`) of 0.01.

5. Visualizes the decision tree using `rpart.plot`.

6. Predicts classes for each record in the testing dataset and stores them in `riceFarmPredictions`.

7. Evaluates the model by forming a confusion matrix and calculating predictive accuracy.

8. Creates a new decision tree model using a different complexity parameter (`cp`) of 0.007.

9. Displays the new decision tree visualization, predicts classes for the new model, and calculates its predictive accuracy.

## Dataset

The dataset, "IndonesianRiceFarms.csv," contains information about Indonesian rice farms, including the independent variables used for prediction and the target variable, `FarmOwnership` (0 for sharecropped, 1 for farmer-owned).

Please ensure you have this dataset in the same directory as the R script or update the file path accordingly.
