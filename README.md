# Rainfall Prediction Using Neural Networks

## About the project

This project looks at whether rainfall can be predicted one week in advance using weather and location-based data.

I worked with daily weather data covering **March 2024 to February 2025** and used several environmental variables, along with the distance of each location from the dataset's centroid, as inputs to a neural network.

The problem is treated as a **binary classification task**:

* `1` → rainfall is expected during the following week
* `0` → rainfall is not expected during the following week

The rainfall label is based on a `Rainf_tavg_mean` threshold of `2.78 × 10⁻⁵`. The week-ahead target was created by shifting the rainfall label by 7 days for each latitude/longitude location.

## What I did

The project involved several stages:

1. Processed the original weather CSV files and calculated daily statistics.
2. Merged the different weather variables into a single dataset.
3. Handled missing values and removed duplicate records.
4. Calculated the Haversine distance from each location to the dataset centroid.
5. Created the rainfall and 7-day-ahead rainfall labels.
6. Split the data chronologically into training, validation, and test sets.
7. Selected nine features for the neural network.
8. Standardised the features using `StandardScaler`.
9. Trained a feedforward neural network using TensorFlow/Keras.
10. Evaluated the model using accuracy, precision, recall, F1 score, and ROC-AUC.

## Dataset

The original dataset consisted of **132 CSV files** covering 11 weather variables from March 2024 to February 2025. After processing and merging the data, the final week-ahead dataset contained **5,093,424 rows and 42 columns**, covering **15,159 unique latitude/longitude pairs**.

The raw dataset isn't included in this repository because of its size.

### Input features

The model uses nine features:

| Feature              | Description                                        |
| -------------------- | -------------------------------------------------- |
| `Tair_f_inst_mean`   | Air temperature                                    |
| `Qair_f_inst_mean`   | Specific humidity                                  |
| `Psurf_f_inst_mean`  | Surface pressure                                   |
| `LWdown_f_tavg_mean` | Longwave radiation                                 |
| `SWdown_f_tavg_mean` | Shortwave radiation                                |
| `Wind_f_inst_mean`   | Wind speed                                         |
| `TVeg_tavg_mean`     | Vegetation transpiration                           |
| `AvgSurfT_inst_mean` | Average surface temperature                        |
| `dist_to_centroid`   | Distance from the location to the dataset centroid |

The feature selection and scaling steps are implemented in the source code.

## Train / validation / test split

I used a chronological split rather than randomly splitting the data. This means the model is trained on earlier dates and evaluated on later dates, which is more appropriate for a forecasting problem.

| Dataset    | Period                |      Rows |
| ---------- | --------------------- | --------: |
| Training   | March–November 2024   | 3,835,227 |
| Validation | December 2024         |   469,929 |
| Test       | January–February 2025 |   788,268 |

This also helps avoid using future observations when training the model.

## Neural network

I used a relatively small feedforward neural network rather than a very deep architecture:

```text
9 input features
       ↓
Dense(64, ReLU)
       ↓
Dropout(0.3)
       ↓
Dense(32, ReLU)
       ↓
Dropout(0.3)
       ↓
Dense(1, Sigmoid)
```

The second hidden layer uses L2 regularisation with a value of `0.01`.

The model was trained using:

* **Optimizer:** Adam
* **Loss:** Binary cross-entropy
* **Learning rate:** 0.001
* **Batch size:** 32
* **Maximum epochs:** 50
* **Early stopping patience:** 5 epochs
* **Dropout:** 0.3

These settings were used to keep the model reasonably simple while also reducing the risk of overfitting.

## Results

The final model was evaluated on the **January–February 2025 test set**, which contained 788,268 observations.

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **73.34%** |
| Precision | **27.01%** |
| Recall    | **90.01%** |
| F1 Score  | **41.56%** |
| ROC-AUC   |  **~0.80** |

The model's strongest result was its **90.01% recall**, meaning it detected most of the rainfall events in the test set.

Precision was much lower at **27.01%**, largely because the dataset is imbalanced and the model produces a relatively high number of false positives. I also experimented with increasing the prediction threshold from 0.5 to 0.7, which improved precision to approximately 30–40% while reducing recall somewhat.

The saved test predictions are available in:

```text
results/test_predictions.csv
```

## Repository structure

```text
rainfall-prediction-neural-network/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── src/
│   └── machine_learning_rain_prediction.py
│
├── results/
│   ├── test_predictions.csv
│   └── README.md
│
└── report/
    └── MACHINE_LEARNING.pdf
```

## Running the project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rainfall-prediction-neural-network.git
cd rainfall-prediction-neural-network
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the raw dataset

The original weather CSV files are not included in the repository.

Place them in:

```text
data/raw/
```

The filenames should follow the naming convention expected by the Python script.

### 5. Run the model

```bash
python src/machine_learning_rain_prediction.py
```

The script will process the weather data, create the week-ahead dataset, train the neural network, generate predictions, and produce the evaluation outputs.

## Project files

### `src/machine_learning_rain_prediction.py`

Contains the complete preprocessing, feature engineering, model training, prediction, and evaluation pipeline.

### `results/test_predictions.csv`

Contains the test-set:

* true labels
* predicted labels
* predicted probabilities

### `report/MACHINE_LEARNING.pdf`

Contains the full coursework report, including the methodology, model configuration, data splitting strategy, evaluation, and results.

## Limitations

One of the main challenges in this project was the imbalance between rainfall and non-rainfall observations. Although the model achieved high recall, its precision was relatively low because of the number of false positives.

Another limitation is that the model only uses the selected environmental and spatial features. There are likely other factors that could improve rainfall forecasting, and more advanced approaches could potentially capture temporal and spatial patterns more effectively.

## What I learned

This project gave me practical experience with a complete machine learning workflow rather than just training a model on an already-prepared dataset.

In particular, I worked with:

* large-scale CSV data processing using pandas
* feature engineering
* Haversine distance calculations
* chronological train/validation/test splitting
* feature standardisation
* neural networks with TensorFlow/Keras
* dropout and L2 regularisation
* early stopping
* classification metrics
* confusion matrices and ROC curves
* handling an imbalanced classification problem

## Report

The full project report is included in the `report/` directory.

---

**Built as part of my Machine Learning coursework.**
