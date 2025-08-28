# Config for Parkinsons – Pro (v8.6)
DATA_PATH = "data/parkinsons.csv"
MODEL_PATH = "models/best_model.joblib"
RUNS_CSV = "assets/runs.csv"

TARGET = "status"
NAME_COL = "name"
FEATURES = [
    "MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)",
    "MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP",
    "MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA",
    "NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"
]

MODEL_LIST = ["LogisticRegression","RandomForest","SVC","GradientBoosting","ExtraTrees","XGBoost","MLP","KerasNN"]
DEFAULT_MODEL = "LogisticRegression"

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "roc_auc"

DEFAULT_PARAMS = {
    "LogisticRegression": {"C":1.0, "max_iter":200, "penalty":"l2"},
    "RandomForest": {"n_estimators":200, "max_depth":None, "min_samples_split":2},
    "SVC": {"C":1.0, "kernel":"rbf", "probability":True},
    "GradientBoosting": {"n_estimators":200, "learning_rate":0.05, "max_depth":2},
    "ExtraTrees": {"n_estimators":300, "max_depth":None, "min_samples_split":2},
    "XGBoost": {"n_estimators":300, "learning_rate":0.05, "max_depth":3, "subsample":0.9, "colsample_bytree":0.9},
    "MLP": {"hidden_layer_sizes":(64,32), "alpha":0.0005, "max_iter":400},
    "KerasNN": {"hidden":(64,32), "dropout":0.2, "epochs":30, "batch_size":16}
}

PARAM_GRIDS = {
    "LogisticRegression": {"clf__C":[0.1,1,3]},
    "RandomForest": {"clf__n_estimators":[200,400], "clf__max_depth":[None,8,16]},
    "SVC": {"clf__C":[0.5,1,2], "clf__kernel":["rbf","poly"]},
    "GradientBoosting": {"clf__n_estimators":[200,400], "clf__learning_rate":[0.05,0.1]},
    "ExtraTrees": {"clf__n_estimators":[200,400], "clf__max_depth":[None,8,16]},
    "XGBoost": {"clf__n_estimators":[200,400], "clf__max_depth":[3,5], "clf__learning_rate":[0.05,0.1]},
    "MLP": {"clf__hidden_layer_sizes":[(64,32),(128,64)], "clf__alpha":[0.0001,0.0005]},
    "KerasNN": {}
}
