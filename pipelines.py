"""Pipeline hlavného (XGBoost) a referenčného (logistická regresia) modelu.
Všetky kroky predspracovania sú vnútri pipeline -> učia sa len na tréningovej časti."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import loguniform, uniform, randint
from common import SEED, CFG


class DropSparse(BaseEstimator, TransformerMixin):
    """Vyradí premenné s podielom chýbajúcich hodnôt > max_na (učí sa na tréningovej časti)
    a premenné bez variability."""
    def __init__(self, max_na=0.5):
        self.max_na = max_na
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        na = np.isnan(X).mean(0)
        sd = np.nanstd(X, 0)
        self.keep_ = np.where((na <= self.max_na) & (np.nan_to_num(sd) > 0))[0]
        return self
    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


def signed_log(X):
    X = np.asarray(X, float)
    return np.sign(X) * np.log1p(np.abs(X))


def make_lr():
    pipe = Pipeline([
        ('drop', DropSparse(CFG['max_na'])),
        ('log', FunctionTransformer(signed_log)),
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(solver='saga', max_iter=5000, l1_ratio=0.0,
                                   penalty='elasticnet', random_state=SEED)),
    ])
    grid = {'clf__C': loguniform(1e-3, 1e1), 'clf__l1_ratio': uniform(0, 1)}
    return pipe, grid


def make_xgb():
    pipe = Pipeline([
        ('drop', DropSparse(CFG['max_na'])),
        ('clf', XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              tree_method='hist', n_jobs=4, random_state=SEED)),
    ])
    grid = {'clf__n_estimators': randint(100, 600),
            'clf__learning_rate': loguniform(0.01, 0.2),
            'clf__max_depth': randint(2, 6),
            'clf__min_child_weight': randint(1, 10),
            'clf__subsample': uniform(0.6, 0.4),
            'clf__colsample_bytree': uniform(0.4, 0.6),
            'clf__reg_lambda': loguniform(0.1, 10)}
    return pipe, grid
