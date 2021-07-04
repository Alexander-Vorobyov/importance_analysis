import pandas
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from statistics import mean
from hyperopt import hp, fmin, tpe
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import make_scorer
from operator import __sub__
from lightgbm.sklearn import LGBMRegressor
from functools import partial
import numpy
import shap
from eli5.sklearn import PermutationImportance
from random import uniform, randint, random
from functools import reduce
import copy
from itertools import chain
from yellowbrick.model_selection import learning_curve
from sklearn.preprocessing._data import StandardScaler, PowerTransformer

def wmape(y_true, y_pred):
    return sum(map(abs, map(__sub__, y_true, y_pred)))/sum(map(abs, y_true))

def get_x_y(path, sep, decimal, thousands, target, drop_cols=[]):
    X = pandas.read_csv(path, sep=sep, decimal=decimal, thousands=thousands).\
        drop(columns=drop_cols)
    Y = X.pop(target)
    return X, Y

def random_xgb():
    return XGBRegressor(**{'learning_rate': uniform(0.01, 0.2),
                           'min_child_weight': randint(1, 10),
                               'max_depth': randint(3, 8),
                               'gamma':random(),
                               'subsample': uniform(0.5, 1.0),
                               'colsample_bytree': uniform(0.5,1.0),
                               'n_jobs': -1
                               })
    
def random_lgbm():
    return LGBMRegressor(**{'learning_rate': uniform(0.01, 0.2),
                            'min_child_weight': randint(1, 10),
                               'max_depth': randint(3, 8),
                               'subsample': uniform(0.5, 1.0),
                               'colsample_bytree': uniform(0.5,1.0),
                               'n_jobs': -1
                               })


def get_ssms_importance(X, Y, evals):
    return list(zip(X.columns, reduce(lambda importance_total,importance_model: importance_total+importance_model, map(lambda model: numpy.sum(abs(shap.Explainer(model)(X).values), axis=1)*(1.0-mean(cross_val_score(model, X, Y, scoring=make_scorer(wmape), cv=TimeSeriesSplit(n_splits=5), n_jobs=-1))), chain((copy.deepcopy(random_xgb()).fit(X,Y) for _ in range(evals)),(copy.deepcopy(random_lgbm()).fit(X,Y) for _ in range(evals)))))))


def tune(model, X, Y, space, evals, cv, scoring):
    def objective(params):
        model.set_params(**params)
        return mean(cross_val_score(model, X, Y, cv=cv, scoring=scoring, n_jobs=-1))
    
    return fmin(objective, space, algo=tpe.suggest, max_evals=evals, show_progressbar=False)


def group_importances(importances, ftg):
    # ftg -> feature_to_group
    groupped = dict([(group, 0.0) for group in ftg.values()])
    
    for pair in importances:
        groupped[ftg[pair[0]]] += pair[1]
        
    return groupped

def get_shap_importance(model, X):
    return list(zip(X.columns,
                    numpy.sum(abs(shap.Explainer(model)(X).values), axis=1)))

def get_feature_permutation_importance(model, X, Y):
    return list(zip(X.columns, 
                    PermutationImportance(model).fit(X, Y).feature_importances_))

def dict_structure(dict_with_numbers):
    sum_of_numbers = sum(list(dict_with_numbers.values()))
    return dict([(key, dict_with_numbers[key]*100/sum_of_numbers)\
                  for key in dict_with_numbers.keys()])

files = ["DAP_PLANT.csv", "DCBT_PLANT.csv", "DGH_PLANT.csv"]
targets = ["OP RUB SPT", "OP RUB CBT", "OP RUB GH"]

for file, target in zip(files, targets):
    print(f"FILE = {file}")
    X, Y = get_x_y(file, ';', ',', ' ', target, drop_cols=["date"])
    xgb = XGBRegressor(**tune(XGBRegressor(),
                              X,
                              Y,
                              {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                               'min_child_weight': 1+hp.randint('min_child_weight', 15),
                               'max_depth': 3+hp.randint('max_depth', 8),
                               'gamma':hp.uniform('gamma', 0.001, 0.999),
                               'subsample': hp.uniform('subsample', 0.5, 1.0),
                               'colsample_bytree':hp.uniform('colsample_bytree', 0.5,1.0),
                               },
                              1000,
                              TimeSeriesSplit(n_splits=5),
                              make_scorer(wmape)))
    print("XGB:", end=' ')
    print(mean(cross_val_score(xgb, X, Y, cv=TimeSeriesSplit(n_splits=5), scoring=make_scorer(wmape))))
    learning_curve(xgb, X, Y, scoring=make_scorer(wmape))
    lgbm = LGBMRegressor(**tune(LGBMRegressor(),
                              X,
                              Y,
                              {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                               'min_child_weight': 1+hp.randint('min_child_weight', 15),
                               'max_depth': 3+hp.randint('max_depth', 8),
                               'subsample': hp.uniform('subsample', 0.5, 1.0),
                               'colsample_bytree':hp.uniform('colsample_bytree', 0.5,1.0),
                               },
                              1000,
                              TimeSeriesSplit(n_splits=5),
                              make_scorer(wmape)))
    print("LGBM:", end=' ')
    print(mean(cross_val_score(lgbm, X, Y, cv=TimeSeriesSplit(n_splits=5), scoring=make_scorer(wmape))))
    learning_curve(lgbm, X, Y, scoring=make_scorer(wmape))
    xgb.fit(X, Y)
    lgbm.fit(X, Y)
    
    learning_curve(XGBRegressor(), X, Y, scoring=make_scorer(wmape))
    learning_curve(LGBMRegressor(), X, Y, scoring=make_scorer(wmape))
    
    feature_to_group = pandas.read_excel("Расшифровка важность.xlsx")
    feature_to_group = dict(zip(feature_to_group["Признак"].values,
                                feature_to_group["Группа"].values))
    
    groups = list(set(feature_to_group.values()))
    
    groupper = partial(group_importances, ftg=feature_to_group)
    
    xgb_shap = dict_structure(groupper(get_shap_importance(xgb, X)))
    lgbm_shap = dict_structure(groupper(get_shap_importance(lgbm, X)))
    xgbraw_shap = dict_structure(groupper(get_shap_importance(XGBRegressor().fit(X,Y), X)))
    lgbmraw_shap = dict_structure(groupper(get_shap_importance(LGBMRegressor().fit(X,Y), X)))
    xgb_eli = dict_structure(groupper(get_feature_permutation_importance(xgb, X, Y)))
    lgbm_eli = dict_structure(groupper(get_feature_permutation_importance(lgbm, X, Y)))
    xgbraw_eli = dict_structure(groupper(get_feature_permutation_importance(XGBRegressor().fit(X,Y), X, Y)))
    lgbraw_eli = dict_structure(groupper(get_feature_permutation_importance(LGBMRegressor().fit(X,Y), X, Y)))
    print("SSMS")
    ssms = dict_structure(groupper(get_ssms_importance(X, Y, 1000)))
    
    
    
    output = pandas.DataFrame({"Фактор": groups, 
                                "Важность XGB [picked], shap, %": [xgb_shap[f] for f in groups],
                                "Важность LGBM [picked], shap, %": [lgbm_shap[f] for f in groups],
                                "Важность XGB [raw], shap, %": [xgbraw_shap[f] for f in groups],
                                "Важность LGBM [raw], shap, %": [lgbmraw_shap[f] for f in groups],
                                "Важность XGB [picked], eli, %": [xgb_eli[f] for f in groups],
                                "Важность LGBM [picked], eli, %": [lgbm_eli[f] for f in groups],
                                "Важность XGB [raw], eli, %": [xgbraw_eli[f] for f in groups],
                                "Важность LGBM [raw], eli, %": [lgbraw_eli[f] for f in groups],
                                "Важность SSMS": [ssms[f] for f in groups]
                                })
    output.to_excel(f"Результаты {file[:-4]}.xlsx", index=False)
    