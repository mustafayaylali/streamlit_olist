import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



def plot_importance(model, features , save=False):
    """
    Model ve modelde yer alan değişkenlerin önem sırasını grafik olarak verir

    SAMPLE:
        independent_df = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

    example:
        plot_importance(rf_model, X_train)
    """
    num = len(features)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def check_models_classifiers(X,y,cv_num=3):
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier()),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=cv_num, scoring=["roc_auc"])
        print(f"AUC: {round(cv_results['test_roc_auc'].mean(), 4)} ({name}) ")

def check_models_regressor(X,y,cv_num=10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42, test_size=20)

    models = {('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              # ("CatBoost", CatBoostRegressor(verbose=False))
              }

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv_num, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        # print("\n################## "+name+" ###################")
        # regressor.fit(X_train,Y_train)
        # Y_train_pred = regressor.predict(X_train)
        # train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
        # print("Train RMSE:{}".format(train_rmse))
        # Y_test_pred = regressor.predict(X_test)
        # test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
        # print("Test RMSE:{}".format(test_rmse))


def find_best_models(X,y,cv_num=10,test_size=0.20):
    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [5, 8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [8, 15, 20],
                 "n_estimators": [200, 500, 1000]}

    xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                      "max_depth": [5, 8, 12, 20],
                      "n_estimators": [100, 200, 300, 500],
                      "colsample_bytree": [0.5, 0.8, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                       "n_estimators": [300, 500, 1500],
                       "colsample_bytree": [0.5, 0.7, 1]}

    regressors = [("CART", DecisionTreeRegressor(), cart_params),
                  ("RF", RandomForestRegressor(), rf_params),
                  ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
                  ('LightGBM', LGBMRegressor(), lightgbm_params)]

    best_models = {}
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=95, test_size=test_size)

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv_num, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        #gs_best = GridSearchCV(regressor, params, cv=cv_num, n_jobs=-1, verbose=False).fit(X, y)
        gs_best = GridSearchCV(regressor, params, cv=cv_num, n_jobs=-1, verbose=False).fit(X_train, Y_train)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=cv_num, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")


        # final_model.fit(X_train,Y_train)
        # Y_train_pred = final_model.predict(X_train)
        # train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
        # print("Train RMSE:{}".format(train_rmse))
        # Y_test_pred = final_model.predict(X_test)
        # test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
        # print("Test RMSE:{}".format(test_rmse))



        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model

    return best_models

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def reg_model(df, Y, algo, test_size=0.20):
    X = df.drop(Y, axis=1)
    Y = df[[Y]]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=95)
    model = algo.fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    print(df.name)
    print(type(model).__name__)
    print("Train RMSE: {}".format(train_rmse))

    Y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    print("Test RMSE: {}".format(test_rmse))
    print('###################################')
    return (df.name, type(model).__name__, train_rmse, test_rmse)


def model_tuning(df, Y, algo_cv, algo, grid, test_size=0.20, cv=10):
    X=df.drop(Y, axis=1)
    Y=df[[Y]]
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=95, test_size=test_size)
    if type(algo()).__name__=='LGBMRegressor':
        model=algo()
        model_cv=algo_cv(model, grid, cv=cv, n_jobs=-1, verbose=2)
        model_cv.fit(X_train, Y_train)
        model_tuned=LGBMRegressor(learning_rate=model_cv.best_params_['learning_rate'],
                         max_depth=model_cv.best_params_['max_depth'],
                         n_estimators=model_cv.best_params_['n_estimators'],
                         boosting_type=model_cv.best_params_['boosting_type'],
                         colsample_bytree=model_cv.best_params_['colsample_bytree'])
    elif type(algo()).__name__=='DecisionTreeRegressor':
        model=algo()
        model_cv=algo_cv(model, grid, cv=cv, n_jobs=-1, verbose=2)
        model_cv.fit(X_train, Y_train)
        model_tuned=DecisionTreeRegressor(min_samples_split=model_cv.best_params_['min_samples_split'],
                         max_depth=model_cv.best_params_['max_depth'])
    elif type(algo()).__name__ == 'XGBRegressor':
        model = algo()
        model_cv = algo_cv(model, grid, cv=cv, n_jobs=-1, verbose=2)
        model_cv.fit(X_train, Y_train)
        model_tuned = XGBRegressor(learning_rate=model_cv.best_params_['learning_rate'],
                                    max_depth=model_cv.best_params_['max_depth'],
                                    n_estimators=model_cv.best_params_['n_estimators'],
                                    colsample_bytree=model_cv.best_params_['colsample_bytree'])
    elif type(algo()).__name__ == 'RandomForestRegressor':
        model = algo()
        model_cv = algo_cv(model, grid, cv=cv, n_jobs=-1, verbose=2)
        model_cv.fit(X_train, Y_train)
        model_tuned = RandomForestRegressor(max_depth=model_cv.best_params_['max_depth'],
                                    max_features=model_cv.best_params_['max_features'],
                                    min_samples_split=model_cv.best_params_['min_samples_split'],
                                    n_estimators=model_cv.best_params_['n_estimators'])
    else:
        model_cv=algo_cv(alphas=grid, cv=cv)
        model_cv.fit(X_train, Y_train)
        model_tuned=algo(alpha=model_cv.alpha_)


    model_tuned.fit(X_train, Y_train)
    print(df.name)
    print(type(model_tuned).__name__)
    Y_train_pred=model_tuned.predict(X_train)
    train_rmse=np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    print("Train RMSE:{}".format(train_rmse))
    Y_test_pred=model_tuned.predict(X_test)
    test_rmse=np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    print("Test RMSE:{}".format(test_rmse))
    print('#####################')
    return (df.name, type(model_tuned).__name__, train_rmse, test_rmse)