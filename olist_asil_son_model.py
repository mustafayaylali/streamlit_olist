import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_libraries.data_prep import *
from my_libraries.eda import *
import time
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler


def read_data():
    start = time.time()

    customers = pd.read_csv("datasets/olist_customers_dataset.csv")
    geolocation = pd.read_csv("datasets/olist_geolocation_dataset.csv")
    order_items = pd.read_csv("datasets/olist_order_items_dataset.csv")
    orders = pd.read_csv("datasets/olist_orders_dataset.csv")
    product = pd.read_csv("datasets/olist_products_dataset.csv")
    sellers = pd.read_csv("datasets/olist_sellers_dataset.csv")
    end = time.time()
    print(f"Verileri okuma süresi: {round((end - start), 2)} saniye")
    return orders, customers, geolocation, sellers, order_items, product

orders_, customers, geolocation, sellers, order_items, product = read_data()

def orders_merge(merge_data):
    start = time.time()

    global customers
    global geolocation
    global sellers
    global order_items
    global product

    # Order Items
    order_items = pd.merge(order_items, product[
        ["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
                           on="product_id")
    # order_items["hacim"] = order_items["product_length_cm"] * order_items["product_height_cm"] * order_items[
    #     "product_width_cm"]

    order_items = order_items.groupby(["order_id", "seller_id", "product_id"])[
        ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]].sum()
    order_items.reset_index(inplace=True)

    merge_data = pd.merge(merge_data, order_items[
        ["order_id", "product_id", "seller_id", "product_weight_g", "product_length_cm", "product_height_cm",
         "product_width_cm"]], on="order_id")

    # Customers
    merge_data = pd.merge(merge_data,
                          customers[["customer_id", "customer_city", "customer_state", "customer_zip_code_prefix"]],
                          on="customer_id")

    # Geolocation - Customers
    geolocation = geolocation.groupby("geolocation_zip_code_prefix").agg(
        {"geolocation_lat": "mean", "geolocation_lng": "mean"})
    geolocation.reset_index(inplace=True)

    merge_data = pd.merge(merge_data, geolocation, left_on="customer_zip_code_prefix",
                          right_on="geolocation_zip_code_prefix")
    merge_data.rename(
        columns={"geolocation_lat": "customer_geolocation_lat", "geolocation_lng": "customer_geolocation_lng"},
        inplace=True)

    # Sellers
    merge_data = pd.merge(merge_data, sellers[["seller_id", "seller_city", "seller_state", "seller_zip_code_prefix"]],
                          on="seller_id")

    # Geolocation - Sellers
    merge_data = pd.merge(merge_data, geolocation, left_on="seller_zip_code_prefix",
                          right_on="geolocation_zip_code_prefix")
    merge_data.rename(
        columns={"geolocation_lat": "seller_geolocation_lat", "geolocation_lng": "seller_geolocation_lng"},
        inplace=True)

    end = time.time()
    print(f"Merge İşlemleri Tamamlanma Süresi: {round((end - start), 2)} saniye")

    return merge_data

orders_ = orders_merge(orders_)
orders_.head(2)
orders_.shape

def categorizing(data, col, target_col, num_of_categories):
    labels = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"]
    mean_data = pd.DataFrame(data.groupby(col)[target_col].mean()).reset_index()

    mean_data["label"] = pd.qcut(mean_data[target_col], num_of_categories, labels=labels[:num_of_categories])

    data = pd.merge(data, mean_data[[col, "label"]], on=col)
    data.rename(columns={"label": f"categorical_{col}"}, inplace=True)
    return data, mean_data

def assign_date(x):
    if x["day_name"] == "Monday":
        return "0"
    elif x["day_name"] == "Tuesday":
        return "0"
    elif x["day_name"] == "Wednesday":
        return "0"
    elif x["day_name"] == "Thursday":
        return "0"
    elif x["day_name"] == "Friday":
        return "0"
    elif x["day_name"] == "Saturday":
        return "1"
    elif x["day_name"] == "Sunday":
        return "1"

# Sınırların belirlenme sebebi Olist'in kendi dokümanı: https://ajuda.olist.com/hc/pt-br/articles/4409253877908
def boyut_sinirla(data):
    print(f"Before Shape: {data.shape}")
    data = data[(data["product_length_cm"] >= 16) &
                (data["product_length_cm"] <= 100) &
                (data["product_height_cm"] >= 2) &
                (data["product_height_cm"] <= 100) &
                (data["product_width_cm"] >= 11) &
                (data["product_width_cm"] <= 100) &
                (data["product_weight_g"] >= 50) &
                (data["product_weight_g"] <= 30000)].reset_index(drop=True)
    print(f"After Shape: {data.shape}")
    return data

def only_one_product(data):
    print(f"Before Shape: {data.shape}")
    # Sadece bir ürün olan siparişler
    index = data["order_id"].value_counts()[data["order_id"].value_counts() == 1].index

    data = data[data.order_id.isin(index)]
    data.reset_index(drop=True, inplace=True)
    data.shape
    print(f"After Shape: {data.shape}")
    return data

def calculate_desi(data):
    data["desi"] = data["hacim"] / 3000
    # Kaynak: https://www.suratkargo.com.tr/BilgiHizmetleri/GonderiStandartlarimiz/TasimaKurallari
    data["desi_kategori"] = "dosya"
    data.loc[(data["desi"] >= 0.51) & (data["desi"] < 1), "desi_kategori"] = "mi"
    data.loc[(data["desi"] >= 1) & (data["desi"] < 20), "desi_kategori"] = "paket"
    data.loc[(data["desi"] >= 20) & (data["desi"] < 100), "desi_kategori"] = "koli"
    data.loc[data["desi"] >= 100, "desi_kategori"] = "palet"
    return data

def data_preprocessing(data):
    start = time.time()
    global customers
    global geolocation
    global sellers
    global order_items
    global product

    # Sadece tamamlanan siparişler
    data = data[data["order_status"] == "delivered"]
    data.reset_index(drop=True, inplace=True)

    # Datetime veri tipinde olması gereken sütunların veri tipini ayarlıyorum
    for col in data.columns[3:8]:
        data[col] = pd.to_datetime(data[col])

    # Eksik Değerlerin Silinmesi
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)

    # Sadece bir ürün olan siparişler
    data = only_one_product(data)

    # Teslim edilme tarihi siparişin kabul tarihinden önce olan siprişlerin silinmesi
    aykiri_ix = data[data["order_approved_at"] >= data["order_delivered_customer_date"]].index
    data = data[~data.index.isin(aykiri_ix)]
    data.reset_index(drop=True, inplace=True)

    # Kargo paket boyutlarının sınırlarının belirlenmesi.
    # Sebebi Olist'in kendi dokümanı: https://ajuda.olist.com/hc/pt-br/articles/4409253877908
    data = boyut_sinirla(data)

    end = time.time()
    print(f"Ön İşlemlerin Tamamlanma Süresi: {round((end - start), 2)} saniye")
    return data

def f_engineering(data):
    start = time.time()
    global customers
    global geolocation
    global sellers
    global order_items
    global product

    # Ürünlerin hacmi
    data["hacim"] = data["product_length_cm"] * data["product_height_cm"] * data["product_width_cm"]

    # Ürünlerin desi bilgilerinin kategorikleştirilmesi
    data = calculate_desi(data)

    # Gün ismi ve ay değerlerini yeni değişken ekliyorum
    data["day_name"] = data["order_purchase_timestamp"].apply(lambda x: x.day_name())
    data["purchase_month"] = data["order_purchase_timestamp"].apply(lambda x: x.month)


    # Sipariş verildikten sonra müşteriye ulaşana kadar geçen süre (gün cinsinden)
    data["order_completion_day"] = [
        (pd.to_datetime(d.strftime("%Y-%m-%d")) - pd.to_datetime(t.strftime("%Y-%m-%d"))).days
        for d, t in zip(data["order_delivered_customer_date"], data["order_purchase_timestamp"])]

    # Satıcı ile müşterinin birbirleriyle olan öklid uzaklıkları
    data["euclidean_distance"] = (np.sqrt(((data["customer_geolocation_lat"] - data["seller_geolocation_lat"]) ** 2) +
                                          ((data["customer_geolocation_lng"] - data["seller_geolocation_lng"]) ** 2)))

    # Değişkenlerin kategorikleştirilmesi
    data, customer_city_mean_data = categorizing(data, "customer_city", "order_completion_day", 6)
    data, seller_city_mean_data = categorizing(data, "seller_city", "order_completion_day", 6)
    data, seller_id_mean_data = categorizing(data, "seller_id", "order_completion_day", 6)
    data, seller_state_mean_data = categorizing(data, "seller_state", "order_completion_day", 6)

    # Haftaiçi haftasonu ayrımının yapılması
    data = data.assign(wknd_or_not=data.apply(assign_date, axis=1))

    end = time.time()
    print(f"Özellik Mühendisliği tamamlanma süresi: {round((end - start), 2)} saniye")
    return data, customer_city_mean_data, seller_city_mean_data, seller_id_mean_data, seller_state_mean_data

orders_ = data_preprocessing(orders_)
orders_, customer_city_mean_data, seller_city_mean_data, seller_id_mean_data, seller_state_mean_data = f_engineering(orders_)
orders_.head(2)
orders_["order_id"].nunique()

def data_k_means(data):
    from sklearn.cluster import KMeans

    data = data[["order_completion_day", "euclidean_distance", "hacim", "product_weight_g", "order_id"]]
    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(data, car_th=25)
    num_cols.remove("order_completion_day")
    ss = StandardScaler()
    for col in num_cols:
        data[col] = ss.fit_transform(data[[col]])

    ocd_mean = data[data["order_completion_day"] < 20].groupby("order_completion_day").agg(
        {"euclidean_distance": "mean", "hacim": "mean", "product_weight_g": "mean",
         "order_id": "count"}).rename(columns={"order_id": "count"})

    # 0 günde tamamlanmış 1 siparişin dışarıda bırakılması.
    ocd_mean = ocd_mean[~(ocd_mean.index == 0)]

    model = KMeans(n_clusters=6, random_state=6)
    # fit the model
    model.fit(ocd_mean[["euclidean_distance", "hacim", "product_weight_g"]])
    # assign a cluster to each example
    yhat = model.predict(ocd_mean[["euclidean_distance", "hacim", "product_weight_g"]])

    ocd_mean["cluster"] = yhat
    return ocd_mean

data_k_means(orders_)

def target_variable_tags(data):
    data.loc[data["order_completion_day"] >= 12, "target"] = "12+"

    data.loc[(data["order_completion_day"] == 1) |
             (data["order_completion_day"] == 2) |
             (data["order_completion_day"] == 3), "target"] = "1-3"

    data.loc[(data["order_completion_day"] == 4) | (data["order_completion_day"] == 5), "target"] = "4-5"

    data.loc[(data["order_completion_day"] == 6) | (data["order_completion_day"] == 7), "target"] = "6-7"

    data.loc[(data["order_completion_day"] == 8) |
             (data["order_completion_day"] == 9) |
             (data["order_completion_day"] == 10) |
             (data["order_completion_day"] == 11), "target"] = "8-11"

    data["target"] = [str(i) for i in data["target"]]
    data = data[~(data["target"] == "nan")]
    data.reset_index(drop=True, inplace=True)

    return data
orders_ = target_variable_tags(orders_)
orders_.head(2)
orders = orders_.copy()

def delete_outlier_values(dataframe, col):
    # Sınırların belirlenmesi
    low_limit, up_limit = outlier_thresholds(dataframe, col, q1=0.25, q3=0.75)

    # Aykırı Değerlerin Silinmesi
    dataframe = dataframe[dataframe[col] < up_limit]
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe, low_limit, up_limit

def standard_scaling(dataframe, col):
    # Standard Scaler
    ss = StandardScaler().fit(dataframe[[col]])
    dataframe[col] = ss.transform(dataframe[[col]])
    return dataframe, ss

def remove_target_outlier_values(data):
    dict = {}
    for i in data["target"].unique():
        ed_low_limit, ed_up_limit = outlier_thresholds(data[data["target"] == i], "euclidean_distance", q1=0.25, q3=0.75)
        pw_low_limit, pw_up_limit = outlier_thresholds(data[data["target"] == i], "product_weight_g", q1=0.25, q3=0.75)
        h_low_limit, h_up_limit = outlier_thresholds(data[data["target"] == i], "hacim", q1=0.25, q3=0.75)

        print(f"Order Completion Day: {i}")
        print(f"Before: {data.shape}")
        aykiri_ix = data[(data["target"] == i) & (data["euclidean_distance"] > ed_up_limit)].index
        data = data[~data.index.isin(aykiri_ix)]
        aykiri_ix = data[(data["target"] == i) & (data["product_weight_g"] > pw_up_limit)].index
        data = data[~data.index.isin(aykiri_ix)]
        aykiri_ix = data[(data["target"] == i) & (data["hacim"] > h_up_limit)].index
        data = data[~data.index.isin(aykiri_ix)]

        # Bütün sınıf etiketleri için aykırı değer sınırlarını döner
        dict[i] = [ed_low_limit, ed_up_limit, pw_low_limit, pw_up_limit, h_low_limit, h_up_limit]
        print(f"After: {data.shape}")
        print(f"{dict[i]}\n")
    return data, dict

def double_check_after_fe(data):
    # Aykırı Değerler
    data, ocd_low_limit, ocd_up_limit = delete_outlier_values(data, "order_completion_day")
    # Diğer aykırı değerlere harici olarak bakılacak.

    # Standart Scaling
    data, weight_ss = standard_scaling(data, "product_weight_g")
    data, hacim_ss = standard_scaling(data, "hacim")
    data, desi_ss = standard_scaling(data, "desi")
    data, ed_ss = standard_scaling(data, "euclidean_distance")

    # Etiketlere göre öklit uzaklığı, hacim ve ağırlığın aykırı değerlerinin silinmesi.
    data, oulier_dict = remove_target_outlier_values(data)

    # One-Hot Encoding
    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(data, car_th=25)
    cat_cols.remove("target")
    cat_cols.remove("seller_state")
    data = one_hot_encoder(data, cat_cols, drop_first=True)
    return data, oulier_dict, ocd_low_limit, ocd_up_limit, weight_ss, hacim_ss, desi_ss, ed_ss

orders, tag_outlier_dict, ocd_low_limit, ocd_up_limit, weight_ss, hacim_ss, desi_ss, ed_ss = double_check_after_fe(orders)
orders.head(2)


def modele_hazirlik(data):
    data = data.drop(columns=["order_id", 'customer_id', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                                'product_id', 'seller_id', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'customer_city',
                                'customer_state', 'customer_zip_code_prefix', 'geolocation_zip_code_prefix_x', 'customer_geolocation_lat',
                                'customer_geolocation_lng', 'seller_city', 'seller_state', 'seller_zip_code_prefix',
                                'geolocation_zip_code_prefix_y', 'seller_geolocation_lat', 'seller_geolocation_lng',
                                'order_completion_day'])

    return data

data = modele_hazirlik(orders)
data.head()
data.shape

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns


models = []
# models.append(('XGBoost', XGBClassifier(random_state=55)))
models.append(('LightGBM', LGBMClassifier(random_state=55)))
models.append(('RandomForest', RandomForestClassifier(random_state=55)))


train, test = train_test_split(data, random_state=55, test_size=0.20)
X_train = train.drop(columns = ["target", "order_delivered_customer_date", "order_estimated_delivery_date"])
y_train = train["target"]
X_test = test.drop(columns = ["target", "order_delivered_customer_date", "order_estimated_delivery_date"])
y_test = test["target"]

X_train.shape
y_train.shape
X_test.shape
y_test.shape
#
# [19:41] mustafayaylali_ (Konuk)
# n_estimators=scipy.stats.randint(50,400)
# max_depth=scipy.stats.randint(2,10)
# lambda_l1=np.arange(0,0.8,0.05)
# lambda_l2=np.arange(0,0.8,0.05)
# extra_trees=[True,False]
# subsample=np.arange(0.3,1.0,0.05)
# bagging_freq=scipy.stats.randint(1,100)
# colsample_bytree=np.arange(0.3,1.0,0.05)
# num_leaves=scipy.stat  s.randint(10,100)
# boosting=['gbdt','dart']
# drop_rate=np.arange(0.1,0.8,0.1)
# skip_drop=np.arange(0,0.7,0.1)
# learning_rate=[0.00001,0.0001,0.001,0.01,0.015,0.02,0.05,0.1,0.15,0.2]
# params = {'n_estimators':n_estimators,'max_depth':max_depth, 'subsample':subsample,
#           'colsample_bytree':colsample_bytree,
#           'learning_rate':learning_rate,
#           'num_leaves':num_leaves,
#           'boosting':boosting,
#           'extra_trees':extra_trees,
#           'lambda_l1':lambda_l1,
#           'lambda_l2':lambda_l2,
#           'bagging_freq':bagging_freq,
#           'drop_rate':drop_rate}
#
#
# params = list(ParameterSampler(params, n_iter=2000))
#



# for name, model in models:
#     model_name = model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f"{name}: {accuracy_score(y_test, y_pred)}")
#
# lgb_params = {"learning_rate": [0.1, 0.01, 0.05],
#               "n_estimators": [100, 200, 500],
#               "max_depth": [-1, 5, 10]}
#
#
# gs_best = GridSearchCV(LGBMClassifier(random_state=55), lgb_params, cv=5, n_jobs=-1, verbose=False).fit(X_train, y_train)
#
# gs_best.best_params_s
#
# final_model = LGBMClassifier(random_state=55).set_params(**gs_best.best_params_).fit(X_train, y_train)
#
# y_pred = final_model.predict(X_test)
# y_prob = final_model.predict_proba(X_test)[:, 1]
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# pd.DataFrame(confusion_matrix(y_test, y_pred), index=np.sort(y_test.unique()), columns=np.sort(y_test.unique()))
#
# data["order_delivered_customer_date"] = [pd.to_datetime(time.strftime("%Y-%m-%d")) for time in data["order_delivered_customer_date"]]
# print(f"Orjinal veri setindeki tahminlerin doğruluk oranı: {accuracy_score(data['order_delivered_customer_date'], data['order_estimated_delivery_date'])}")
#
#
# def plot_importance(model, features, num=len(train), save=False):
#     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
#     plt.figure(figsize=(10, 10))
#     sns.set(font_scale=1)
#     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
#                                                                       ascending=False)[0:num])
#     plt.title('Features')
#     plt.tight_layout()
#     plt.show()
#     if save:
#         plt.savefig('importances.png')
#
# plot_importance(final_model, X_train)


"""               precision    recall  f1-score   support
         1-3       0.50      0.68      0.57       984
         12+       0.66      0.80      0.72      5299
         4-5       0.40      0.38      0.39      1441
         6-7       0.40      0.23      0.29      2156
        8-11       0.45      0.38      0.41      3520
    accuracy                           0.54     13400
   macro avg       0.48      0.49      0.48     13400
weighted avg       0.52      0.54      0.52     13400"""


"""
      1-3   12+  4-5  6-7  8-11
1-3   667    11  219   59    28
12+    54  4213  115  196   721
4-5   239   136  553  166   347
6-7   217   577  296  499   567
8-11  161  1484  207  315  1353

"""

from dateutil.relativedelta import relativedelta


# Uzunluk, boy, en product tablosundan çek
# Satıcı ve müşteri için enlem ve boylam bilgilerini çek.
# TODO customer_city_mean_data, seller_city_mean_data, seller_id_mean_data, seller_state_mean_data merge işlemleri
# weight_ss, hacim_ss, desi_ss, ed_ss scale işlemleri

def data_pipeline(data):
    df = pd.DataFrame(columns=X_train.columns)
    df["product_weight_g"] = data["product_weight_g"]
    df["hacim"] = data["hacim"]
    df["desi"] = data["desi"]
    df["purchase_month"] = data["purchase_month"]
    df["euclidean_distance"] = data["euclidean_distance"]
    df[["desi_kategori_mi", "desi_kategori_paket"]] = [[1, 0] if value == "mi" else [0, 1] for value in data["desi_kategori"]]

    df[['day_name_Monday', 'day_name_Saturday', 'day_name_Sunday', 'day_name_Thursday',
        'day_name_Tuesday', 'day_name_Wednesday']] = [[1, 0, 0, 0, 0, 0] if day == "Monday" else
                                                          [0, 1, 0, 0, 0, 0] if day == "Saturday" else
                                                          [0, 0, 1, 0, 0, 0] if day == "Sunday" else
                                                          [0, 0, 0, 1, 0, 0] if day == "Thursday" else
                                                          [0, 0, 0, 0, 1, 0] if day == "Tuesday" else
                                                          [0, 0, 0, 0, 0, 1] for day in data["day_name"]]

    df['wknd_or_not_1'] = data["wknd_or_not"]

    df[['categorical_customer_city_iki', 'categorical_customer_city_üç',
        'categorical_customer_city_dört', 'categorical_customer_city_beş',
        'categorical_customer_city_altı']] = [[1, 0, 0, 0, 0] if value == "iki" else
                                                          [0, 1, 0, 0, 0] if value == "üç" else
                                                          [0, 0, 1, 0, 0] if value == "dört" else
                                                          [0, 0, 0, 1, 0] if value == "beş" else
                                                          [0, 0, 0, 0, 1] for value in data["categorical_customer_city"]]


    df[['categorical_seller_city_iki', 'categorical_seller_city_üç',
       'categorical_seller_city_dört', 'categorical_seller_city_beş',
        'categorical_seller_city_altı']] = [[1, 0, 0, 0, 0] if value == "iki" else
                                                          [0, 1, 0, 0, 0] if value == "üç" else
                                                          [0, 0, 1, 0, 0] if value == "dört" else
                                                          [0, 0, 0, 1, 0] if value == "beş" else
                                                          [0, 0, 0, 0, 1] for value in data["categorical_seller_city"]]

    df[['categorical_seller_id_iki', 'categorical_seller_id_üç',
        'categorical_seller_id_dört', 'categorical_seller_id_beş',
        'categorical_seller_id_altı']] = [[1, 0, 0, 0, 0] if value == "iki" else
                                            [0, 1, 0, 0, 0] if value == "üç" else
                                            [0, 0, 1, 0, 0] if value == "dört" else
                                            [0, 0, 0, 1, 0] if value == "beş" else
                                            [0, 0, 0, 0, 1] for value in data["categorical_seller_id"]]

    df[['categorical_seller_state_iki', 'categorical_seller_state_üç',
        'categorical_seller_state_dört', 'categorical_seller_state_beş',
        'categorical_seller_state_altı']] = [[1, 0, 0, 0, 0] if value == "iki" else
                                            [0, 1, 0, 0, 0] if value == "üç" else
                                            [0, 0, 1, 0, 0] if value == "dört" else
                                            [0, 0, 0, 1, 0] if value == "beş" else
                                            [0, 0, 0, 0, 1] for value in data["categorical_seller_state"]]

    return df

def pipeline_hazirlik(data):
    global product
    global geolocation
    
    # Hacim hesaplamak için merge
    data = pd.merge(data, product[["product_id", "product_weight_g", "product_length_cm", "product_height_cm",
                                   "product_width_cm"]], on="product_id")

    # Ürünlerin hacmi
    data["hacim"] = data["product_length_cm"] * data["product_height_cm"] * data["product_width_cm"]

    # Ürünlerin desi bilgisi    
    data = calculate_desi(data)

    # Zip codelar için ortalama enlem, boylam bilgileri
    geolocation = geolocation.groupby("geolocation_zip_code_prefix").agg(
        {"geolocation_lat": "mean", "geolocation_lng": "mean"})
    geolocation.reset_index(inplace=True)
    
    # Müşteri enlem, boylam bilgisi
    data = pd.merge(data, geolocation, left_on="zip_code", right_on="geolocation_zip_code_prefix")\
        .rename(columns={"geolocation_lat": "customer_geolocation_lat", "geolocation_lng": "customer_geolocation_lng"})

    # Satıcı enlem, boylam, şehir, bölge bilgisi
    data = pd.merge(data, sellers, on="seller_id")

    data = pd.merge(data, geolocation, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix")\
        .rename(columns={"geolocation_lat": "seller_geolocation_lat", "geolocation_lng": "seller_geolocation_lng"})

    # print(data.head(3))
    # Gün ismi ve ay değerlerini yeni değişken ekliyorum
    data["day_name"] = data["order_purchase_timestamp"].apply(lambda x: x.day_name())
    data["purchase_month"] = data["order_purchase_timestamp"].apply(lambda x: x.month)

    # Satıcı ile müşterinin birbirleriyle olan öklid uzaklıkları
    data["euclidean_distance"] = (np.sqrt(((data["customer_geolocation_lat"] - data["seller_geolocation_lat"]) ** 2) +
                                          ((data["customer_geolocation_lng"] - data["seller_geolocation_lng"]) ** 2)))
    

    data = pd.merge(data, customers[["customer_zip_code_prefix", "customer_city"]], left_on="zip_code", 
             right_on="customer_zip_code_prefix")
    

    data = pd.merge(data, customer_city_mean_data[["customer_city", "label"]], on="customer_city")\
        .rename(columns={"label": "categorical_customer_city"})
    
    data = pd.merge(data, seller_city_mean_data[["seller_city", "label"]], on="seller_city")\
        .rename(columns={"label": "categorical_seller_city"})
    
    data = pd.merge(data, seller_id_mean_data[["seller_id", "label"]], on="seller_id")\
        .rename(columns={"label": "categorical_seller_id"})
    
    data = pd.merge(data, seller_state_mean_data[["seller_state", "label"]], on="seller_state")\
        .rename(columns={"label": "categorical_seller_state"})
    

    # Haftaiçi haftasonu ayrımının yapılması
    data = data.assign(wknd_or_not=data.apply(assign_date, axis=1))

    # Standart Scaler
    data[["product_weight_g"]] = weight_ss.transform(data[["product_weight_g"]])
    data[["hacim"]] = hacim_ss.transform(data[["hacim"]])
    data[["desi"]] = desi_ss.transform(data[["desi"]])
    data[["euclidean_distance"]] = ed_ss.transform(data[["euclidean_distance"]])

    return data

def ml_pipeline(id_product,zip_code,id_seller):
    global orders
    #
    # id_product = str(input("Satın almak istediğiniz ürünün id bilgisini giriniz: "))
    # zip_code = int(input("Bulunduğunuz bölgenin zip kod bilgisini giriniz: "))
    # id_seller = str(input("Ürünü satın almak istediğiniz satıcının id bilgisini giriniz:"))
    #
    id_product = id_product
    zip_code = int(zip_code)
    id_seller = id_seller


    # id_product = "1e9e8ef04dbcff4541ed26657ea517e5"
    # zip_code = 83203
    # id_seller = "0015a82c2db000af6aaaf3ae2ecb0532"

    date = orders["order_purchase_timestamp"].max() + relativedelta(days=1)
    data = {"product_id": id_product, "zip_code": zip_code, "seller_id": id_seller, "order_purchase_timestamp": date}
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"])
    data = pd.DataFrame(data, index=[0])
    data = pipeline_hazirlik(data)
    data = data.sample(n=1).reset_index(drop=True)
    
    data = data.drop(columns=['product_id', 'zip_code', 'seller_id', 'order_purchase_timestamp', 'product_length_cm', 
                       'product_height_cm', 'product_width_cm', 'geolocation_zip_code_prefix_x', 
                       'customer_geolocation_lat', 'customer_geolocation_lng', 'seller_zip_code_prefix', 
                       'seller_city', 'seller_state', 'geolocation_zip_code_prefix_y', 'seller_geolocation_lat',
                       'seller_geolocation_lng', 'customer_zip_code_prefix', 'customer_city'])

    data = data_pipeline(data)

    data["wknd_or_not_1"] = data["wknd_or_not_1"].astype(int)

    #print(f"Ürünün Tahmini Teslimat Süresi (Tempo estimado de entrega do produto): {final_model.predict(data)[0]}")

    return data




