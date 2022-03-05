def read_data():
    """
    Veri setlerini okuyup bir dataframe de tutar.

    Parameters
    ------

    Returns
    ------
        orders: dataframe
            Sipariş detaylarının tutulduğu tablo
        customers: dataframe
            Müşteri bilgilerinin tutulduğu tablo
        geolocation: dataframe
            Konum bilgilerinin tutulduğu tablo
        sellers: dataframe
            Satıcı bilgilerinin tutulduğu tablo
        order_items: dataframe
            Sipariş&Ürün bilgilerinin tutulduğu tablo
        product: dataframe
            Ürün detaylarının tutulduğu tablo

    Examples
    ------
        orders_, customers, geolocation, sellers, order_items, product = read_data()
    """
    import time
    import pandas as pd
    start = time.time()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    customers = pd.read_csv("datasets/olist_customers_dataset.csv")
    geolocation = pd.read_csv("datasets/olist_geolocation_dataset.csv")
    order_items = pd.read_csv("datasets/olist_order_items_dataset.csv")
    orders = pd.read_csv("datasets/olist_orders_dataset.csv")
    product = pd.read_csv("datasets/olist_products_dataset.csv")
    sellers = pd.read_csv("datasets/olist_sellers_dataset.csv")
    end = time.time()
    print(f"Verileri okuma süresi: {round((end - start), 2)} saniye\n")
    return orders, customers, geolocation, sellers, order_items, product

def calculate_desi(data):
    """
        Ürün desi bilgisini hesaplar ve kategorize eder.

        Parameters
        ------
            data: dataframe
                Desi hesaplaması yapılcak tablo.
        Returns
        ------
            data: dataframe
                Desi hesaplaması yapılmış tablo.
        Examples
        ------
            data = calculate_desi(data)
        Notes
        ------
            Desi; kargo işlemlerinde hacimsel ağırlık anlamına gelen kelimedir.
            Özellikle taşımacılıkta kullanılan bir terimdir.
            Kargo ile gönderilecek kolinin desisini  hesaplamak için en, boy ve yükseklik birbiri ile çarpılır ve
            3000’e bölünür. Bu ölçü birimi kargo firmaları tarafından ürünün kargo bedelinin hesaplanmasında kullanılır.
            Taşımacılıkta, taşınan yükün ağırlığı ile birlikte hacmi yani kapladığı alan da önem göstermektedir.
    """


    data["desi"] = data["hacim"] / 3000
    # Kaynak: https://www.suratkargo.com.tr/BilgiHizmetleri/GonderiStandartlarimiz/TasimaKurallari
    data["desi_kategori"] = "dosya"
    data.loc[(data["desi"] >= 0.51) & (data["desi"] < 1), "desi_kategori"] = "mi"
    data.loc[(data["desi"] >= 1) & (data["desi"] < 20), "desi_kategori"] = "paket"
    data.loc[(data["desi"] >= 20) & (data["desi"] < 100), "desi_kategori"] = "koli"
    data.loc[data["desi"] >= 100, "desi_kategori"] = "palet"
    return data

def assign_date(x):
    """
        Belirtilen gün haftasonu mu olup olmadığını etiketleyen yardımcı fonksiyon

        Parameters
        ------
            x: dataframe
                Yeni sütun eklenecek hedef tablo
        Returns
        ------
            0 veya 1 değerlerini atar.

        Examples
        ------
            data = data.assign(wknd_or_not=data.apply(assign_date, axis=1))
    """

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

def data_pipeline(data):
    import pandas as pd
    df = pd.DataFrame(columns=['product_weight_g', 'hacim', 'desi', 'purchase_month',
                               'euclidean_distance', 'desi_kategori_mi', 'desi_kategori_paket',
                               'day_name_Monday', 'day_name_Saturday', 'day_name_Sunday',
                               'day_name_Thursday', 'day_name_Tuesday', 'day_name_Wednesday',
                               'wknd_or_not_1', 'categorical_customer_city_iki',
                               'categorical_customer_city_üç', 'categorical_customer_city_dört',
                               'categorical_customer_city_beş', 'categorical_customer_city_altı',
                               'categorical_seller_city_iki', 'categorical_seller_city_üç',
                               'categorical_seller_city_dört', 'categorical_seller_city_beş',
                               'categorical_seller_city_altı', 'categorical_seller_id_iki',
                               'categorical_seller_id_üç', 'categorical_seller_id_dört',
                               'categorical_seller_id_beş', 'categorical_seller_id_altı',
                               'categorical_seller_state_iki', 'categorical_seller_state_üç',
                               'categorical_seller_state_dört', 'categorical_seller_state_beş',
                               'categorical_seller_state_altı'])
    df["product_weight_g"] = data["product_weight_g"]
    df["hacim"] = data["hacim"]
    df["desi"] = data["desi"]
    df["purchase_month"] = data["purchase_month"]
    df["euclidean_distance"] = data["euclidean_distance"]
    df[["desi_kategori_mi", "desi_kategori_paket"]] = [[1, 0] if value == "mi" else [0, 1] for value in
                                                       data["desi_kategori"]]

    df[['day_name_Monday', 'day_name_Saturday', 'day_name_Sunday', 'day_name_Thursday',
        'day_name_Tuesday', 'day_name_Wednesday']] = [[1, 0, 0, 0, 0, 0] if day == "Monday" else
                                                      [0, 1, 0, 0, 0, 0] if day == "Saturday" else
                                                      [0, 0, 1, 0, 0, 0] if day == "Sunday" else
                                                      [0, 0, 0, 1, 0, 0] if day == "Thursday" else
                                                      [0, 0, 0, 0, 1, 0] if day == "Tuesday" else
                                                      [0, 0, 0, 0, 0, 1] for day in data["day_name"]]

    df['wknd_or_not_1'] = data["wknd_or_not"]
    df["wknd_or_not_1"] = df["wknd_or_not_1"].astype(int)

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
    import pickle
    import pandas as pd
    import numpy as np
    orders, customers, geolocation, sellers, order_items, product = read_data()

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
    data = pd.merge(data, geolocation, left_on="zip_code", right_on="geolocation_zip_code_prefix") \
        .rename(columns={"geolocation_lat": "customer_geolocation_lat", "geolocation_lng": "customer_geolocation_lng"})

    # Satıcı enlem, boylam, şehir, bölge bilgisi
    data = pd.merge(data, sellers, on="seller_id")

    data = pd.merge(data, geolocation, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix") \
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

    customer_city_mean_data = pd.read_csv("Helper Datasets/customer_city_mean_data.csv")
    seller_city_mean_data = pd.read_csv("Helper Datasets/seller_city_mean_data.csv")
    seller_id_mean_data = pd.read_csv("Helper Datasets/seller_id_mean_data.csv")
    seller_state_mean_data = pd.read_csv("Helper Datasets/seller_state_mean_data.csv")

    data = pd.merge(data, customer_city_mean_data[["customer_city", "label"]], on="customer_city") \
        .rename(columns={"label": "categorical_customer_city"})

    data = pd.merge(data, seller_city_mean_data[["seller_city", "label"]], on="seller_city") \
        .rename(columns={"label": "categorical_seller_city"})

    data = pd.merge(data, seller_id_mean_data[["seller_id", "label"]], on="seller_id") \
        .rename(columns={"label": "categorical_seller_id"})

    data = pd.merge(data, seller_state_mean_data[["seller_state", "label"]], on="seller_state") \
        .rename(columns={"label": "categorical_seller_state"})

    print(data.head())

    # Haftaiçi haftasonu ayrımının yapılması
    data = data.assign(wknd_or_not=data.apply(assign_date, axis=1))

    # Standart Scaler
    weight_ss = pickle.load(open("Pickles/weight_ss.sav", 'rb'))
    hacim_ss = pickle.load(open("Pickles/hacim_ss.sav", 'rb'))
    desi_ss = pickle.load(open("Pickles/desi_ss.sav", 'rb'))
    ed_ss = pickle.load(open("Pickles/ed_ss.sav", 'rb'))

    data[["product_weight_g"]] = weight_ss.transform(data[["product_weight_g"]])
    data[["hacim"]] = hacim_ss.transform(data[["hacim"]])
    data[["desi"]] = desi_ss.transform(data[["desi"]])
    data[["euclidean_distance"]] = ed_ss.transform(data[["euclidean_distance"]])

    return data

def ml_pipeline(id_product, id_seller, zip_code,purchase_date):
    from dateutil.relativedelta import relativedelta
    import pickle
    import pandas as pd
    orders, customers, geolocation, sellers, order_items, product = read_data()

    # id_product = str(input("Satın almak istediğiniz ürünün id bilgisini giriniz: "))
    # zip_code = int(input("Bulunduğunuz bölgenin zip kod bilgisini giriniz: "))
    # id_seller = str(input("Ürünü satın almak istediğiniz satıcının id bilgisini giriniz:"))

    # id_product = "1e9e8ef04dbcff4541ed26657ea517e5"
    # zip_code = 83203
    # id_seller = "0015a82c2db000af6aaaf3ae2ecb0532"

    # date = pd.to_datetime(orders["order_purchase_timestamp"].max()) + relativedelta(days=1)
    # date = pd.to_datetime("2019-05-06 05:05:34")
    date = purchase_date
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

    final_model = pickle.load(open("Pickles/finalized_model.sav", 'rb'))

    # print(f"Ürünün Tahmini Teslimat Süresi (Tempo estimado de entrega do produto): {final_model.predict(data)[0]}")
    tahmini_gun = final_model.predict(data)[0]

    return tahmini_gun

# def predict():
#     import pickle
#
#     final_model = pickle.load(open("Pickles/finalized_model.sav", 'rb'))
#     print(
#         f"Ürünün Tahmini Teslimat Süresi (Tempo estimado de entrega do produto): {final_model.predict(ml_pipeline())[0]} gün")
#

#predict()


# 1e9e8ef04dbcff4541ed26657ea517e5
# 83203
# 0015a82c2db000af6aaaf3ae2ecb0532
