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


    customers = pd.read_csv("Datasets/olist_customers_dataset.csv")
    geolocation = pd.read_csv("Datasets/olist_geolocation_dataset.csv")
    order_items = pd.read_csv("Datasets/olist_order_items_dataset.csv")
    orders = pd.read_csv("Datasets/olist_orders_dataset.csv")
    product = pd.read_csv("Datasets/olist_products_dataset.csv")
    sellers = pd.read_csv("Datasets/olist_sellers_dataset.csv")
    end = time.time()
    print(f"Verileri okuma süresi: {round((end - start), 2)} saniye\n")
    return orders, customers, geolocation, sellers, order_items, product

def orders_merge():
    """
    Model hazırlığı için tablolaların gerekli birleştirme işlemlerini yapıp dataframe de tutar.

    Parameters
    ------
    Returns
    ------
        orders_: dataframe
            Birleştirme işlemlerinin tamamlandığı tablo
    Examples
    ------
        orders = orders_merge()

    """
    import time
    import pandas as pd
    start = time.time()

    orders, customers, geolocation, sellers, order_items, product = read_data()

    # Order Items
    order_items = pd.merge(order_items, product[
        ["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
                           on="product_id")

    order_items = order_items.groupby(["order_id", "seller_id", "product_id"])[
        ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]].sum()
    order_items.reset_index(inplace=True)

    orders = pd.merge(orders, order_items[
        ["order_id", "product_id", "seller_id", "product_weight_g", "product_length_cm", "product_height_cm",
         "product_width_cm"]], on="order_id")

    # Customers
    orders = pd.merge(orders,
                          customers[["customer_id", "customer_city", "customer_state", "customer_zip_code_prefix"]],
                          on="customer_id")


    # Bir zip kodu birden fazla enlem, boylam bilgisi içerebiliyor. Bu da merge işlemi yaparken çoklamaya sebep oluyor.
    # Her zip kodu için standart bir enlem boylam bilgisi belirleyebilmek için enlem ve boylamın ortalamaları alındı.
    # Geolocation - Customers
    geolocation = geolocation.groupby("geolocation_zip_code_prefix").agg(
        {"geolocation_lat": "mean", "geolocation_lng": "mean"})
    geolocation.reset_index(inplace=True)

    orders = pd.merge(orders, geolocation, left_on="customer_zip_code_prefix",
                          right_on="geolocation_zip_code_prefix")
    orders.rename(
        columns={"geolocation_lat": "customer_geolocation_lat", "geolocation_lng": "customer_geolocation_lng"},
        inplace=True)

    # Sellers
    orders = pd.merge(orders, sellers[["seller_id", "seller_city", "seller_state", "seller_zip_code_prefix"]],
                          on="seller_id")

    # Geolocation - Sellers
    orders = pd.merge(orders, geolocation, left_on="seller_zip_code_prefix",
                          right_on="geolocation_zip_code_prefix")
    orders.rename(
        columns={"geolocation_lat": "seller_geolocation_lat", "geolocation_lng": "seller_geolocation_lng"},
        inplace=True)

    end = time.time()
    print(f"Merge İşlemleri Tamamlanma Süresi: {round((end - start), 2)} saniye\n")

    return orders

def categorizing(data, col, target_col, num_of_categories):
    """
        Kategorik değişkenleri hedef değişkene göre gruplandırır.

        Parameters
        ------
            data: dataframe
                Gruplandırma yapılacak sütunun bulunduğu tablo
            col: column
                Gruplandırma yapılacak sütun
            target_col: column
                Hedef değişken
            num_of_categories: int
                Grup sayısı
        Returns
        ------
            data: dataframe
                Gruplandırma işlemi tamamlanmış tablo
            mean_data: dataframe
                Gruplandırma yapılacak sütunun etiketlerinin belirlenmiş olan halini tutan tablo

        Examples
        ------
            data, customer_city_mean_data = categorizing(data, "customer_city", "order_completion_day", 6)

        Notes
        ------
            Sadece 10 gruba kadar gruplandırma yapılabilir. (num_of_categories <= 10)
        """

    import pandas as pd

    labels = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"]
    mean_data = pd.DataFrame(data.groupby(col)[target_col].mean()).reset_index()

    mean_data["label"] = pd.qcut(mean_data[target_col], num_of_categories, labels=labels[:num_of_categories])

    data = pd.merge(data, mean_data[[col, "label"]], on=col)
    data.rename(columns={"label": f"categorical_{col}"}, inplace=True)
    return data, mean_data

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

def boyut_sinirla(data):
    """
        Ürün boyutlarınnı Olist firmasının nakliye standartlarına göre sınırlar.

        Parameters
        ------
            data: dataframe
                Boyut sınırlaması yapılcak tablo.
        Returns
        ------
            data: dataframe
                Boyut sınırlaması yapılmış tablo.
        Examples
        ------
            data = boyut_sinirla(data)
        Notes
        ------
            Kaynak: https://ajuda.olist.com/hc/pt-br/articles/4409253877908
    """

    print(f"Boyut Sınırlama Öncesi Shape: {data.shape}")
    data = data[(data["product_length_cm"] >= 16) &
                (data["product_length_cm"] <= 100) &
                (data["product_height_cm"] >= 2) &
                (data["product_height_cm"] <= 100) &
                (data["product_width_cm"] >= 11) &
                (data["product_width_cm"] <= 100) &
                (data["product_weight_g"] >= 50) &
                (data["product_weight_g"] <= 30000)].reset_index(drop=True)
    print(f"Boyut Sınırlama Sonrası Shape: {data.shape}\n")
    return data

def only_one_product(data):
    """
        İçerisinde sadece bir ürün olan siparişleri filtreler.

        Parameters
        ------
            data: dataframe
                Filtrelenecek tablo.
        Returns
        ------
            data: dataframe
                Filtrelenmiş tablo.
        Examples
        ------
            data = only_one_product(data)

        Notes
        ------
            Bu projede amacımız ürün teslim tarihini doğru tahmin etmek.
            Ancak birden fazla ürün siparişi olan siparişler de şöyle bir durum oluyordu;
            Örneğin bir siparişte A ve B ürün sipariş edildiğini düşünelim.
            A ürünü 7 günde teslim olurken B ürünü 15 günde teslim oluyor.
            Veri setinde iki ürünün de teslim tarihi ortak yani 15 gün olarak gözüküyordu.
            Bu da kirli bir veri oluşturuyordu, bunu önlemek için bu verileri sildik.
    """

    print(f"Tek Ürün Filtrelemesi Yapılmadan Önce Shape: {data.shape}")
    # Sadece bir ürün olan siparişler
    index = data["order_id"].value_counts()[data["order_id"].value_counts() == 1].index

    data = data[data.order_id.isin(index)]
    data.reset_index(drop=True, inplace=True)
    print(f"Tek Ürün Filtrelemesi Yapılmadan Sonra Shape: {data.shape}\n")
    return data

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

def data_preprocessing():
    """
        Veri ön işleme süreçleri yapılır.

        Parameters
        ------
        Returns
        ------
            orders_: dataframe
                Ön işleme yapılmış tablo.
        Examples
        ------
            orders_ = data_preprocessing(orders_)

    """
    import time
    import pandas as pd

    start = time.time()
    orders_ = orders_merge()

    # Sadece gönderimi tamamlanmış siparişler.
    # Hedef değişkende boş değerlerin olmaması için sadece sipariş süreci tamamlanmış siparişleri alıyoruz.
    orders_ = orders_[orders_["order_status"] == "delivered"]
    orders_.reset_index(drop=True, inplace=True)

    # Datetime veri tipinde olması gereken sütunların veri tipini ayarlıyoruz.
    for col in orders_.columns[3:8]:
        orders_[col] = pd.to_datetime(orders_[col])

    # Eksik Değerlerin Silinmesi
    orders_ = orders_.dropna()
    orders_.reset_index(drop=True, inplace=True)

    # Sadece tek ürün olan siparişler.
    orders_ = only_one_product(orders_)

    # Teslim edilme tarihi siparişin kabul tarihinden önce olan siprişlerin silinmesi.
    aykiri_ix = orders_[orders_["order_approved_at"] >= orders_["order_delivered_customer_date"]].index
    orders_ = orders_[~orders_.index.isin(aykiri_ix)]
    orders_.reset_index(drop=True, inplace=True)

    # Kargo paket boyutlarının sınırlarının belirlenmesi.
    orders_ = boyut_sinirla(orders_)

    end = time.time()
    print(f"Ön İşlemlerin Tamamlanma Süresi: {round((end - start), 2)} saniye\n")
    return orders_

def f_engineering():
    """
        Özellik mühendisliği süreçleri tamamlanır.

        Parameters
        ------
        Returns
        ------
            data: dataframe
                Özellik mühendisliği süreçleri yapılmış tablo.
        Examples
        ------
            orders_ = f_engineering()

    """
    import time
    import pandas as pd
    import numpy as np

    start = time.time()
    data = data_preprocessing()

    # Ürünlerin hacmi
    data["hacim"] = data["product_length_cm"] * data["product_height_cm"] * data["product_width_cm"]

    # Ürünlerin desi bilgilerinin kategorikleştirilmesi
    data = calculate_desi(data)

    # Gün ismi ve ay değerlerini yeni değişken eklenmesi
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
    customer_city_mean_data.to_csv("Helper Datasets/customer_city_mean_data.csv")
    data, seller_city_mean_data = categorizing(data, "seller_city", "order_completion_day", 6)
    seller_city_mean_data.to_csv("Helper Datasets/seller_city_mean_data.csv")
    data, seller_id_mean_data = categorizing(data, "seller_id", "order_completion_day", 6)
    seller_id_mean_data.to_csv("Helper Datasets/seller_id_mean_data.csv")
    data, seller_state_mean_data = categorizing(data, "seller_state", "order_completion_day", 6)
    seller_state_mean_data.to_csv("Helper Datasets/seller_state_mean_data.csv")

    # Haftaiçi haftasonu ayrımının yapılması
    data = data.assign(wknd_or_not=data.apply(assign_date, axis=1))

    end = time.time()
    print(f"Özellik Mühendisliği tamamlanma süresi: {round((end - start), 2)} saniye\n")
    return data

def data_k_means():
    """
        Hedef değişkeni etiketlemek için K-Means modeli kurar.

        Parameters
        ------
        Returns
        ------
            ocd_mean: dataframe
                K-Means sonucu günlerin grup bilgilerini içeren dataframe.
        Examples
        ------
            df = data_k_means()
        Notes
        ------
            Modeli kurmak için veri setindeki sadece 3 değişken kullanıldı.
            Bunun sebebi, temel bir tahmin modeli kurduğumuz da en önemli değişkenlerin bu 3 değişken olması.

    """
    import pandas as pd
    from sklearn.cluster import KMeans
    from Scripts import feature_engineering as fe
    from sklearn.preprocessing import StandardScaler
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    # çıktının tek bir satırda olmasını sağlar.
    pd.set_option('display.expand_frame_repr', False)

    data = f_engineering()
    data = data[["order_completion_day", "euclidean_distance", "hacim", "product_weight_g", "order_id"]]
    cat_cols, num_cols, cat_but_car = fe.grab_col_names(data, car_th=25)
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

data_k_means()
# Sınıf etiketleri küme numaralarına göre belirlenir.
# Örneğin; 1, 2 ve 3 gün aynı kümede oldukları için 1-3 olarak tek sınıf yapılacaktır.

def target_variable_tags():
    """
        Sınıf değişkenlerinin etiketlerini belirler.

        Parameters
        ------
        Returns
        ------
            data: dataframe
                Etiketleme işlemi yapılmış dataframe.
        Examples
        ------
            data = target_variable_tags()
        Notes
        ------
            Etiketler K-Means algoritması sonucuna göre belirlenmiştir.
    """

    data = f_engineering()
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
    # Sipariş tamamlanma süresi 0 gün olarak gözüken 1 sipariş var. Bu siparişi siliyoruz.
    data = data[~(data["target"] == "nan")]
    data.reset_index(drop=True, inplace=True)

    return data

def delete_outlier_values(dataframe, col):
    """
        Sayısal değişkenin aykırı değerlerini veriden siler.

        Parameters
        ------
            dataframe: dataframe
                Aykırı değerleri silinecek olan değişkenin bulunduğu tablo.
            col: column
                Aykırı değerleri kontrol edilecek değişken.
        Returns
        ------
            dataframe: dataframe
                Aykırı değerleri silinmiş olan değişkenin bulunduğu tablo.
        Examples
        ------
            data = delete_outlier_values(data, "order_completion_day")
    """

    # Sınırların belirlenmesi
    low_limit, up_limit = fe.outlier_thresholds(dataframe, col, q1=0.25, q3=0.75)

    # Aykırı Değerlerin Silinmesi
    dataframe = dataframe[dataframe[col] < up_limit]
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

def standard_scaling(dataframe, col):
    """
        Sayısal değişkenleri standart scaler ile scale eder.

        Parameters
        ------
            dataframe: dataframe
                Scale edilecek değişkenin bulunduğu tablo.
            col: column
                Scale edilecek değişken.
        Returns
        ------
            dataframe: dataframe
                Scale edilmiş değişkenin bulunduğu tablo.
            ss:
                Değişkenin fit edilmiş standart scaler nesnesi.
        Examples
        ------
            data, weight_ss = standard_scaling(data, "product_weight_g")
        Notes
        ------
            Daha sonra pipeline da kullanılmak amacıyla fit edilmiş standart scaler nesnesi return edilmiştir.
            Eğitim verisiyle test verisi arasındaki ölçek farkı sorunundan kurtulmak amacıyla
            fit edilmiş standart scaler nesnesi return edilmiştir.
    """

    # Standard Scaler
    ss = StandardScaler().fit(dataframe[[col]])
    dataframe[col] = ss.transform(dataframe[[col]])
    return dataframe, ss

def remove_target_outlier_values(data):
    """
        Her sınıf için ayrı ayrı aykırı değer analizi yapılıp aykırı değerlerin veriden silinmesi.

        Parameters
        ------
            data: dataframe
                Aykırı değerleri silinecek olan değişkenin bulunduğu tablo.
        Returns
        ------
            dataframe: dataframe
                Aykırı değerleri silinmiş olan değişkenin bulunduğu tablo.
        Examples
        ------
            data = remove_target_outlier_values(data)
        Notes
        ------
            Her sınıf için ayrı ayrı aykırı değer analizi yapılması gerektiğini düşündük.
            Çünkü, örneğin 1-3 gün için uzaklık mesafesinin aykırı değeri,
            bütün sınıflar ile yapılan aykırı değer analizinde yüksek ihtimal aykırı değer olarak gözükmeyecekti.
            Her sınıf için ayrı ayrı genelleme kaygımız olduğunda bunun yapılaması daha doğru bulduk.
    """
    print("Her Sınıf İçin Ayrı Ayrı Aykırı Değer Analizi\n"
          "---------------------------------------------\n")

    for i in data["target"].unique():
        ed_low_limit, ed_up_limit = fe.outlier_thresholds(data[data["target"] == i], "euclidean_distance", q1=0.25, q3=0.75)
        pw_low_limit, pw_up_limit = fe.outlier_thresholds(data[data["target"] == i], "product_weight_g", q1=0.25, q3=0.75)
        h_low_limit, h_up_limit = fe.outlier_thresholds(data[data["target"] == i], "hacim", q1=0.25, q3=0.75)

        print(f"Order Completion Day: {i}")
        print(f"Before: {data.shape}")
        aykiri_ix = data[(data["target"] == i) & (data["euclidean_distance"] > ed_up_limit)].index
        data = data[~data.index.isin(aykiri_ix)]
        aykiri_ix = data[(data["target"] == i) & (data["product_weight_g"] > pw_up_limit)].index
        data = data[~data.index.isin(aykiri_ix)]
        aykiri_ix = data[(data["target"] == i) & (data["hacim"] > h_up_limit)].index
        data = data[~data.index.isin(aykiri_ix)]

        print(f"After: {data.shape}\n")
    return data

def double_check_after_fe():
    """
        Özellik mühendisliği sonrası son veri ön işleme işlerinin yapılması.

        Parameters
        ------
        Returns
        ------
            data: dataframe
                En son veri ön işlemleri yapılmış tablo.
        Examples
        ------
            orders, weight_ss, hacim_ss, desi_ss, ed_ss = double_check_after_fe()
    """
    import pickle

    data = target_variable_tags()

    # Aykırı Değerler
    data = delete_outlier_values(data, "order_completion_day")
    # Diğer aykırı değerlere harici olarak bakılacak.

    # Standart Scaling
    data, weight_ss = standard_scaling(data, "product_weight_g")
    data, hacim_ss = standard_scaling(data, "hacim")
    data, desi_ss = standard_scaling(data, "desi")
    data, ed_ss = standard_scaling(data, "euclidean_distance")

    # Kullanılmasının amacı, pipeline da kullanılacak olan nesnelerin kaydedilmesidir.
    # Pipeline performansını artırır.
    pickle.dump(weight_ss, open("Pickles/weight_ss.sav", 'wb'))
    pickle.dump(hacim_ss, open("Pickles/hacim_ss.sav", 'wb'))
    pickle.dump(desi_ss, open("Pickles/desi_ss.sav", 'wb'))
    pickle.dump(ed_ss, open("Pickles/ed_ss.sav", 'wb'))


    # Etiketlere göre öklit uzaklığı, hacim ve ağırlığın aykırı değerlerinin silinmesi.
    data = remove_target_outlier_values(data)

    # One-Hot Encoding
    cat_cols, num_cols, cat_but_car = fe.grab_col_names(data, car_th=25)
    cat_cols.remove("target")
    cat_cols.remove("seller_state")
    data = fe.one_hot_encoder(data, cat_cols, drop_first=True)
    return data

def modele_hazirlik():
    """
        Veri seti modele girmeden önce son değişken silme işlemleri yapılır.

        Parameters
        ------
        Returns
        ------
            data: dataframe
                Değişken silme işlemleri tanımlanmış en son güncel veri.
        Examples
        ------
            orders = modele_hazirlik()
    """
    data = double_check_after_fe()

    data = data.drop(columns=["order_id", 'customer_id', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                                'product_id', 'seller_id', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'customer_city',
                                'customer_state', 'customer_zip_code_prefix', 'geolocation_zip_code_prefix_x', 'customer_geolocation_lat',
                                'customer_geolocation_lng', 'seller_city', 'seller_state', 'seller_zip_code_prefix',
                                'geolocation_zip_code_prefix_y', 'seller_geolocation_lat', 'seller_geolocation_lng',
                                'order_completion_day'])

    return data

def model():
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    import pickle

    data = modele_hazirlik()
    models = []
    models.append(('LightGBM', LGBMClassifier(random_state=55)))
    models.append(('RandomForest', RandomForestClassifier(random_state=55)))


    train, test = train_test_split(data, random_state=55, test_size=0.20)
    X_train = train.drop(columns = ["target", "order_delivered_customer_date", "order_estimated_delivery_date"])
    y_train = train["target"]
    X_test = test.drop(columns = ["target", "order_delivered_customer_date", "order_estimated_delivery_date"])
    y_test = test["target"]

    print("\nBase Model Accuracy Skorları")
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name}: {accuracy_score(y_test, y_pred)}")

    lgb_params = {"learning_rate": [0.1, 0.01, 0.05],
                  "n_estimators": [100, 200, 500],
                  "max_depth": [-1, 5, 10]}


    gs_best = GridSearchCV(LGBMClassifier(random_state=55), lgb_params, cv=5, n_jobs=-1, verbose=False).fit(X_train, y_train)

    final_model = LGBMClassifier(random_state=55).set_params(**gs_best.best_params_).fit(X_train, y_train)

    print("Parametre optimizasyonu tamamlandı.")
    y_pred = final_model.predict(X_test)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}\n")
    data["order_delivered_customer_date"] = [pd.to_datetime(time.strftime("%Y-%m-%d")) for time in
                                             data["order_delivered_customer_date"]]
    print(
        f"Orjinal veri setindeki tahminlerin doğruluk oranı: {accuracy_score(data['order_delivered_customer_date'], data['order_estimated_delivery_date'])}\n")
    filename = 'Pickles/finalized_model.sav'
    pickle.dump(final_model, open(filename, 'wb'))
    print("Final modeli kaydedildi...\n")

    print(f"Classification Report\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix\n{pd.DataFrame(confusion_matrix(y_test, y_pred), index=np.sort(y_test.unique()), columns=np.sort(y_test.unique()))}\n")


    return final_model, X_train

def plot_importance(model, features, num, save=False):
    import seaborn as sns
    from matplotlib import pyplot as plt

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from Scripts import feature_engineering as fe
    import time
    from sklearn.preprocessing import StandardScaler

    final_model, X_train = model()
    plot_importance(final_model, X_train, len(X_train))

