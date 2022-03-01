import pandas as pd
import numpy as np
import seaborn as sns
import zipfile
import pickle
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

################################
# ENCODING FONKSİYONLARI
################################

def label_encoder(dataframe, binary_col):
    """
    Verilen sutunu label_encoder e çevirir
    :param dataframe: veri seti
    :param binary_col: dönüştürülecek sutun
    :return: encode edilmiş sutunu içeren veri setini döndürür

    example:
        binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

        for col in binary_cols:
            label_encoder(df, col)
    """
    from sklearn.preprocessing import LabelEncoder

    for col in binary_col:
        le = LabelEncoder()
        dataframe[col] = le.fit_transform(dataframe[col])

    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """

    :param dataframe:
    :param categorical_cols:
    :param drop_first:
    :return:

    example:
        ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
        one_hot_encoder(df, ohe_cols).head()
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def rare_analyser(dataframe, target, cat_cols):
    """
    Bağımlı değişken ve kategorik değişkenleri birlikte analiz eder

    :param dataframe:
    :param target:
    :param cat_cols:
    example:
        rare_analyser(df, "TARGET", cat_cols)
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    """
    Verilen oran altında kalan kategorik değişken sınıflarını biraraya getirir

    :param dataframe:
    :param rare_perc:
    :return:

    example:
            new_df = rare_encoder(df, 0.01)
            rare_analyser(new_df, "TARGET", cat_cols)
    """
    temp_df = dataframe.copy()

    # her değişken için oranı verilen orandan daha düşük bir categorik değişken varsa onları bulur
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    # rare sınıfa sahip değişkenlerin indexleri bulunup yerlerine Rare yazılır
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

###################################
# STANDARLAŞTIRMA vs. FONKSİYONLARI
###################################

def standardization(df, column_name, min=1, max=5, time="No"):
    """
        Task        :   Dataframe içerisinde verilen bir alan için verilen aralıklarda STANDARTLAŞTIRMA işlemi uygular

        Parameters  :   df = Veri setini içerir
                        column_name =  Standartlaştırma uygulanacak alandır
                        min =  Standarlaştırma aralığında yer alan alt değer
                        max =  Standarlaştırma aralığında yer alan üst değer
                        time = Standartlaştırılacak alanın Gün/ay/Yıl gibi zaman içerip içermediği bilgisi

        Returns     :   Fonskiyon geriye "_scaled" ismiyle standartlaştırılmış bir alan ekleyerek df'i döndürür.

        Example     :   standardization(df, "day_diff", 1, 5, "Yes") : 1-5 arasına günleri standarlaştırır.
                            "Yes" : 5 en yakın zaman demektir
                            "No"  : 5 en yüksek değer demektir
    """
    from sklearn.preprocessing import MinMaxScaler

    if time == "Yes":
        df[column_name + "_scaled"] = 5 - MinMaxScaler(feature_range=(min, max)). \
            fit(df[[column_name]]). \
            transform(df[[column_name]])
    else:
        df[column_name + "_scaled"] = MinMaxScaler(feature_range=(min, max)). \
            fit(df[[column_name]]). \
            transform(df[[column_name]])

    return df
def minmax_scaler(dataframe, col_names, target_col, feature_range=(0,1)):
    minmax_scaler = MinMaxScaler(feature_range=feature_range)
    col_names=[col for col in col_names if col !=target_col]
    dataframe[col_names] = minmax_scaler.fit_transform(dataframe[col_names])
    return dataframe
def num_to_cat(df,column_name,part_num=5):
    #qcut sıralama yapar ve çeyrekliklere göre x parçaya böler
    df[column_name+"_qcut"] = pd.qcut(df[column_name], part_num)
    return df

###########################################
# MISSING VALUE (EKSİK DEĞER) FONKSİYONLARI
############################################

def missing_values_table(dataframe, na_name=False):
    '''
    Eksik değere sahip olan sutunlar ve oranlarını hesaplayarak ekrana yazdırır

    :param dataframe: veriseti
    :param na_name: sutun isimleridir

    :return:
    '''

    import numpy as np
    print("\n-------- EKSİK DEĞER ANALİZİ START----------\n")
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    print("\n-------- EKSİK DEĞER ANALİZİ END----------\n")
    if na_name:
        return na_columns
def missing_values_visualization(df):
    import missingno as msno
    msno.bar(df)
    plt.show()

    msno.matrix(df)
    plt.show()

    msno.heatmap(df)
    plt.show()
def missing_vs_target(dataframe, target, na_columns):
    """
    Eksik Değerlerin Bağımlı Değişken ile Analizi.

    :param dataframe: veri seti
    :param target: Bağımlı değişken
    :param na_columns: eksik değeri olan değişkenler , incelenecek değişkenler
    :return: Değişkenlerin Bağımlı Değişkene göre eksik değerlerini ekrana yazdırır
    """

    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    """
    # Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
    :param data:
    :param num_method:
    :param cat_length:
    :param target:
    :return:
    """
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data
def fill_missing_KNN(dataframe,n_neighbors=5):
    """
    Eksik gözlemler KNN Algoritması ile doldurulur.
    :param dataframe:
    :return:
    """
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_filled = imputer.fit_transform(dataframe)

    df = pd.DataFrame(df_filled, columns=dataframe.columns)

    return df

###########################################
# OUTLIER VALUE (AYKIRI DEĞER) FONKSİYONLARI
############################################

def check_outlier(dataframe, col_name,box_plot=False):

    '''
    Aykırı Değer var mı yok mu Kontrol edilir.
    :param dataframe:

    :param col_name:

    :return:

    Examples:
            for col in num_cols:
                print(col, check_outlier(dff, col))
    '''

    #print("Aykırı değer kontrol ediliyor...")
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    df_outlier = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]

    if box_plot:
        import seaborn as sns
        sns.boxplot(x=dataframe[col_name])
        plt.show()


    #print("Aykırı değer analizi bitti...")
    # aykırı değişken var mı?
    if df_outlier.any(axis=None):
        return True
    else:
        return False
def check_all_outliers(df):

    for feature in df:

        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3-Q1
        upper = Q3 + 1.5*IQR
        lower = Q1 - 1.5*IQR

        if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
            print(feature,"yes")
            print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
        else:
            print(feature, "no")
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99, boxplot=False):
    '''
    AYKIRI DEĞER SINILARINI GETİRİR
    :param dataframe:
    :param col_name:
    :param q1:
    :param q3:
    :param boxplot:
    :return:
    '''
    from matplotlib import pyplot as plt
    import sns as sns

    #print("Aykırı değer sınırları hesaplanıyor...")

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1


    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    if boxplot:
        sns.boxplot(x=dataframe[col_name])
        plt.show()

    #print("Aykırı değer analizi bitti...")

    return low_limit, up_limit
def grab_outliers(dataframe, col_name, index=False):
    '''
    Aykırı değerleri döndürür. (index leri opsiyonel)
    :param dataframe:
    :param col_name:
    :param index:
    :return: aykırı değerleri döndürür
    '''

    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
def remove_outlier(dataframe, col_name):
    '''
    aykırı değerleri silerek dataframe i döndürür

    :param dataframe:
    :param col_name:
    :return:
    '''

    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
def replace_with_thresholds(dataframe, col_name):
    '''
    BASKILAMA YAPILIR, aykırı değerler eşik değerler ile eşitlenir
    :param dataframe:
    :param variable:
    :return:
    '''
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

    return dataframe



def outlier_lof(df,n_neighb=20):
    """
    Local Outlier Factor Yöntemi ile Aykırı Gözlem Analizi (LOF)
    :param df:
    :return:
    """
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=n_neighb)
    lof.fit_predict(df)

    # Skor değerleri gelmiştir.
    df_scores = lof.negative_outlier_factor_

    # Eşik değeri belirlenilmiştir.
    threshold = np.sort(df_scores)[9]

    # Belirlenen eşik değer veri setine uyarlanarak aykırı gözlemlerden kurtulunmuş olundu.
    outlier = df_scores > threshold
    df = df[outlier]
    return df

def lof_scores(df):
    """
    Her bir gözlemin LOF Score unu hesaplar
    :param df:
    :return:
    """
    clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    clf.fit_predict(df)
    df_scores=clf.negative_outlier_factor_
    sns.boxplot(df_scores)
    plt.show()
    return df_scores
def lof_threshold(df, df_scores, threshold):
    """
    Lof Score lara bakılarak Anormal olan veri silinir.
    :param df:
    :param df_scores:
    :param threshold:
    :return:
    """
    not_outlier = df_scores >threshold
    value = df[df_scores == threshold]
    outliers = df[~not_outlier]
    res = outliers.to_records(index=False)
    res[:] = value.to_records(index=False)
    not_outlier_df = df[not_outlier]
    outliers = pd.DataFrame(res, index = df[~not_outlier].index)
    df_res = pd.concat([not_outlier_df, outliers], ignore_index = True)
    return df_res



#########EKSTRA
def calc_weighted_average(dataframe, ref_column_name, target_column_name, thresold_list=[30, 90, 180],
                          weights_list=[28, 26, 24, 22]):
    """
        Task        :   Bir Dataframe içerisinde referans alanı verilen thresoldlara dilimleyerek yine verilen alanın ağırlıklı ortalamasını hesaplar.

        Parameters  :   dataframe = Veri setini içerir
                        ref_column_name =  Filtrelenecek, dilimlenmede kullanılacak alan ismi
                        target_column_name =  Ortalaması alınacak alan ismi
                        thresold_list =  Dilim eşik değerlerinin belirlendiği liste
                        weights_list =  Dilimlerde çarpılacak ağırlık katsayıları

        Returns     :   Fonskiyon geriye ağırlıklı ortalama değeri çevirir.

        Example     :   calc_weighted_average(df, "day_diff", "overall", [30,90,100],[1,2,3,4]) : df datasında yer alan
                        day_diff alanını 30,90,100 aralıklarına bölerek 1,2,3,4 katsayıları ile çarpar ve
                        ortalamaları toplayarak TOPLAM ORTALAMA AĞIRLIK değerini döndürür.
    """

    total_weights = sum(weights_list)

    total_value = 0

    for i in range(0, len(thresold_list)):
        if i == 0:
            total_value += dataframe.loc[dataframe[ref_column_name] <= thresold_list[i], target_column_name].mean() * \
                           weights_list[i] / total_weights
        if i == 0 | i != len(thresold_list) - 1:
            total_value += dataframe.loc[(dataframe[ref_column_name] > thresold_list[i]) & (
                    dataframe[ref_column_name] <= thresold_list[i + 1]), target_column_name].mean() * \
                           weights_list[i + 1] / total_weights
        else:
            total_value += dataframe.loc[dataframe[ref_column_name] > thresold_list[i], target_column_name].mean() * \
                           weights_list[i + 1] / total_weights

    return total_value

# Online_Retail_|| dataset
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


####### FEATURE ENGINEERING
def create_date_features(df):
    """
    Tarih değişkeninden yeni değişkenler türetilir
    :param df:
    :return:
    """
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

def random_noise(dataframe):
    """
    Rastgele gürültüler eklenir
    :param dataframe:
    :return:
    """
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def lag_features(dataframe, lags):
    """
    Gecikme eklemeyi sağlar.

    Example: Zaman serisinin en önemli varsayımlarından biri olan; t anındaki değişkeni en çok t-1 andaki değişken etkiler ifadesi için
            modele ilgili periyodun 1 öncesi 2 öncesi 3 öncesi gecikmeleri ekliyoruz.
            Bunu bilgiyi gelecekteki tahminleri elde etmek için yapıyoruz.
    """
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe



def roll_mean_features(dataframe, windows):
    """
    HAREKETLİ ORTALAMA HESABI
    :param dataframe:
    :param windows:
    :return:
    """
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe