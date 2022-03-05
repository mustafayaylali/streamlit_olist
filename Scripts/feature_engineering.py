import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Veriye Genel Bakış
def check_df(dataframe):
    """
        Veriye şöyle bir genel bakış atmak isteriz belki.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Özeti istenilen dataframe.

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            check_df(df)
    """

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        cat_cols, num_cols, cat_but_car = grab_col_names(df)


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}\n')
    return cat_cols, num_cols, cat_but_car

# Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    """
        Kategorik değişken analizi yapar. Kategorik değişken sınıflarının frekanslarını ve toplam veri içindeki bulunma yüzdelerini döner.

        Parameters
        ------
            dataframe: dataframe
                    Muhatap olunan dataframe.
            col_name: string
                    Veri setinde analizi yapılacak olan kategorik değişkenin ismi.
            plot: bool
                    Eğer True girilirse değişkenin grafiği çizdirilir.
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Sayısal Değişken Analizi
def num_summary(dataframe, numerical_col, plot=False):
    """
    Sayısal değişken analizi.
    Numerik değişkenin 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99 çeyreklik değerlerini döner.

    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

# Kategorik değişkenlere göre hedef değişkenin ortalaması
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Hedef değişkeni kategorik değişkenler nezdinde analiz eder.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# Hedef değişkene göre numerik değişkenlerin ortalaması
def num_col_summary_with_target(dataframe, target, numeric_col, plot=False):
    """
        Hedef değişkene göre sayısal değerlerin ortalamalarını hesaplar.
    """
    dataframe = pd.DataFrame({f"{numeric_col}_Mean": dataframe.groupby(target)[numeric_col].mean()})
    print(pd.DataFrame(dataframe), end="\n\n\n")
    dataframe = dataframe.sort_values(f"{numeric_col}_Mean", ascending=False)

    if plot:
        plt.bar(dataframe.index, dataframe[f"{numeric_col}_Mean"])
        plt.xlabel(f"{numeric_col}_Mean")
        plt.title(f"{numeric_col}_Mean")
        plt.show()

#Aykırı değerler için alt ve üst limit belirleyecek fonksiyon
def outlier_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    """
        Veri setindeki numerik değişkenlerin aykırı değerlerinin belirlenebilmesi için gereken alt ve üst sınırları belirler.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenen dataframe.
            variable: string
                    Aykırı değeri ölçülmek istenen dataframe içindeki değişken.
            q1: float, optional
                    Alt sınır belirlenebilmesi için değişken içinden alınacak olan değerin yeri gibi bir şey. Anlatabildim mi? :)
            q3: float, optional
                    Üst sınır belirlenebilmesi için değişken içinden alınacak olan değerin yeri gibi bir şey. Anlatabildim mi? :)
        Returns
        ------
            low_limit: float
                    Aykırı değerler için alt sınır.
            up_limit: float
                    Aykırı değerler için üst sınır.

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            low_limit, up_limit = outlier_thresholds(df, "sepal_length", q1=0.25, q3=0.75)

    """

    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#Aykırı değerleri duruma göre alt ve üst limite baskılayan fonksiyon
def replace_with_thresholds(dataframe, variable, q1=0.1, q3=0.9):
    """
        Aykırı değerleri alt ve üst limite baskılar.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenen dataframe.
            variable: string
                    Aykırı değeri ölçülmek istenen dataframe içindeki değişken.
            q1: float, optional
                    Alt sınır belirlenebilmesi için değişken içinden alınacak olan değerin yeri gibi bir şey. Anlatabildim mi? :)
            q3: float, optional
                    Üst sınır belirlenebilmesi için değişken içinden alınacak olan değerin yeri gibi bir şey. Anlatabildim mi? :)

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            replace_with_thresholds(df, "sepal_length", q1=0.25, q3=0.75)
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    return dataframe

# Aykırı değer var mı yok mu?
def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    """
        Aykırı değer var mı yok mu?
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Eksik gözlem analizi
def missing_values_table(dataframe, na_name=False):
    """
        Veri setindeki hangi değişkende kaç eksik veri var, bu yüzde kaçına tekabül ediyor bunun analizini döner.
    """

    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

# Eksik değerleri doldurma
def quick_missing_imp(data, num_method="median", cat_length=20, target=None):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[targt]

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

# Korelasyon analizi
def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    """
    Korelason analizi yapor. plot argümanı True girilirse grafiğini de çizer.
    Ayrıca çok yüksek korelasyonu bulunan bir değişken varsa bunu da sonuç olarak döner.
    Bu değişken çok yüksek bir oranla başka bir değişkenle de temsil edilebildiğinden silinmesi düşünülebilir. Yoksa yanlılık olabilir.

    """

    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show()
    print("Drop List\n##########################")
    print(drop_list)
    return drop_list

# Bağımlı değişkenle korelasyonu düşük olan bağımsız değikenler
def low_correlated_cols(dataframe, target_col, corr_th=0.10):
    """
        Bağımlı değişkenle korelasyonu düşük olan bağımsız değikenleri liste olarak döner.
    """

    corr = dataframe.corr()
    cor_matrix = corr.abs()
    drop_list = cor_matrix[cor_matrix[target_col] < corr_th].index
    print("Drop List\n##########################")
    print(drop_list)
    return drop_list

# Eksik değerlerin hedef değişken ile arasındaki ilişkiyi analiz eder.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
