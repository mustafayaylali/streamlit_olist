import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from warnings import filterwarnings
filterwarnings('ignore')

def data_summary(dataframe):
    print("-------------------DATA SUMMARY---------\n")
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types 2 #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("\n-------------------DATA SUMMARY END---------\n")
def num_summary(dataframe, numerical_col, plot=False):
    """
    Bir sayısal değişkenin çeyreklik değerlerini gösterir ve histogram olusturur

    example:
        for col in age_cols:
            num_summary(df, col, plot=True)
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("\n-------------NUM SUMMARY START--------------\n")
    print(dataframe[numerical_col].describe(quantiles).T)
    print("\n-------------NUM SUMMARY END--------------\n")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def cat_summary(dataframe, col_name, plot=False):
    """
    Verilen kategorik değişken için frekans-oran detaylarını yazdırır.
    :param dataframe:
    :param col_name:
    :param plot:

    example:
        for col in cat_cols:
            cat_summary(df, col)

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    kategorik değişkenlere göre hedef değişkenin ortalamasını verir

    example:
    for col in cat_cols:
        target_summary_with_cat(df,"SalePrice",col)
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

    print("Kategorik değişkenlerin sınıf oranları:")
    for col in categorical_col:
        print("\n"+col+" :\n"+str(100 * dataframe[col].value_counts() / len(dataframe)))

    print("\n"+target+" min-max : "+ str(dataframe[target].min())+" - "+str(dataframe[target].max()))

    # Hedef değişkeninin histogram ve yoğunluk grafiği çizdirilmiştir.
    sns.distplot(dataframe[target])
def target_summary_with_num(dataframe, target, numeric_col):
    """
    hedef değişkene göre numerik değişkenlerin ortalamasını verir

    example:
    for col in cat_cols:
        target_summary_with_cat(df,"SalePrice",col)
    """
    print("-------------NUM SUMMARY WITH "+target+ " START------------\n")
    print(dataframe.groupby(target)[numeric_col].mean())
    print("\n-------------NUM SUMMARY WITH " + target + " END------------")

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    """
    Thresold değerinden büyükse siler, yüksek korelasyonları çıkartıyoruz
    :param dataframe:
    :param plot:
    :param corr_th:
    :return:
    example:
        drop_list = high_correlated_cols(df, plot=True)
        df.drop(drop_list, axis=1)
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

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
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat

def correlation_analysis(dataframe,target=False):
    """
    Genel korelasyon grafiği ve Hedef değişkene göre korelasyon grafiği oluşturulur
    :param dataframe:
    :param target:
    :return:
    """
    mask = np.triu(np.ones_like(dataframe.corr(), dtype=np.bool))

    f, ax = plt.subplots(figsize=[12, 6])
    sns.heatmap(dataframe.corr(), mask=mask, annot=True, fmt=".2f", ax=ax, cmap="BrBG")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show()

    if target:
        plt.figure(figsize=(8, 12))
        heatmap = sns.heatmap(dataframe.corr()[['Salary']].sort_values(by='Salary', ascending=False), vmin=-1,
                              vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating with '+target, fontdict={'fontsize': 18}, pad=16);
        plt.show()


def num_cols_visualization(num_cols,df):
    for i in num_cols:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
        sns.histplot(df[i], bins=10, ax=axes[0])
        axes[0].set_title(i)

        sns.boxplot(df[i], ax=axes[1])
        axes[1].set_title(i)

        sns.kdeplot(df[i], ax=axes[2])
        axes[2].set_title(i)
        plt.show()

