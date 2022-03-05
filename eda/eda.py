import matplotlib.pyplot as plt
import pandas as pd

from my_libraries.data_prep import *
from my_libraries.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == "__main__":
    '''
    yaylali.hello()

    df_order_payments = yaylali.read_data_from_zip("C:/Users/Mustafa/PycharmProjects/vbo_e-commerce/brazilian_ecommerce.zip","olist_order_payments_dataset.csv")
    """
    payment_sequential   : Ödeme sırasını belirtir    
    payment_type         : Ödeme tipi   (credit_card, boleto(bilet), voucher (senet), debit_card(banka kartı))
    payment_installments : Ödeme taksitleri
    payment_value        : Ödeme değeri
    """

    yaylali.data_summary(df_order_payments)

    cat_cols, cat_but_car, num_cols, num_but_cat = yaylali.grab_col_names(df_order_payments)

    yaylali.num_summary(df_order_payments,num_cols)

    for col in cat_cols:
        yaylali.cat_summary(df_order_payments,col)

    yaylali.rare_analyser(df_order_payments,"payment_value",cat_cols)

    '''
    ######################
    ### SHIPPING LIMIT DATE İNCELEMESİ ####

    df_orders = pd.read_csv("datasets/olist_orders_dataset.csv")
    df_sellers = pd.read_csv("datasets/olist_sellers_dataset.csv")
    df_seller_mean_data = pd.read_csv("Helper Datasets/seller_id_mean_data.csv")
    df_geolocation = pd.read_csv("datasets/olist_geolocation_dataset.csv")
    df_customers = pd.read_csv("datasets/olist_customers_dataset.csv")

    new_table = pd.DataFrame(columns=["delivered", "estimated"])

    new_table[['delivered', 'estimated']] = df_orders[['order_delivered_customer_date', 'order_estimated_delivery_date']].apply(pd.to_datetime)  # if conversion required
    new_table['day_difference'] = (new_table['estimated'] - new_table['delivered']).dt.days


    new_table.loc[new_table["day_difference"] == 0,"category"] = "Same Day"
    new_table.loc[new_table["day_difference"] > 0,"category"] = "Early"
    new_table.loc[new_table["day_difference"] < 0,"category"] = "Late"
    new_table.loc[new_table["day_difference"].isnull(),"category"] = "Not Delivered"

    new_table["category"].value_counts()
    # 87187 adet ERKEN
    # 7827 adet GEÇ
    # 2965 adet Teslim Edilemeyen
    # 1462 aynı gün teslim edilen

    # YÜZDELİK olarak BAKARSAK, yaklaşık %95.2 si tahmin edilen zamandan farklı teslim edilmiş
    new_table['category'].value_counts(normalize=True) * 100

    # GRAFİK olarak
    mylabels = new_table['category'].unique()

    plt.pie(new_table['category'].value_counts(),
            labels = mylabels,
            autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)

    aa =new_table['category'].value_counts()

    plt.title("TESLİMAT ZAMANLARI GRAFİĞİ")

    plt.show()

    ###### TESLİM EDİLMEYENLERİ İNCELEYELİM 2965 adet
    not_delivered = df_orders[df_orders["order_delivered_customer_date"].isnull()]

    # 2965 siparişin Durumları,
    not_delivered["order_status"].value_counts()

    # GRAFİK OLARAK
    mylabels = not_delivered['order_status'].unique()
    plt.pie(not_delivered['order_status'].value_counts(),
            labels=mylabels,
            autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title("Teslimatı Gerçekleşmeyenlerin Güncel Durumları")
    plt.show()


    # 8 adet TESLİM EDİLMEDİĞİ Halde Teslim edildi olarak işaretlenmiş veri bulunmakta
    not_delivered[not_delivered["order_status"] == "delivered"]

    new_table['day_difference'].max() # 146 gün önce teslim edilen
    new_table['day_difference'].min() # 189 gün geç teslim edilen

    ###### GÜN BAZINDA GENEL TESLİMLERİ İNCELEYİM
    plt.hist(new_table['day_difference'],
             density=1,
             color='green',
             alpha=0.9
             )
    plt.xlabel('Day')
    plt.ylabel('Freq')
    plt.title('GÜN BAZINDA TESLİMAT')
    plt.show()


    #BOX PLOT ile günler
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x=new_table['day_difference'])


    ###### ERKEN-GEÇ Teslimlerin histogram dağılımı
    abc = new_table[(new_table['category'] == 'Early') | (new_table['category'] == 'Late')]
    abc['day_difference'].hist(by=abc['category'])
