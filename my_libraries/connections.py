import pandas as pd
import numpy as np
import seaborn as sns
import zipfile
import pickle
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def read_data_from_zip(zip_path, f_name, sheet=None):
    """
        Task        :   Pickle kontrol edilir eğer yoksa,
                        Zip dosyasından okuma işlemi yapılır ve pickle kaydedilir

        Parameters  :   zip_path    = Zip dosyası konumu
                        f_name      = Zip dosyası içerisinde okunacak olan dosya
                        sheet       = Dosya sheet ismi (Not Required)

        Returns     :   Fonskiyon geriye bir dataframe çevirir.

        Example     :   lb.read_data_from_zip(".../.../abc.zip", "xyz.csv") :  abc zip dosyasındasn xyz csv dosyasını okur.
    """

    file_name = f_name[: f_name.index('.')]  # uzantısı silinir

    if sheet == None:
        file_name = file_name + ".pkl"  # pkl uzantısı eklenir
    else:
        file_name = file_name + "_" + sheet + ".pkl"  # pkl uzantısı eklenir

    df = read_pickle(file_name)

    if df.empty:
        print("DOSYA OKUNUYOR")

        z_file = zipfile.ZipFile(zip_path)
        try:
            if sheet is not None:
                df = pd.read_excel(z_file.open(f_name), sheet_name=sheet)
            else:
                df = pd.read_excel(z_file.open(f_name))
        except:
            if sheet is not None:
                df = pd.read_csv(z_file.open(f_name), sheet_name=sheet)
            else:
                df = pd.read_csv(z_file.open(f_name))

        pickle.dump(df, open(file_name, 'wb'))
        return df
    else:
        print("DOSYA ZATEN VAR")

    return df
def read_pickle(file_name):
    """
        Task        :   Pickle ile dosya okuma işlemi yapılır

        Parameters  :   file_name    = okunacak dosya ismi  .pkl formatı ile

        Returns     :   Fonskiyon geriye bir dataframe çevirir veya None çevirir

        Example     :   file_name("abc.pkl")
    """
    print(">>" + file_name)
    try:
        with open(file_name, "rb") as f:
            df = pickle.load(f)
    except Exception:
        df = pd.DataFrame()

    return df

def save_to_excel(df):
    writer = pd.ExcelWriter('output.xlsx')
    df.to_excel(writer)
    writer.save()


