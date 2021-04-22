import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import MultiLabelBinarizer

def cleanCSV():
    path = r'Guild_CSVs'                    
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f).assign(Guild = f[11:-4]).assign(Guild_id = i) 
        for i, f in enumerate(all_files))
    guilds_df   = pd.concat(df_from_each_file, ignore_index=True)

    guilds_df[["Hardiness Zone: low", "Hardiness Zone: high"]]= guilds_df["Hardiness Zone"].str.split(" -- ", expand = True).replace('?', '0').astype('float64')
    guilds_df["Soil Moisture"]= guilds_df["Soil Moisture"].str.split(", ", expand = False)
    guilds_df["Ecological Function"]= guilds_df["Ecological Function"].str.split(", ", expand = False)
    guilds_df["Bloom Time"]= guilds_df["Bloom Time"].str.split(" - ", expand = False)
    guilds_df["Seasonal Interest"]= guilds_df["Seasonal Interest"].str.split("-", expand = False)
    guilds_df["Human Use/Crop"]= guilds_df["Human Use/Crop"].str.split(", ", expand = False)
    guilds_df["Root Type"]= guilds_df["Root Type"].str.split(", ", expand = False)
    guilds_df["Light"]= guilds_df["Light"].str.split(", ", expand = False)
    guilds_df[["Soil pH: low", "Soil pH: high"]]= guilds_df["Soil pH"].str.split(" - ", expand = True).replace('?', '0').astype('float64')
    guilds_df["Fruit Time"]= guilds_df["Fruit Time"].str.split(" - ", expand = False)
    guilds_df = guilds_df.drop(columns=['Flower Color'])
    guilds_df = guilds_df.drop(columns=['Hardiness Zone'])
    guilds_df = guilds_df.drop(columns=['Soil pH'])

    guilds_df.to_csv('data/concat_guilds.csv', index = False)
    return guilds_df

def associate(guilds, companions):
    df_g = guilds[["Name", "Guild"]]
    df_c = companions
    #df_x = competitors
    df_j = df_g.merge(df_g, on="Guild")
    df_j.to_csv('data/joined.csv', index = False)
    df_combined = pd.concat(
        [df_c.rename(columns={'Related Plant':'Association'}),
        df_j.drop(columns=['Guild']).rename(columns={'Name_x':'Plant', 'Name_y':'Association'})], 
        ignore_index=True)
    df_combined.to_csv('data/associations.csv', index = False)
    return df_combined

def encode(associations):
    df_a = associations
    grouped = df_a.groupby('Plant').aggregate(lambda x: tuple(x))
    grouped.to_csv('data/tupled.csv', index = True)
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(grouped['Association'])
    df_e = pd.DataFrame(encoded)
    df_e.to_csv('data/encoded.csv', index = False)
    return df_e


