import mechanize
import http.cookiejar as cookielib
from bs4 import BeautifulSoup
import html2text
import pandas as pd
import string
import numpy as np
import glob
import os
from sklearn.preprocessing import MultiLabelBinarizer

def cleanCSV():
    path = 'data/csv_raw'                    
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

    guilds_df.to_csv('data/all_guilds.csv', index = False)
    return guilds_df

def scrape():
    # Browser
    br = mechanize.Browser()

    # Cookie Jar
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)

    # Browser options
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

    br.addheaders = [('User-agent', 'Chrome')]

    # The site we will navigate into, handling it's session
    br.open('https://permacultureplantdata.com/register?view=login&return=aW5kZXgucGhwP0l0ZW1pZD02NjM=')

    # Select the second (index one) form (the first form is to register)
    br.select_form(nr=1)

    # User credentials
    br.form['username'] = 'spencerjsmall'
    br.form['password'] = 'Sp3nc3r1'

    # Login
    br.submit()

    pc_df = pd.DataFrame(columns=['Plant', 'Related Plant', 'Compatible', 'Notes', 'Reference'])

    for letter in string.ascii_uppercase:
        pc = br.open('https://permacultureplantdata.com/plant-database/plant-companions-list?vw=complist&start=' + letter)
        dfs = pd.read_html(pc)
        if (len(dfs) > 1):
            df = dfs[1]
            new_header = df.iloc[0] #grab the first row for the header
            df = df[1:] #take the data less the header row
            df.columns = new_header #set the header row as the df header
            pc_df = pc_df.append(df)

    pc_df_clean = pc_df.drop(columns=['Notes', 'Reference']).fillna(method='ffill')
    companions = pc_df_clean.loc[pc_df_clean['Compatible'] == 'Yes'].drop(columns=['Compatible']).rename(columns={'Related Plant':'Companion'})
    competitors = pc_df_clean.loc[pc_df_clean['Compatible'] == 'No'].drop(columns=['Compatible']).rename(columns={'Related Plant':'Competitor'})
    companions.to_csv('data/companions.csv', index = True)
    competitors.to_csv('data/competitors.csv', index = True)
    return companions, competitors

def associate(guilds_df, companions_df, competitors_df):
    c_users_df = pd.DataFrame(columns = ['User','Plant','Rating'])
    for i, row in companions_df.iterrows():
        c_users_df.loc[2*i] = ['Companion {}'.format(i), row['Plant'], 1]
        c_users_df.loc[2*i+1] = ['Companion {}'.format(i), row['Companion'], 1]
    
    x_users_df = pd.DataFrame(columns = ['User','Plant','Rating'])
    for i, row in competitors_df.iterrows():
        x_users_df.loc[2*i] = ['Competitor {}'.format(i), row['Plant'], -1]
        x_users_df.loc[2*i+1] = ['Competitor {}'.format(i), row['Competitor'], -1]
    
    g_users_df = guilds_df[["Guild", "Name"]].rename(columns={'Guild':'User', 'Name':'Plant'})
    g_users_df["Rating"] = 1

    users_df = pd.concat([c_users_df, x_users_df, g_users_df])
    users_df.to_csv('data/ratings.csv', index = False)
    return users_df

## ARCHIVED CODE FOR BINARY EMBEDDING:
# def encode(train, test):
#     mlb = MultiLabelBinarizer(sparse_output=False)
    
#     train_companions = train.loc[train['Rating'] == 1]
#     test_companions = test.loc[test['Rating'] == 1]
    
#     train_grouped = train_companions[['User', 'Plant']].groupby('User').aggregate(lambda x: tuple(x))
#     train_grouped.to_csv('data/train_tupled.csv', index = True)
#     train_tupled = pd.read_csv("data/train_tupled.csv")
#     train_encoded = mlb.fit_transform(train_grouped["Plant"])
#     train_encoded_df = pd.DataFrame(train_encoded, index=train_tupled["User"], columns=mlb.classes_)
#     train_encoded_df.to_csv('data/train_encoded.csv', index = True)

#     test_grouped = test_companions[['User', 'Plant']].groupby('User').aggregate(lambda x: tuple(x))
#     test_grouped.to_csv('data/test_tupled.csv', index = True)
#     test_tupled = pd.read_csv("data/test_tupled.csv")
#     test_encoded = mlb.transform(test_grouped["Plant"])
#     test_encoded_df = pd.DataFrame(test_encoded, index=test_tupled["User"], columns=mlb.classes_)
#     test_encoded_df.to_csv('data/test_encoded.csv', index = True)

#     return train_encoded_df, test_encoded_df

# def encode(ratings):
#     mlb = MultiLabelBinarizer(sparse_output=False)
    
#     companions = ratings.loc[train['Rating'] == 1]
    
#     grouped = ratings[['User', 'Plant']].groupby('User').aggregate(lambda x: tuple(x))
#     grouped.to_csv('data/ratings_tupled.csv', index = True)
#     tupled = pd.read_csv("data/ratings_tupled.csv")
#     encoded = mlb.fit_transform(grouped["Plant"])
#     encoded_df = pd.DataFrame(encoded, index=tupled["User"], columns=mlb.classes_)
#     encoded_df.to_csv('data/encoded.csv', index = True)

#     return encoded_df