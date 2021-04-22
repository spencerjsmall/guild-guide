import pandas as pd
import random
import numpy as np
import argparse
from os import path
import matplotlib.pyplot as plt

import recmetrics
from surprise import Dataset, Reader, SVD, SVDpp, NMF
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from scipy import sparse

from plot import seasonalPlots, boxPlots, rangePlots
from preprocess import scrape, cleanCSV, associate

#source venv/bin/activate

def loadData():
    global ratings_df
    global encoded_df
    
    #Get all known guilds
    if path.exists("data/all_guilds.csv"):
        guilds_df = pd.read_csv("data/all_guilds.csv")
    else:
        print("CLEANING GUILD CSV") 
        guilds_df = cleanCSV()

    if path.exists("data/companions.csv"):
        companions_df = pd.read_csv("data/companions.csv")
        competitors_df = pd.read_csv("data/competitors.csv")
    else:
        print("SCRAPING PLANT ASSOCIATIONS") 
        companions_df, competitors_df = scrape()

    #Scrape related plants and associatie them with guilds
    if path.exists("data/ratings.csv"):
        ratings_df = pd.read_csv("data/ratings.csv")
    else:
        print("CREATING RATINGS MATRIX") 
        ratings_df = associate(guilds_df, companions_df, competitors_df)

    if path.exists("data/encoded.csv"):
        encoded_df = pd.read_csv("data/encoded.csv")
    else:
        print("ENCODING USER MATRIX") 
        encoded_df = encode(ratings_df)


####ARCHIVED METHOD FOR COSINE SIMILARITY
# def cosRec(sel_arr, sim_df, guilds_df):
#     label_col = sim_df.iloc[:,0].rename("Name")
#     sim_df = sim_df.drop(sim_df.columns[0], 1)
#     sel_idcs = map(lambda p: sim_df.columns.get_loc(p), sel_arr)
#     sel_idcs = np.array(list(sel_idcs))
#     sel_vector = np.zeros(len(sim_df.columns))
#     sel_vector[sel_idcs] = 1
    
#     # Join genus and calculate the score.
#     scores = sim_df.dot(sel_vector.astype(int)).div(sim_df.sum(axis=1)).rename("Score")
#     score_df = pd.concat([label_col, scores], axis=1)
#     genus_df = guilds_df[["Name", "Scientific name"]].groupby('Name').first().reset_index()
#     genus_df["Genus"] = genus_df["Scientific name"].apply(lambda x: x.split(' ', 1)[0])
#     genus_df = genus_df.drop(columns=['Scientific name'])
#     score_df = score_df.merge(genus_df)

#     # Remove the known likes from the recommendation.
#     score_df = score_df.drop(p for p in sel_idcs)

#     # Print the known likes and the top 20 recommendations.
#     return score_df.nlargest(10, 'Score')

def get_users_predictions(user, n, model):
    recommended_items = pd.DataFrame(model.loc[user])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)    
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()

def rmse_vs_factors(algorithm, data):
    """Returns: rmse_algorithm i.e. a list of mean RMSE of CV = 5 in cross_validate() for each  factor k in range(1, 101, 1)
    100 values 
    Arg:  i.) algorithm = Matrix factoization algorithm, e.g SVD/NMF/PMF, ii.)  data = surprise.dataset.DatasetAutoFolds
    """
    
    rmse_algorithm = []
    
    for k in range(1, 101, 1):
        print(k)
        algo = algorithm(n_factors = k)
        
        #["test_rmse"] is a numpy array with min accuracy value for each testset
        loss_fce = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)["test_rmse"].mean() 
        rmse_algorithm.append(loss_fce)
    
    return rmse_algorithm
    
def plot_rmse(rmse, algorithm):
    """Returns: sub plots (2x1) of rmse against number of factors. 
        Vertical line in the second subplot identifies the arg for minimum RMSE
        
        Arg: i.) rmse = list of mean RMSE returned by rmse_vs_factors(), ii.) algorithm = STRING! of algo 
    """
    
    plt.figure(num=None, figsize=(11, 5), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(2,1,1)
    plt.plot(rmse)
    plt.xlim(0,100)
    plt.title("{0} Performance: RMSE Against Number of Factors".format(algorithm), size = 20 )
    plt.ylabel("Mean RMSE (cv=5)")

    plt.subplot(2,1,2)
    plt.plot(rmse)
    plt.xlim(0,50)
    plt.xticks(np.arange(0, 52, step=2))

    plt.xlabel("{0}(n_factor = k)".format(algorithm))
    plt.ylabel("Mean RMSE (cv=5)")
    plt.axvline(np.argmin(rmse), color = "r")
    plt.show()

def longTail():
    fig = plt.figure(figsize=(15, 7))
    recmetrics.long_tail_plot(df=ratings_df, 
             item_id_column="User", 
             interaction_type="associations", 
             percentage=0.5,
             x_labels=False)

def cfCompare():
    ratings = ratings_df
    nn_ratings = ratings_df.loc[ratings_df['Rating'] == 1]
    users = ratings["User"].value_counts()
    
    reader = Reader(rating_scale=(-1, 1))
    nn_reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(ratings, reader)
    nn_data = Dataset.load_from_df(nn_ratings, nn_reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=420)
    nn_trainset, nn_testset = train_test_split(nn_data, test_size=0.25, random_state=420)

    svd = SVD(n_factors = 4)
    svdpp = SVDpp(n_factors = 4)
    nmf = NMF(n_factors=17)
    
    svd.fit(nn_trainset)
    svd_test = svd.test(nn_testset)
    svd_test = pd.DataFrame(svd_test)
    svd_test.drop("details", inplace=True, axis=1)
    svd_test.columns = ['User', 'Plant', 'actual', 'cf_predictions']

    svdpp.fit(nn_trainset)
    svdpp_test = svdpp.test(nn_testset)
    svdpp_test = pd.DataFrame(svdpp_test)
    svdpp_test.drop("details", inplace=True, axis=1)
    svdpp_test.columns = ['User', 'Plant', 'actual', 'cf_predictions']

    nmf.fit(nn_trainset)
    nmf_test = nmf.test(nn_testset)
    nmf_test = pd.DataFrame(nmf_test)
    nmf_test.drop("details", inplace=True, axis=1)
    nmf_test.columns = ['User', 'Plant', 'actual', 'cf_predictions']
    #print(test.head())

    svd_model = svd_test.pivot_table(index='User', columns='Plant', values='cf_predictions').fillna(0)
    svdpp_model = svdpp_test.pivot_table(index='User', columns='Plant', values='cf_predictions').fillna(0)
    nmf_model = nmf_test.pivot_table(index='User', columns='Plant', values='cf_predictions').fillna(0)

    svd_test = svd_test.copy().groupby('User', as_index=False)['Plant'].agg({'actual': (lambda x: list(set(x)))})
    svdpp_test = svdpp_test.copy().groupby('User', as_index=False)['Plant'].agg({'actual': (lambda x: list(set(x)))})
    nmf_test = nmf_test.copy().groupby('User', as_index=False)['Plant'].agg({'actual': (lambda x: list(set(x)))})

    svd_test = svd_test.set_index("User")
    svdpp_test = svdpp_test.set_index("User")
    nmf_test = nmf_test.set_index("User")

    svd_recs = [] = []
    svdpp_recs = [] = []
    nmf_recs = [] = []
    for user in svd_test.index:
        svd_predictions = get_users_predictions(user, 10, svd_model)
        svd_recs.append(svd_predictions)
        svdpp_predictions = get_users_predictions(user, 10, svdpp_model)
        svdpp_recs.append(svdpp_predictions)
        nmf_predictions = get_users_predictions(user, 10, nmf_model)
        nmf_recs.append(nmf_predictions)
        
    svd_test['cf_predictions'] = svd_recs
    svdpp_test['cf_predictions'] = svdpp_recs
    nmf_test['cf_predictions'] = nmf_recs
    #print(test.head())
    
    #POPULARITY
    #make recommendations for all members in the test data
    popularity_recs = ratings.Plant.value_counts().head(10).index.tolist()

    pop_recs = []
    for user in svd_test.index:
        pop_predictions = popularity_recs
        pop_recs.append(pop_predictions)
        
    svd_test['pop_predictions'] = pop_recs
    svdpp_test['pop_predictions'] = pop_recs
    nmf_test['pop_predictions'] = pop_recs

    #RANDOM
    ran_recs = []
    for user in svd_test.index:
        random_predictions = ratings.Plant.sample(10).values.tolist()
        ran_recs.append(random_predictions)
        
    svd_test['random_predictions'] = ran_recs 
    svdpp_test['random_predictions'] = ran_recs 
    nmf_test['random_predictions'] = ran_recs

    def mark():
        actual = svd_test.actual.values.tolist()
        #maybe issue with ony one actual?
        svd_predictions = svd_test.cf_predictions.values.tolist()
        svdpp_predictions = svdpp_test.cf_predictions.values.tolist()
        nmf_predictions = nmf_test.cf_predictions.values.tolist()
        pop_predictions = svd_test.pop_predictions.values.tolist()
        random_predictions = svd_test.random_predictions.values.tolist()

        pop_mark = []
        for K in np.arange(1, 11):
            pop_mark.extend([recmetrics.mark(actual, pop_predictions, k=K)])
        
        random_mark = []
        for K in np.arange(1, 11):
            random_mark.extend([recmetrics.mark(actual, random_predictions, k=K)])

        svd_mark = []
        for K in np.arange(1, 11):
            svd_mark.extend([recmetrics.mark(actual, svd_predictions, k=K)])

        svdpp_mark = []
        for K in np.arange(1, 11):
            svdpp_mark.extend([recmetrics.mark(actual, svdpp_predictions, k=K)])

        nmf_mark = []
        for K in np.arange(1, 11):
            nmf_mark.extend([recmetrics.mark(actual, nmf_predictions, k=K)])

        mark_scores = [random_mark, pop_mark, svd_mark, svdpp_mark, nmf_mark]
        index = range(1,10+1)
        names = ['Random Recommender', 'Popularity Recommender', 'SVD CF', 'SVDpp CF', 'NMF CF']

        fig = plt.figure(figsize=(15, 7))
        recmetrics.mark_plot(mark_scores, model_names=names, k_range=index)

    def coverage():
        ##PREDICTION COVERAGE
        catalog = ratings.Plant.unique().tolist()
        random_coverage = recmetrics.prediction_coverage(ran_recs, catalog)
        pop_coverage = recmetrics.prediction_coverage(pop_recs, catalog)
        svd_coverage = recmetrics.prediction_coverage(svd_recs, catalog)
        svdpp_coverage = recmetrics.prediction_coverage(svdpp_recs, catalog)
        nmf_coverage = recmetrics.prediction_coverage(nmf_recs, catalog)

        ##CATALOG COVERAGE
        # N=100 observed recommendation lists
        random_cat_coverage = recmetrics.catalog_coverage(ran_recs, catalog, 100)
        pop_cat_coverage = recmetrics.catalog_coverage(pop_recs, catalog, 100)
        svd_cat_coverage = recmetrics.catalog_coverage(svd_recs, catalog, 100)
        svdpp_cat_coverage = recmetrics.catalog_coverage(svdpp_recs, catalog, 100)
        nmf_cat_coverage = recmetrics.catalog_coverage(nmf_recs, catalog, 100)
        
        
        ##COVERAGE PLOT
        # plot of prediction coverage
        coverage_scores = [random_coverage, pop_coverage, svd_coverage, svdpp_coverage, nmf_coverage]
        model_names = ['Random Recommender', 'Popularity Recommender', 'SVD CF', 'SVDpp CF', 'NMF CF']

        fig = plt.figure(figsize=(7, 5))
        recmetrics.coverage_plot(coverage_scores, model_names)

    mark()
    coverage()

def cfEval(model, n):
    print("{} TESTING".format(model.__name__))
    if model == NMF:
        ratings = ratings_df.loc[ratings_df['Rating'] == 1]
        users = ratings["User"].value_counts()
        
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(ratings, reader)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=420)

        param_grid = {'n_factors': [11,14,15,16,17,18,20]}
        gs = GridSearchCV(NMF, param_grid, measures=['rmse'], cv=5)
        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])

        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])

    else:
        ratings = ratings_df
        #ratings = ratings_df.loc[ratings_df['Rating'] == 1]
        users = ratings["User"].value_counts()
        
        reader = Reader(rating_scale=(-1, 1))
        #reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(ratings, reader)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=420)

        param_grid = {'n_factors': [4,6,9,11,14,18,29]}
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])

        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])

    algo = model(n_factors = n)
    
    algo.fit(trainset)
    test = algo.test(testset)
    test = pd.DataFrame(test)
    test.drop("details", inplace=True, axis=1)
    test.columns = ['User', 'Plant', 'actual', 'cf_predictions']

    random_df = pd.DataFrame(np.random.randint(0,2,size=(test.shape[0], 1)))
    print(random_df.head())
    print("RANDOM MSE: ", recmetrics.mse(test.actual, random_df))
    print("RANDOM RMSE: ", recmetrics.rmse(test.actual, random_df))
    print("MSE: ", recmetrics.mse(test.actual, test.cf_predictions))
    print("RMSE: ", recmetrics.rmse(test.actual, test.cf_predictions))
    test.to_csv('data/svd_test.csv', index = False)

    cf_model = test.pivot_table(index='User', columns='Plant', values='cf_predictions').fillna(0)
    test = test.copy().groupby('User', as_index=False)['Plant'].agg({'actual': (lambda x: list(set(x)))})
    test = test.set_index("User")

    cf_recs = [] = []
    for user in test.index:
        cf_predictions = get_users_predictions(user, 10, cf_model)
        cf_recs.append(cf_predictions)
        
    test['cf_predictions'] = cf_recs
    
    #POPULARITY
    #make recommendations for all members in the test data
    popularity_recs = ratings.Plant.value_counts().head(10).index.tolist()

    pop_recs = []
    for user in test.index:
        pop_predictions = popularity_recs
        pop_recs.append(pop_predictions)
        
    test['pop_predictions'] = pop_recs

    #RANDOM
    ran_recs = []
    for user in test.index:
        random_predictions = ratings.Plant.sample(10).values.tolist()
        ran_recs.append(random_predictions)
        
    test['random_predictions'] = ran_recs 

    def mark():
        actual = test.actual.values.tolist()
        cf_predictions = test.cf_predictions.values.tolist()
        pop_predictions = test.pop_predictions.values.tolist()
        random_predictions = test.random_predictions.values.tolist()

        pop_mark = []
        for K in np.arange(1, 11):
            pop_mark.extend([recmetrics.mark(actual, pop_predictions, k=K)])
        
        random_mark = []
        for K in np.arange(1, 11):
            random_mark.extend([recmetrics.mark(actual, random_predictions, k=K)])

        mark = []
        for K in np.arange(1, 11):
            mark.extend([recmetrics.mark(actual, cf_predictions, k=K)])

        mark_scores = [random_mark, pop_mark, mark]
        index = range(1,10+1)
        names = ['Random Recommender', 'Popularity Recommender', 'SVD CF']

        fig = plt.figure(figsize=(15, 7))
        recmetrics.mark_plot(mark_scores, model_names=names, k_range=index)

    def coverage():
        ##PREDICTION COVERAGE
        catalog = ratings.Plant.unique().tolist()
        random_coverage = recmetrics.prediction_coverage(ran_recs, catalog)
        pop_coverage = recmetrics.prediction_coverage(pop_recs, catalog)
        cf_coverage = recmetrics.prediction_coverage(cf_recs, catalog)

        ##CATALOG COVERAGE
        # N=100 observed recommendation lists
        random_cat_coverage = recmetrics.catalog_coverage(ran_recs, catalog, 100)
        pop_cat_coverage = recmetrics.catalog_coverage(pop_recs, catalog, 100)
        cf_cat_coverage = recmetrics.catalog_coverage(cf_recs, catalog, 100)
        
        ##COVERAGE PLOT
        # plot of prediction coverage
        coverage_scores = [random_coverage, pop_coverage, cf_coverage]
        model_names = ['Random Recommender', 'Popularity Recommender', 'CF']

        fig = plt.figure(figsize=(7, 5))
        recmetrics.coverage_plot(coverage_scores, model_names)

    def novelty():
        nov = ratings.Plant.value_counts()
        pop = dict(nov)

        random_novelty,random_mselfinfo_list = recmetrics.novelty(ran_recs, pop, len(users), 10)
        pop_novelty,pop_mselfinfo_list = recmetrics.novelty(pop_recs, pop, len(users), 10)
        cf_novelty,cf_mselfinfo_list = recmetrics.novelty(cf_recs, pop, len(users), 10)

        print(random_novelty, pop_novelty, cf_novelty)
    
    def classProb():
        class_one_probs = np.random.normal(loc=.7, scale=0.1, size=1000)
        class_zero_probs = np.random.normal(loc=.3, scale=0.1, size=1000)
        actual = [1] * 1000
        class_zero_actual = [0] * 1000
        actual.extend(class_zero_actual)

        pred_df = pd.DataFrame([np.concatenate((class_one_probs, class_zero_probs), axis=None), actual]).T
        pred_df.columns = ["predicted", "truth"]
        print(pred_df.head())

        recmetrics.class_separation_plot(pred_df, n_bins=45, title="Class Separation Plot")
    
    def ROC():
        model_probs = np.concatenate([np.random.normal(loc=.2, scale=0.5, size=500), np.random.normal(loc=.9, scale=0.5, size=500)])
        actual = [0] * 500
        class_zero_actual = [1] * 500
        actual.extend(class_zero_actual)

        recmetrics.roc_plot(actual, model_probs, model_names="one model",  figsize=(10, 5))

    mark()
    coverage()
    novelty()
    classProb()
    ROC()
    mse_model = rmse_vs_factors(model, data)
    plot_rmse(rmse_model, model.__name__)

def main():
    loadData()
    longTail()
    #seasonalPlots(guilds_df)
    #boxPlots(guilds_df) 
    #rangePlots(guilds_df)
    cfCompare()
    cfEval(SVD, 4)
    cfEval(SVDpp, 4)
    cfEval(NMF, 17)

if __name__ == '__main__':
    main()