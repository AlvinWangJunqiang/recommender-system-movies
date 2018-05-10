from surprise import SVD, accuracy, Reader, Dataset
from surprise.model_selection import KFold
import pandas as pd

#Load CSV train data
cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
df = pd.read_csv('train_data.csv',sep=',',header=0
                 ,encoding='latin-1',names=cols,parse_dates=True) 
df.rating = round(df.rating,0)
df = df.drop('unix_timestamp',axis=1)

# Prepare the data to be used in Surprise
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split the dataset into 5 folds and choose the algorithm
kf = KFold(n_splits=5)

# Use the famous SVD algorithm.
algo = SVD()


for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)

# Predict a certain item
userid = str(196)
itemid = str(302)
actual_rating = 4
print(algo.predict(userid, itemid, actual_rating))

#Read test data
cols_test = ['user_id', 'item_id', 'unix_timestamp']
df_test = pd.read_csv('test_data.csv',sep=',',header=0
                 ,encoding='latin-1',names=cols_test,parse_dates=True)
df_test['rating'] = 1
df_test = df_test.drop('unix_timestamp',axis=1)


# Prepare the data to be used in Surprise
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data_test = Dataset.load_from_df(df[['user_id', 'item_id','rating']], reader)

#Test data rating prediction
pred = []
for i in range(len(df_test)):
    userid = df_test.user_id[i]
    itemid = df_test.item_id[i]
    actual_rating = df_test.rating[i]
    pred.append(algo.predict(userid, itemid, actual_rating)[3])
df_test.rating = pred
df_test.rating = round(df_test.rating,0)

#Write into a new csv file
df_test.to_csv('test_data_with_ratings.csv', sep=',')