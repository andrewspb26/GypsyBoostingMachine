from sklearn.tree import DecisionTreeRegressor
from collections import Counter
from matplotlib import pyplot as plt

# loading dataset
path = "http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv"
df = pd.read_csv(path)


#time features extraction
df['PublishDate'] = pd.to_datetime(df['PublishDate'])
df['dow'] = df['PublishDate'].dt.dayofweek
df['hour'] = df['PublishDate'].dt.hour


#1 topic & source frequency encoding
src_count = Counter(df.Source)
encoded_src = {k:v/df.shape[0] for k, v in src_count.items()}

tpc_count = Counter(df.Topic)
encoded_tpc = {k:v/df.shape[0] for k, v in tpc_count.items()}

#2 topic and source mean value by group encoding
encoded_src = df.groupby('Source')['Facebook'].mean().to_dict()
encoded_tpc = df.groupby('Topic')['Facebook'].mean().to_dict()

df['Source'] = df['Source'].map(encoded_src)
df['Topic'] = df['Topic'].map(encoded_tpc)


# target, features
y = np.array(df['Facebook'])
df = df[['Source', 'Topic', 'dow', 'hour', 'SentimentTitle', 
         'SentimentHeadline', 'GooglePlus', 'LinkedIn']]
df.fillna(0, inplace=True)
X = np.array(df.astype(float))


# learning  
def loss(y, f):
    return 0.5*(y - f)**2

# ordering boosting
lossc = []
gb = GypsyBoost(loss, DecisionTreeRegressor(max_depth=5))
for iter_ in gb.grow_ensemble(100, X, y, validation=0.3, shuffle=False, ordering=True):
    lossc.append(np.log(iter_))

# simple boosting
lossc1 = []    
gb = GypsyBoost(loss, DecisionTreeRegressor(max_depth=5))
for iter_ in gb.grow_ensemble(100, X, y, validation=0.3, shuffle=False, ordering=False):
    lossc1.append(np.log(iter_))

plt.plot(lossc)
plt.plot(lossc1)

