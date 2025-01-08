import numpy as np
from collections import Counter
from DecisionTree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def bootstrap_sample(X,y):
    n_samples=X.shape[0]
    idxs=np.random.choice(n_samples,size=n_samples,replace=True)
    #advance numpy indexing
    return X[idxs],y[idxs]

def most_common_label(y):
    counter=Counter(y)
    return counter.most_common(1)[0][0]

class RandomForest:
    def __init__(self,n_trees=100,min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees=n_trees
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_feats=n_feats
        self.trees=[]

    def fit(self,X,y):
        #reset tree list
        self.trees=[]
    
        for _ in range(self.n_trees):
            tree=DecisionTree(min_samples_split=self.min_samples_split,
                              max_depth=self.max_depth,
                              n_feats=self.n_feats)
            X_sample,y_sample=bootstrap_sample(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)



    def predict(self,X):
        tree_preds=np.array(tree.predict(X) for tree in self.trees)
        #suppose we have 4 samples, tree_preds will return [[1,0,1,0],[1,1,1,1],[0,0,1,0]]. But what we want is [[1,0,1,0]]
        #The shape changes from (n_trees, n_samples) → (n_samples, n_trees).
        tree_preds=np.swapaxes(tree_preds,0,1)
        y_pred=[most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

data = datasets.load_breast_cancer()
X, y = data.data, data.target
#print(X.shape)




X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

clf = RandomForest(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)