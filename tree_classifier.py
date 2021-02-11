from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


iris = load_iris()
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(data=iris.data, columns=cols)
df['target'] = iris.target

def gini_impurity(classes):
    #assert isinstance(classes, np.array), 'classes not a np.array'
    n_obs = len(classes)
    uniques, counts = np.unique(classes, return_counts=True)
    gini_components = [1] + [-(class_count/n_obs)**2 for class_count in counts]
    return sum(gini_components)

def calculate_loss_func(m_left, m_right, gini_left, gini_right):
    m = m_left + m_right
    return (m_left/m)*gini_left + (m_right/m)*gini_right


def get_split(df, features, target):
    prev_gini = gini_impurity(df[target].values)
    best_loss = np.inf
    
    for feature in features:
        unique_values = df[feature].unique()
        for value in unique_values:
            below = df[df[feature]<=value][target].values
            above = df[df[feature]>value][target].values
            gini_left = gini_impurity(below)
            gini_right = gini_impurity(above)
            loss = calculate_loss_func(len(below), len(above), gini_left, gini_right)
            if (loss<best_loss)&((gini_left+gini_right)<prev_gini):
                best_loss = loss
                best_feature = feature
                best_value = value
                
    if best_loss==np.inf:
        return dict(df.groupby(target).count().iloc[:, 0]/len(df))
    else:
        return best_feature, best_value


def get_decision_path(df, features, target):
    split = get_split(df, features, target)
    if not isinstance(split, tuple):
        return split
    left_side = df[df[split[0]]<=split[1]]
    right_side = df[df[split[0]]>split[1]]
    left_split = get_decision_path(left_side, features, target)
    right_split = get_decision_path(right_side, features, target)
    decision = [split, {'L':left_split, 'R':right_split}]
    return decision

def prepare_prediction_dict(X, decision_path):
    if isinstance(decision_path, dict):
        return decision_path
    feature, value = decision_path[0]
    dictionary = decision_path[1]
    if X[feature]<=value:
        return prepare_prediction_dict(X, dictionary['L'])
    elif X[feature]>value:
        return prepare_prediction_dict(X, dictionary['R'])

def return_preds(pred_dict):
    return max(pred_dict, key=pred_dict.get)

class CARTClassifier:
    
    def fit(self, df, features, target):
        self.features = features
        self.target = target
        self.decision_path = get_decision_path(df, features, target)
        
    def predict(self, df):
        preds = np.array([])
        for idx, row in df.iterrows():
            prediction_dict = prepare_prediction_dict(row[self.features], self.decision_path)
            pred = return_preds(prediction_dict)
            preds = np.append(preds, pred)
        return preds
    
cart = CARTClassifier()
cart.fit(df, cols, 'target')
from sklearn.metrics import accuracy_score
print(accuracy_score(df['target'], cart.predict(df)))