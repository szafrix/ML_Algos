from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

wine = load_wine()
cols = ['alcohol',
  'malic_acid',
  'ash',
  'alcalinity_of_ash',
  'magnesium',
  'total_phenols',
  'flavanoids',
  'nonflavanoid_phenols',
  'proanthocyanins',
  'color_intensity',
  'hue',
  'od280/od315_of_diluted_wines',
  'proline']
df = pd.DataFrame(data=wine.data, columns=cols)
df['target'] = wine.target

#def gini_impurity(classes):
#    #assert isinstance(classes, np.array), 'classes not a np.array'
#    n_obs = len(classes)
#    uniques, counts = np.unique(classes, return_counts=True)
#    gini_components = [1] + [-(class_count/n_obs)**2 for class_count in counts]
#    return sum(gini_components)

def gini_impurity(classes):
    #assert isinstance(classes, np.array), 'classes not a np.array'
    n_obs = len(classes)
    uniques, counts = np.unique(classes, return_counts=True)
    return 1 + np.sum(-np.square(counts/n_obs))

def calculate_loss_func(m_left, m_right, gini_left, gini_right):
    m = m_left + m_right
    return (m_left/m)*gini_left + (m_right/m)*gini_right


def get_split(df, features, target):
    prev_gini = gini_impurity(df[target].values)
    best_loss = np.inf
    
    for feature in features:
        unique_values = np.sort(df.loc[:, feature].unique())
        for value in unique_values:
            below = np.sort(df.loc[df[feature]<=value, target].values)
            above = np.sort(df.loc[df[feature]>value, target].values)
            gini_left = gini_impurity(below)
            gini_right = gini_impurity(above)
            loss = calculate_loss_func(len(below), len(above), gini_left, gini_right)
            if (loss<best_loss)&(max(gini_left, gini_right)<prev_gini):
                best_loss = loss
                best_feature = feature
                best_value = value
                
    if best_loss==np.inf:
        return dict(df.groupby(target).count().iloc[:, 0]/len(df))
    else:
        return best_feature, best_value

def check_for_tree_depth(decision_path, left_path, right_path, path_list):
    print(left_path, right_path)
    if isinstance(decision_path, list):
        return check_for_tree_depth(decision_path[1], left_path, right_path, path_list)
    elif isinstance(decision_path, dict):
        left_split = decision_path['L']
        left_path += 1
        right_split = decision_path['R']
        right_path += 1
        if isinstance(left_split, dict):
            path_list.append(left_path)
        elif isinstance(left_split, list):
            return check_for_tree_depth(left_split, left_path, right_path, path_list)
        
        if isinstance(right_split, dict):
            path_list.append(right_path)
        elif isinstance(right_split, list):
            return check_for_tree_depth(right_split, left_path, right_path, path_list)

def check_depth(decision, current_depth=''):
    if isinstance(decision, dict):
        return current_depth
    else:
        next_path = decision[1]
        left_ = next_path['L']
        right_ = next_path['R']
        #current_depth += 1
        left_depth = check_depth(left_, current_depth+'L')
        right_depth = check_depth(right_, current_depth+'R')
        return max(left_depth, right_depth)

def get_decision_path(df, features, target):
    split = get_split(df, features, target)
    if not isinstance(split, tuple):
        return split
    left_side = df.loc[df[split[0]]<=split[1], :]
    right_side = df.loc[df[split[0]]>split[1], :]
    left_split = get_decision_path(left_side, features, target)
    right_split = get_decision_path(right_side, features, target)
    decision = [split, {'L':left_split, 'R':right_split}]
    #check depth
    depth = check_depth(decision, '')
    print(depth)
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

from time import time

time1 = time()
cart = CARTClassifier()
cart.fit(df, cols, 'target')
p = cart.predict(df)
time2 = time()
print(time2-time1)

time1 = time()
skcart = DecisionTreeClassifier()
skcart.fit(df[cols], df['target'])
p = skcart.predict(df[cols])
time2 = time()
print(time2-time1)
print('acc sklearn: ', accuracy_score(df['target'], p))