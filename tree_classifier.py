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


def split(df, features, target):
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
        return 'done'
    else:
        return best_feature, best_value
    
    
'''predykcja:
-> calc initial gini
-> get next split
'''