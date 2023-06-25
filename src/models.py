from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np


model_options = {
    'Логистическая регрессия': LogisticRegression(),
    'Дерево решений': DecisionTreeClassifier(),
    'Случайный лес': RandomForestClassifier(),
    'Метод опорных векторов': SVC(),
    'Наивный Байесовский классификатор': GaussianNB(),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Градиентный бустинг': GradientBoostingClassifier(),
    'Метод стохастического градиентного спуска': SGDClassifier(),
    'Многослойный перцептрон': MLPClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Мультиномиальный наивный Байес': MultinomialNB(),
}


def get_selected_model(name):
    return model_options[name]


def calculate_scores(model, X, y):
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    scores = np.sqrt(-scores)
    return scores.mean(), scores.std()