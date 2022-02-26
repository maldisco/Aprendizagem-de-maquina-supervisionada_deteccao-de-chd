import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def evaluate(Y_validation, predictions):
    # Matriz de confusão
    # fonte:  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
    ConfusionMatrixDisplay.from_predictions(Y_validation, predictions)
    pyplot.show()

    # Curva ROC AUC
    # fonte: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
    fpr, tpr, thresholds = roc_curve(Y_validation, predictions)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    pyplot.show()


def analisis(features, feature_names):
    # plotagem das quantidades médias
    pyplot.figure(figsize=(17, 5))
    pyplot.bar([i+1 for i in range(9)], [f.mean()
               for f in features.transpose()], tick_label=feature_names, width=0.8)
    pyplot.xlabel('Variáveis')
    pyplot.ylabel('Média')
    pyplot.show()

    # plotagem do desvio padrão
    pyplot.figure(figsize=(17, 5))
    pyplot.bar([i+1 for i in range(9)], [f.std()
               for f in features.transpose()], tick_label=feature_names, width=0.8)
    pyplot.xlabel('Variáveis')
    pyplot.ylabel('Desvio padrão')
    pyplot.show()


def id3_decision_tree(X, y):
    print("================= Árvore de decisão ID3 =======================")

    # Separação dos dados entre 90% para treino e 10% para validação
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.10, random_state=1)

    # Árvore de decisão (ID3) com 10 rodadas de validação cruzada
    # fonte: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    model = DecisionTreeClassifier(criterion='entropy')
    cv_results = cross_val_score(model, X_train, Y_train, cv=10)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    evaluate(Y_validation, predictions)


def random_forest_all_features(X, y):
    print("================= Floresta randômica =======================")
    # Separação dos dados entre 90% para treino e 10% para validação
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.10, random_state=1)

    # Floresta randômica com 10 rodadas de validação cruzada
    # fonte: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    model = RandomForestClassifier(
        criterion='entropy', n_estimators=100, max_features=9)
    cv_results = cross_val_score(model, X_train, Y_train, cv=10)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    evaluate(Y_validation, predictions)


def random_forest_sqrt_features(X, y, feature_names):
    print("================= Floresta randômica (sqrt) =======================")

    # separação dos dados entre 90% para treino e 10% para validação
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.10, random_state=1)

    model = RandomForestClassifier(
        criterion='entropy', random_state=1, max_features=3, n_estimators=100)

    # validação cruzada
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=10, scoring='roc_auc')

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Impressão das duas variáveis mais importantes
    # fonte: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    print("Variáveis mais importantes:")
    # lista numérica de variáveis mais importantes gerada pelo scikitlearn
    importances = list(model.feature_importances_)
    # lista de tuplas com nomes da variável e sua importancia
    feature_importances = [(feature, round(importance, 2))
                           for feature, importance in zip(feature_names, importances)]
    # ordenação
    feature_importances = sorted(
        feature_importances, key=lambda x: x[1], reverse=True)
    # impressão das duas variáveis mais importantes
    [print('{:20} Importância: {}'.format(*pair))
     for pair in feature_importances[:2]]

    evaluate(Y_validation, predictions)


url = "C:/Users/filip/OneDrive/Documentos/H4CK3RMAN/Projetos FSI/chd-detection/SA_heart.csv"
dataset = read_csv(url, header=0, usecols=[i+1 for i in range(10)])

y = np.array(dataset['chd'])
dataset = dataset.drop('chd', axis=1)
feature_names = list(dataset.columns)
features = np.array(dataset)


questoes = [
    'Análise estatística inicial',
    'Árvore de decisão ID3 (curva ROC, ROC AUC e matriz de confusão)',
    'Floresta randômica com todas as variáveis preditoras',
    'Floresta randômica com a raíz quadrada das variáveis preditoras',
    'Sair'
]
q = 0
while q != 5:
    print("================= Projeto 1 =================")
    for i in range(5):
        print(f"{i+1} - {questoes[i]}")
    print("Digite a questão a ser acessada (ou 5 para sair): ")
    q = int(input())
    if q == 1:
        analisis(features, feature_names)
    elif q == 2:
        id3_decision_tree(features, y)
    elif q == 3:
        random_forest_all_features(features, y)
    elif q == 4:
        random_forest_sqrt_features(features, y, feature_names)
