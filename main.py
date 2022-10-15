import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse

def decision_tree(x_train, y_train, x_test, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train,y_train)
    dt_predictions = dt_model.predict(x_test)
    print(accuracy_score(y_test ,dt_predictions))
    print(confusion_matrix(y_test,dt_predictions))
    
def log_regression(x_train, y_train, x_test, y_test):
    log_model = LogisticRegression()
    log_model.fit(x_train,y_train)
    log_predictions = log_model.predict(x_test)
    print(accuracy_score(y_train,log_predictions))
    print(confusion_matrix(y_test,log_predictions))
    
def random_forest(x_train, y_train, x_test, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_predictions = rfc.predict(x_test)
    print(accuracy_score(y_test, rfc_predictions))
    print(confusion_matrix(y_test,rfc_predictions))
    
def support_vector(x_train, y_train, x_test, y_test):
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(x_train, y_train)
    c = classifier.predict(x_test)
    print(accuracy_score(y_test, c))
    print(confusion_matrix(y_test,c))


df = pd.read_csv("data/extracted_features.csv")
x = df[['url_length','hostname_length',
       'path_length', 'fd_length', 'tld_length',  'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www','count-digits',
       'count-letters', 'count_dir', 'use_of_ip', 'short_url']]

y = df['result']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("-c", "--classifier", required=True, help="dt: decision_tree, rt: random_forest, lr: log_regression, sv: support_vector")
    args = vars(arg.parse_args())

    if args['classifier'] == 'dt':
        decision_tree(x_train, y_train, x_test, y_test)
    elif args['classifier'] == 'rt':
        random_forest(x_train, y_train, x_test, y_test)
    elif args['classifier'] == 'lr':
        log_regression(x_train, y_train, x_test, y_test)
    elif args['classifier'] == 'sv':
        support_vector(x_train, y_train, x_test, y_test)
    else:
        print("Incorrect Module: Type --help with file name to see options.")