import numpy as np
import pandas as pd




def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values
    y = df_y.values
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)

print(X_train_norm.shape)

# print(y_test)
# print(y_test.ravel())
print(X_test_norm)
print(y_test.size)
print(y_test.ravel().size)

unique, counts = np.unique(y_test, return_counts=True)
print(unique)
print(counts)
