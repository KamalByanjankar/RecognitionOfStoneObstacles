from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from ConfusionMatrix import create_confusion_matrix
from MachineLearning_helper import split_data, normalization

def train_MLP(data, label, normalized_X_train, train_y):   
    clf = MLPClassifier(solver='adam', batch_size=500, max_iter=500, alpha=1e-5,hidden_layer_sizes=(44), random_state=12,activation="relu")
    model = clf.fit(normalized_X_train, train_y)
    return model
    
def test_MLP(model, normalized_X_test, test_y):
    result = model.predict(normalized_X_test)
    print(confusion_matrix(result, test_y))
    print("Accuracy:", accuracy_score(result, test_y))
    create_confusion_matrix(result, test_y, 'MLP Confusion Matrix')
