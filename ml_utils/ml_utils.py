import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np

class MLUtils:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier()
        self.pca = PCA(n_components=2)
        
    def preprocess_data(self, features):
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not fitted. Call 'train_knn' first.")
        if not hasattr(self.pca, 'mean_'):
            raise ValueError("PCA not fitted. Call 'train_knn' first.")
        scaled_features = self.scaler.transform(features)
        reduced_features = self.pca.transform(scaled_features)
        return reduced_features
    
    def train_knn(self, features, labels):
        # Fit the scaler with the features
        scaled_features = self.scaler.fit_transform(features)
        reduced_features = self.pca.fit_transform(scaled_features)
        
        X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)
        self.knn.fit(X_train, y_train)
        accuracy = self.knn.score(X_test, y_test)
        return accuracy
        
    def predict_knn(self, features):
        reduced_features = self.preprocess_data(features)
        return self.knn.predict(reduced_features)
    
    def save_models(self):
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.knn, 'knn.pkl')
        joblib.dump(self.pca, 'pca.pkl')
    
    def load_models(self):
        self.scaler = joblib.load('scaler.pkl')
        self.knn = joblib.load('knn.pkl')
        self.pca = joblib.load('pca.pkl')

# test the MLUtils class

if __name__ == "__main__":
    # Create a random dataset
    np.random.seed(0)
    features = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, 100)
    
    # Initialize the MLUtils class
    ml_utils = MLUtils()
    
    # Train the KNN model
    accuracy = ml_utils.train_knn(features, labels)
    print(f'Accuracy: {accuracy}')
    
    # Save the models
    ml_utils.save_models()
    
    # Load the models
    ml_utils.load_models()
    
    # Test the KNN model
    test_features = np.random.rand(10, 10)
    predictions = ml_utils.predict_knn(test_features)
    print(f'Predictions: {predictions}')
