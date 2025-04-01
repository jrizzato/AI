import unittest
from src.model_training import train_model
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset for testing
        self.X, self.y = make_classification(n_samples=100, n_features=4, n_classes=3, random_state=42)
        self.model = RandomForestClassifier()

    def test_train_model(self):
        # Train the model
        trained_model = train_model(self.model, self.X, self.y)
        
        # Check if the model is fitted
        self.assertTrue(hasattr(trained_model, "estimators_"))

    def test_model_accuracy(self):
        # Train the model
        trained_model = train_model(self.model, self.X, self.y)
        
        # Calculate accuracy
        accuracy = trained_model.score(self.X, self.y)
        
        # Check if accuracy is greater than a threshold
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()