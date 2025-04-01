import unittest
import pandas as pd
from src.model_evaluation import evaluate_model

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.y_true = [0, 1, 2, 0, 1, 2]
        self.y_pred = [0, 2, 1, 0, 0, 2]
        self.expected_accuracy = 0.5  # Example expected accuracy

    def test_accuracy(self):
        accuracy = evaluate_model(self.y_true, self.y_pred)['accuracy']
        self.assertAlmostEqual(accuracy, self.expected_accuracy, places=2)

    def test_confusion_matrix(self):
        confusion_matrix = evaluate_model(self.y_true, self.y_pred)['confusion_matrix']
        expected_confusion_matrix = [[2, 0, 0],
                                      [1, 1, 1],
                                      [0, 1, 1]]
        pd.testing.assert_frame_equal(pd.DataFrame(confusion_matrix), pd.DataFrame(expected_confusion_matrix))

    def test_classification_report(self):
        report = evaluate_model(self.y_true, self.y_pred)['classification_report']
        self.assertIn('accuracy', report)

if __name__ == '__main__':
    unittest.main()