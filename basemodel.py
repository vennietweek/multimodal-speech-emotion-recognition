from keras.models import Model
from keras.optimizers.legacy import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class BaseModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        pass

    def compile(self, lr=0.001):
        self.model.compile(
            optimizer=Adam(lr=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=-1).flatten()
        y_true = np.argmax(y_test, axis=-1).flatten()

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        # Return a dictionary of metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
    
    def predict(self, x_new):
        return self.model.predict(x_new)

    def train_and_evaluate_on_test(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        # Train the model on the entire training dataset
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Evaluate the model on the test dataset
        test_metrics = self.evaluate(x_test, y_test)

        print("Test Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1_score']:.4f}")

        return history, test_metrics
