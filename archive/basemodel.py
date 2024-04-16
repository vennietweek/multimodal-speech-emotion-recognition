import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
from keras.optimizers.legacy import Adam
import tensorflow as tf 

class BaseModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        pass

    def compile(self, lr=0.001):
        self.model = self.build_model()
        self.model.compile(
            optimizer=Adam(lr=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=-1).flatten()
        y_true = np.argmax(y_test, axis=-1).flatten()

        mask = (y_true != 0) & (y_pred != 0)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        print("Unique labels in y_true after masking:", np.unique(y_true))
        print("Unique labels in y_pred after masking:", np.unique(y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        self.test_metrics =  {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cm': confusion_matrix(y_true, y_pred)
        }
       
    def cross_val_train(self, data, labels, num_folds=5, epochs=10, batch_size=32):
        kf = KFold(n_splits=num_folds, shuffle=True)
        accuracies, precisions, recalls, f1_scores = [], [], [], []

        for train_index, val_index in kf.split(data):
            x_train_fold, x_val_fold = data[train_index], data[val_index]
            y_train_fold, y_val_fold = labels[train_index], labels[val_index]

            # Reset the model for each fold
            self.model = self.build_model()
            self.compile(lr=0.001)

            # Train the model on the fold's training data
            self.model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold), epochs=epochs, batch_size=batch_size, verbose=1)

            # Evaluate the model on the validation fold
            y_pred = self.model.predict(x_val_fold)
            y_pred = np.argmax(y_pred, axis=-1).flatten()
            y_true = np.argmax(y_val_fold, axis=-1).flatten()

            mask = (y_true != 0) & (y_pred != 0)
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            
            # Calculate and store metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
            
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Calculate average metrics across all folds
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)

        # Calculate standard deviations
        std_accuracy = np.std(accuracies)
        std_precision = np.std(precisions)
        std_recall = np.std(recalls)
        std_f1 = np.std(f1_scores)

        print(f"K-Fold Cross-Validation Results ({num_folds} folds):")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")

    def train_and_evaluate_on_test(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Evaluate the model on the test dataset
        self.evaluate(x_test, y_test)

    def print_metrics(self):
        print("Test Metrics:")
        print(f"Accuracy: {self.test_metrics['accuracy']:.4f}")
        print(f"Precision: {self.test_metrics['precision']:.4f}")
        print(f"Recall: {self.test_metrics['recall']:.4f}")
        print(f"F1 Score: {self.test_metrics['f1_score']:.4f}")

        class_labels = ['hap', 'ang', 'sad', 'neu']
        print("Confusion Matrix:")
        plt.figure(figsize=(8,6))
        sns.heatmap(self.test_metrics['cm'], annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
