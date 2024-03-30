import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold


# Define cross-validation training function
def cross_val_training(model_class, data, labels, num_folds=5, epochs=10, batch_size=32):
    kf = KFold(n_splits=num_folds, shuffle=True)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for train_index, val_index in kf.split(data):
        x_train_fold, x_val_fold = data[train_index], data[val_index]
        y_train_fold, y_val_fold = labels[train_index], labels[val_index]

        model = model_class(input_shape=x_train_fold.shape[1:], num_classes=y_train_fold.shape[2])
        model.compile(lr=0.001)
        model.train(x_train_fold, y_train_fold, x_val_fold, y_val_fold, epochs=epochs, batch_size=batch_size)
        
        # Evaluate the model on the validation fold
        y_pred = model.model.predict(x_val_fold)
        y_pred = np.argmax(y_pred, axis=-1).flatten()
        y_true = np.argmax(y_val_fold, axis=-1).flatten()
        
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