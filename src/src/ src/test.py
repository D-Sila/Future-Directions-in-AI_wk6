from sklearn.metrics import classification_report, confusion_matrix
# After model.predict on test set:
y_true = [...]
y_pred = [...]
print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))
