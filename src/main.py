 import model as ml_model

# Load data
train_data, test_data = ml_model.load_data('data/churn-bigml-80.csv', 'data/churn-bigml-20.csv')

# Preprocess data
train_data, scaler = ml_model.preprocess_data(train_data)
test_data, _ = ml_model.preprocess_data(test_data)

# Split data
X_train, X_val, y_train, y_val = ml_model.split_data(train_data)

# Train model
model, params = ml_model.train_model(X_train, y_train)

# Optimize hyperparameters
best_model = ml_model.optimize_hyperparameters(X_train, y_train)

# Evaluate the model
accuracy, precision, recall, f1 = ml_model.evaluate_model(best_model, X_val, y_val)
print(f'Model accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the model
ml_model.save_model(best_model, 'models/best_model.pkl')

# Load the model
loaded_model = ml_model.load_model('models/best_model.pkl')

# Verify loaded model
X_test = test_data.drop(columns=['Churn'])
y_test = test_data['Churn']
accuracy_loaded, precision_loaded, recall_loaded, f1_loaded = ml_model.evaluate_model(loaded_model, X_test, y_test)
print(f'Loaded model accuracy: {accuracy_loaded}')
print(f'Precision: {precision_loaded}')
print(f'Recall: {recall_loaded}')
print(f'F1 Score: {f1_loaded}')
