import argparse
import os
import model as ml_model
import mlflow
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser(description="ML Project Pipeline")
    parser.add_argument('--prepare', action='store_true', help='Prepare the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--save', type=str, help='Save the model to a file')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the testing data')
    
    args = parser.parse_args()

    # Configure l'URI de suivi de MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Définir l'expérience MLflow
    mlflow.set_experiment("churn prediction")

    if args.prepare:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        print("Data prepared successfully.")

    if args.train:
        with mlflow.start_run():  # Start a new MLflow run
            X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
            model = ml_model.train_model(X_train, y_train)
            best_model = ml_model.optimize_hyperparameters(X_train, y_train)
            mlflow.sklearn.log_model(best_model, "model")  # Log the model
            print("Model trained and hyperparameters optimized successfully.")

    if args.evaluate:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        model = ml_model.load_model(args.save)
        accuracy, precision, recall, f1 = ml_model.evaluate_model(model, X_test, y_test)
        print(f'Model accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

    if args.save:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        model = ml_model.train_model(X_train, y_train)
        best_model = ml_model.optimize_hyperparameters(X_train, y_train)
        ml_model.save_model(best_model, args.save)
        print(f'Model saved to {args.save}')

if __name__ == "__main__":
    main()
