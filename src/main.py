import argparse
import os
import model as ml_model
import tensorflow as tf
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="ML Project Pipeline")
    parser.add_argument('--prepare', action='store_true', help='Prepare the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--save', type=str, help='Save the model to a file')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the testing data')
    
    args = parser.parse_args()

    # Configure TensorBoard logging
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn prediction")

    if args.prepare:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        print("Data prepared successfully.")

    if args.train:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(args.train_path, args.test_path)
        model = ml_model.train_model(X_train, y_train)
        best_model = ml_model.optimize_hyperparameters(X_train, y_train)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters and metrics to MLflow
            mlflow.log_param("n_estimators", best_model.get_params()['n_estimators'])
            mlflow.log_param("max_depth", best_model.get_params()['max_depth'])
            mlflow.log_param("min_samples_split", best_model.get_params()['min_samples_split'])
            mlflow.log_param("min_samples_leaf", best_model.get_params()['min_samples_leaf'])

            accuracy, precision, recall, f1 = ml_model.evaluate_model(best_model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model with signature
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

            # Register the model
            model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)
            registered_model = mlflow.register_model(model_uri, "ChurnPredictionModel")
            
            # Transition model to staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="ChurnPredictionModel",
                version=registered_model.version,
                stage="None"
            )

            # Log metrics to TensorBoard
            with file_writer.as_default():
                tf.summary.scalar('accuracy', accuracy, step=1)
                tf.summary.scalar('precision', precision, step=1)
                tf.summary.scalar('recall', recall, step=1)
                tf.summary.scalar('f1_score', f1, step=1)
        
        # Save the model with versioning
        model_version_dir = "models/version_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(model_version_dir, exist_ok=True)
        model_path = os.path.join(model_version_dir, "best_model.pkl")
        ml_model.save_model(best_model, model_path)
        
        print(f"Model trained and hyperparameters optimized successfully. Model saved to {model_path}")

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
