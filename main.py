import sys
from scripts.preprocessing import preprocess_data, load_data, get_column_definitions, split_data
from scripts.visualization import visualize_data
from scripts.model_training import logistic_regression_pipeline, ada_boost_pipeline, save_best_model
from scripts.hyperparameter_tuning import ada_boost_tuning
from scripts.deployment.app import run_app

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        # Run the Flask app
        run_app()
    else:
        # Load and preprocess data
        df = load_data("data/Transactions_Dataset.csv")
        cat_cols, num_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order = get_column_definitions()
        visualize_data(df, cat_cols, num_cols)
        X, y = preprocess_data(df, cat_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order)

        # Train/test split
        x_train, x_test, y_train, y_test = split_data(X, y)

        # Train and evaluate models
        logistic_regression_pipeline(x_train, y_train, x_test, y_test)
        best_model = ada_boost_pipeline(x_train, y_train, x_test, y_test)

        # Save the best model
        save_best_model(best_model, "models/best_model.pkl")

        # Perform hyperparameter tuning
        #ada_boost_tuning(x_train, y_train, x_test, y_test)
