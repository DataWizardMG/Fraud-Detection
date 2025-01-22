from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from scripts.model_training import print_performance_metrics

def ada_boost_tuning(x_train, y_train, x_test, y_test):
    """Perform hyperparameter tuning on AdaBoost."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    grid_search = GridSearchCV(
        AdaBoostClassifier(random_state=42),
        param_grid,
        scoring='f1_weighted',
        cv=3
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print_performance_metrics(y_test, y_pred, 'AdaBoost (Tuned)')
