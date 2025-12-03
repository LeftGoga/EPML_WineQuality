from features import engineer_features, scale_features
from model import evaluate_model, save_model, train_model
from utils import plot_correlation_heatmap, plot_feature_importances, plot_quality_distribution

from data import create_target, get_features_and_target, load_data, split_data

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    plot_quality_distribution(df)
    df = create_target(df)
    df = engineer_features(df)

    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    model = train_model(X_train_sc, y_train, n_estimators=300)
    evaluate_model(model, X_test_sc, y_test)

    plot_correlation_heatmap(df.drop(columns=["quality", "good_quality"]))
    plot_feature_importances(model, X.columns.tolist())

    save_model(model)
