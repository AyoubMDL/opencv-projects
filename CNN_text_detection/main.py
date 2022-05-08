from scripts.cnn_training import *

if __name__ == '__main__':
    images, labels, num_classes = get_labeled_data("./resources")
    X_train, X_test, X_validation, y_train, y_test, y_validation, num_samples = split_data(images, labels, num_classes, 0.2, 0.2)
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_validation = preprocess(X_validation)
    y_train = encode(y_train, num_classes)
    y_validation = encode(y_validation, num_classes)
    y_test = encode(y_test, num_classes)

    model = cnn_model(num_classes)
    # model.summary()
    history, model = run_model(model, X_train, y_train, X_validation, y_validation)

    display_score(model, X_test, y_test)
    plot_results(history)
    save_model(model, "model")