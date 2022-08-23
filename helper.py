

import matplotlib.pyplot as plt


def plot_prediction(train_data= X_train,
                    train_labels= y_train,
                    test_data=X_test,
                    test_labels= y_test,
                    file_name = file_name,
                    predictions=None):
    """:cvar
    Plots training data, test data and compares predictions. 
    """
    
    plt.figure(figsize=(10, 7))

    #Plot the training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data ")

    # Plot the testing data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data ")


    #Are there prediction?
    if predictions is not None:
        # Plot the predictions if they existing
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions ")

    #Show the legend
    plt.legend(prop={"size":14});
    plt.savefig('/home/mhamdan/DeepLearning_PyTorch_2022/output/'+file_name+'.png')
