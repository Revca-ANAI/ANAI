from colorama import Fore


class Best:
    """
    Best is used to utilise the best model when predictor = 'all' is used.

    """

    def __init__(self, best_model, tune, isReg=False):
        self.__best_model = best_model
        self.model = self.__best_model["Model"]
        self.name = self.__best_model["Name"]
        self.tune = tune
        if isReg:
            self.r2_score = self.__best_model["R^2 Score"]
            self.mae = self.__best_model["Mean Absolute Error"]
            self.rmse = self.__best_model["Root Mean Squared Error"]
        if not isReg:
            self.accuracy = self.__best_model["Accuracy"]
        self.kfold_acc = self.__best_model["Cross Validated Accuracy"]
        if tune == True:
            self.best_params = self.__best_model["Best Parameters"]
            self.best_accuracy = self.__best_model["Best Accuracy"]
        else:
            self.best_params = "Run with tune = True to get best parameters"
        self.isReg = isReg

    def summary(self):
        """Returns a summary of the best model"""
        print("\nBest Model Summary:")
        print(Fore.CYAN + "Name: ", self.name)
        if self.isReg:
            print(Fore.CYAN + "R^2 Score: ", self.r2_score)
            print(Fore.CYAN + "Mean Absolute Error: ", self.mae)
            print(Fore.CYAN + "Root Mean Squared Error: ", self.rmse)
        else:
            print(Fore.CYAN + "Accuracy: ", self.accuracy)
        print(Fore.CYAN + "Cross Validated Accuracy: ", self.kfold_acc)
        if self.tune:
            print(Fore.CYAN + "Best Parameters: ", self.best_params)
            print(Fore.CYAN + "Best Accuracy: ", self.best_accuracy)
        print("\n")

    def predict(self, pred):
        """Predicts the output of the best model"""
        prediction = self.__best_model["Model"].predict(pred)
        return prediction
