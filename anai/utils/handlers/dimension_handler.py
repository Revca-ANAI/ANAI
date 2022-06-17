from anai.utils.configs import *
from colorama import Fore
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time


class DimensionHandler:

    def __init__(self) -> None:
        self.X_train_original = None
        self.y_train_original = None
        self.X_val_original = None
        self.y_val_original = None
        self.y_train = None
        self.X_train = None
        self.X_val = None
        self.y_val = None

    def dimensionality_reduction(
        self,
        lda,
        pca,
        X_train,
        X_val,
        y_train,
        n_components_lda,
        n_components_pca,
        pca_kernel,
        start,
    ):
        """
        Performs Dimensionality Reduction on Training and Validation independent variables.
        """
        try:
            if lda:
                lda = LDA(n_components=n_components_lda)
                X_train = lda.fit_transform(X_train, y_train)
                X_val = lda.transform(X_val)
            if pca and not lda :
                if not pca_kernel == "linear":
                    try:

                        kpca = KernelPCA(
                            n_components=n_components_pca, kernel=pca_kernel
                        )
                        X_train = kpca.fit_transform(X_train)
                        X_val = kpca.transform(X_val)
                    except MemoryError as error:
                        print(error)
                        end = time.time()
                        print("Time Elapsed :", end - start)
                        return

                elif pca_kernel == "linear":
                    pca = PCA(n_components=n_components_pca)
                    X_train = pca.fit_transform(X_train)
                    X_val = pca.transform(X_val)
                else:
                    print("Un-identified PCA Kernel\n")
                    return
            return (X_train, X_val)
        except Exception as error:
            print(
                Fore.RED + "Dimensionality Reduction Failed with error :", error, "\n"
            )
            return (X_train, X_val)
