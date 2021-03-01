from sklearn.decomposition import NMF

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html


class NonNegativeMatrixFactorization:

    def __init__(self, df,  n_comp=None):
        self.X = df.values
        self.nmf = NMF(n_components=n_comp, init='random', random_state=42)
