from sklearn.decomposition import FactorAnalysis

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html


class FA:
    def __init__(self, df, n_comp):
        self.X = df.values
        self.fa = FactorAnalysis(n_components=n_comp, random_state=42)
