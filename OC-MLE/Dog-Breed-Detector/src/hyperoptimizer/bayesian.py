import optuna


class BayesianOptimizer:

    def __init__(self, objective_function, optimization_type, pruner_type=optuna.pruners.MedianPruner()):
        self.objective_function = objective_function
        self.optimization_type = optimization_type
        self.pruner_type = pruner_type

    def build_study(self, trials, n_cores=None, verbose=False):
        study = optuna.create_study(direction=self.optimization_type, pruner=self.pruner_type)
        study.optimize(self.objective_function, n_trials=trials, n_jobs=n_cores)
        if verbose:
            self.display_study_statistics(study)
        return study

    @staticmethod
    def display_study_statistics(study):
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials),
              f' ({round(len(pruned_trials) / len(study.trials) * 100, 2)}%)')
        print("  Number of complete trials: ", len(complete_trials),
              f' ({round(len(complete_trials) / len(study.trials) * 100, 2)}%)')
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        return

    @staticmethod
    def plot_optimization_history(study):
        return optuna.visualization.plot_optimization_history(study)

    @staticmethod
    def plot_param_importances(study):
        return optuna.visualization.plot_param_importances(study)

    @staticmethod
    def plot_edf(study):
        return optuna.visualization.plot_edf(study)