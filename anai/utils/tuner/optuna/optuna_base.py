import optuna


class Tuner:
    def __init__(self, sampler, n_trials=100, direction="maximize"):
        self.n_trials = n_trials
        self.sampler = sampler
        self.direction = direction

    def tune(self, objective):
        study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        study.optimize(
            objective, n_trials=self.n_trials, n_jobs=-1, gc_after_trial=True
        )
        params = study.best_params
        best_score = study.best_value
        return params, best_score
