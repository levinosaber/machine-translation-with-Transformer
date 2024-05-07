import torch 


class EarlyStopping:
    '''
    Early stop the training if current metric is worse than the best one for longer than
    number of wait_epochs or if metric stops changing.
    params:
        wiat_epochs: int, optional (default=2)
            Number of epochs to wait for the metric to improve before stopping the training.
    '''
    def __init__(self, wait_epochs=4) -> None:
        self.wait_epochs = wait_epochs  
        self.num_bad_scores = 0
        self.num_constant_scores = 0
        self.best_score = None
        self.best_metric = None

    def stop(self, metric, model, metric_type="better_decrease", delta=0.03):
        '''
        stop the training if metric criteria aren't met
        params:
            metric: float
                current calculated metric used to evaluate the validation performance.
            model: torch.nn.Module
                model instance
            metric_type: str, optional (default="better_decrease")
                Type of metric to use for early stopping. 
                "better_decrease": metric should be decreasing
                "better_increase": metric should be increasing
            delta: float, optional (default=0.03)
                Minimum change in metric to consider as improvement
        Returns:
            bool: True if the training should stop, False otherwise.
        '''
        self.delta = delta
        delta = self.delta * metric

        if self.best_score is None:
            self.best_score = metric
            self.save_model_state(metric, model)
            return False

        if abs(metric - self.best_score) < delta / 3 * metric:
            self.num_constant_scores += 1
            if self.num_constant_scores >= self.wait_epochs + 1:
                print("\n Training stopped by early stopping")
                return True
            else:
                self.num_constant_scores = 0
        
        if metric_type == "better_decrease":
            if metric > self.best_score + delta:
                self.num_bad_scores += 1
            elif metric > self.best_score:
                self.num_bad_scores = 0
            else:
                self.best_score = metric
                self.save_model_state(metric, model)
                return False
        else:
            # better_increase  mode
            if metric < self.best_score - delta:
                self.num_bad_scores += 1
            elif metric < self.best_score:
                self.num_bad_scores = 0
            else:
                self.best_score = metric
                self.save_model_state(metric, model)
                self.num_bad_scores = 0

        if self.num_bad_scores >= self.wait_epochs:
            print("\n Training stopped by early stopping")
            return True
        
        return False

    def save_model_state(self, metric, model):
        '''
        save the best model state
        '''
        torch.save(model.state_dict(), "best_model.pt")
        self.best_metric = metric