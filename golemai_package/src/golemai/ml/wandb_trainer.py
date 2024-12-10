from matplotlib import pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

class WandbTrainer:
    SEED = 42

    def __init__(self, project_name, api_token=None):
        """
        Initialize WandB Trainer.

        Args:
            project_name (str): Name of the WandB project.
            api_token (str, optional): WandB API token for authentication.
        """
        self.project_name = project_name
        if api_token:
            wandb.login(key=api_token)

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """
        Compute AUC and AUPRC metrics.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted probabilities.

        Returns:
            dict: Computed metrics.
        """
        return {
            "auc": roc_auc_score(y_true, y_pred[:, 1]),
            "auprc": average_precision_score(y_true, y_pred[:, 1]),
        }

    @staticmethod
    def log_curves(model, X_data, y_data, dataset_type):
        """
        Log ROC and Precision-Recall curves to WandB.

        Args:
            model: Trained model.
            X_data (array-like): Feature set.
            y_data (array-like): Labels.
            dataset_type (str): Dataset type ('train', 'val', 'test').
        """
        roc_curve = RocCurveDisplay.from_estimator(model, X_data, y_data)
        pr_curve = PrecisionRecallDisplay.from_estimator(model, X_data, y_data)
        wandb.log({
            f"roc_curve_{dataset_type}": wandb.Image(roc_curve.figure_),
            f"pr_curve_{dataset_type}": wandb.Image(pr_curve.figure_),
        })

    def _prepare_datasets(self, train_ds, test_ds):
        """
        Prepare datasets by splitting features and labels.

        Args:
            train_ds (pd.DataFrame): Training dataset.
            test_ds (pd.DataFrame): Test dataset.

        Returns:
            tuple: Prepared training and test datasets with features and labels.
        """
        # Log dataset information
        train_datasets = train_ds.groupby(['dataset', 'label']).size()
        test_datasets = test_ds.groupby(['dataset', 'label']).size()
        print(f"Train datasets: {train_datasets}")
        print(f"Test datasets: {test_datasets}")

        train_datasets_dict = {str(key): value for key, value in train_datasets.to_dict().items()}
        test_datasets_dict = {str(key): value for key, value in test_datasets.to_dict().items()}

        # Split data into features and labels
        X_train, y_train = train_ds.drop(columns=['label', 'dataset']), train_ds['label']
        X_test, y_test = test_ds.drop(columns=['label', 'dataset']), test_ds['label']

        return X_train, y_train, X_test, y_test, train_datasets_dict, test_datasets_dict

    def _initialize_wandb_run(self, model, train_datasets_dict, test_datasets_dict, description, run_name, **kwargs):
        """
        Initialize WandB run with model and dataset configurations.

        Args:
            model: Machine learning model.
            train_datasets_dict (dict): Training dataset configuration.
            test_datasets_dict (dict): Test dataset configuration.
            description (str): Description of the training run.
            run_name (str, optional): Specific name for the WandB run.
        """
        wandb.init(
            project=self.project_name,
            name=run_name,  # Added run name parameter
            config={
                **model.get_params(),
                "dataset_description": {
                    "train_datasets": train_datasets_dict,
                    "test_datasets": test_datasets_dict
                },
                "description": description,
                **kwargs,
            },
        )

    def train_model_and_evaluate(self, model, train_ds, test_ds,run_name=None, validation_size=0.2, description=""):
        """
        Train the specified model and evaluate it.

        Args:
            model: Machine learning model or pipeline.
            train_ds (pd.DataFrame): Training dataset.
            test_ds (pd.DataFrame): Test dataset.
            validation_size (float): Validation dataset size as a fraction of training data.
            description (str): Description of the training run.
            run_name (str, optional): Specific name for the WandB run.

        Returns:
            dict: Metrics logged to WandB.
        """
        X_train, y_train, X_test, y_test, train_datasets_dict, test_datasets_dict = self._prepare_datasets(train_ds, test_ds)

        # Split train data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, 
            y_train, 
            test_size=validation_size, 
            stratify=y_train, 
            random_state=self.SEED
        )

        # Initialize WandB run
        self._initialize_wandb_run(model, train_datasets_dict, test_datasets_dict, description, run_name, validation_size=validation_size)

        # Train model
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)
        y_pred_val = model.predict_proba(X_val)
        y_pred_test = model.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            "train": self.compute_metrics(y_train, y_pred_train),
            "validation": self.compute_metrics(y_val, y_pred_val),
            "test": self.compute_metrics(y_test, y_pred_test),
        }
        flat_metrics = {f"{key}_{subkey}": value for key, submetrics in metrics.items() for subkey, value in submetrics.items()}

        # Log metrics
        wandb.log(flat_metrics)

        # Plot and log curves
        wandb.sklearn.plot_roc(y_test, y_pred_test)
        wandb.sklearn.plot_precision_recall(y_test, y_pred_test)
        self.log_curves(model, X_train, y_train, "train")
        self.log_curves(model, X_val, y_val, "val")
        self.log_curves(model, X_test, y_test, "test")

        wandb.finish()
        return flat_metrics

    def plot_roc_curves(self, y_true_dict, y_proba_dict, k_folds):
        """
        Create ROC curves for K-Fold cross-validation across test, train, and validation sets.

        Args:
            y_true_dict (dict): Dictionary of true labels for each dataset type and fold.
            y_proba_dict (dict): Dictionary of predicted probabilities for each dataset type and fold.
            k_folds (int): Number of folds used in cross-validation.

        Returns:
            dict: Dictionary of matplotlib figure objects for each dataset type.
        """
        # Dataset types to plot
        dataset_types = ['train', 'validation', 'test']
        
        # Store figures
        roc_figures = {}

        for dataset_type in dataset_types:
            plt.figure(figsize=(10, 8))
            plt.axes().set_aspect('equal', 'datalim')

            tprs = []
            base_fpr = np.linspace(0, 1, 101)

            # Plot individual fold ROC curves
            for i in range(k_folds):
                y_true = y_true_dict[dataset_type][i]
                y_proba = y_proba_dict[dataset_type][i]
                
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                plt.plot(fpr, tpr, 'b', alpha=0.15)
                
                # Interpolate ROC curve
                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            # Calculate mean and standard deviation of ROC curves
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            std = tprs.std(axis=0)

            tprs_upper = np.minimum(mean_tprs + std, 1)
            tprs_lower = mean_tprs - std

            # Plot mean ROC curve with confidence interval
            plt.plot(base_fpr, mean_tprs, 'b', label='Mean ROC')
            plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label='±1 std dev')

            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
            
            # Formatting
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title(f'ROC Curves for {k_folds}-Fold Cross-Validation - {dataset_type.capitalize()} Set')
            plt.legend(loc='lower right')

            # Store the figure
            roc_figures[dataset_type] = plt.gcf()

        return roc_figures

    def train_model_and_evaluate_kfold(self, model, train_ds, test_ds, run_name=None, description="", k_folds=5):
        """
        Train the specified model and evaluate it using KFold cross-validation.

        Args:
            model: Machine learning model or pipeline.
            train_ds (pd.DataFrame): Training dataset.
            test_ds (pd.DataFrame): Test dataset.
            description (str): Description of the training run.
            run_name (str, optional): Specific name for the WandB run.

        Returns:
            dict: Metrics logged to WandB.
        """
        X_train, y_train, X_test, y_test, train_datasets_dict, test_datasets_dict = self._prepare_datasets(train_ds, test_ds)

        # Initialize WandB run
        self._initialize_wandb_run(model, train_datasets_dict, test_datasets_dict, run_name=run_name, description=description, k_folds=k_folds)

        results = {
            "train": [],
            "validation": [],
            "test": [],
        }

        # Dictionaries to store fold-wise true labels and probabilities for ROC curves
        all_y_true = {
            "train": [],
            "validation": [],
            "test": []
        }
        all_y_proba = {
            "train": [],
            "validation": [],
            "test": []
        }

        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.SEED)

        for train_index, val_index in kfold.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict probabilities
            y_pred_train = model.predict_proba(X_train_fold)
            y_pred_val = model.predict_proba(X_val_fold)
            y_pred_test = model.predict_proba(X_test)

            # Store true labels and probabilities for ROC curves
            all_y_true["train"].append(y_train_fold)
            all_y_true["validation"].append(y_val_fold)
            all_y_true["test"].append(y_test)

            all_y_proba["train"].append(y_pred_train[:, 1])
            all_y_proba["validation"].append(y_pred_val[:, 1])
            all_y_proba["test"].append(y_pred_test[:, 1])

            # Calculate metrics
            metrics = {
                "train": self.compute_metrics(y_train_fold, y_pred_train),
                "validation": self.compute_metrics(y_val_fold, y_pred_val),
                "test": self.compute_metrics(y_test, y_pred_test),
            }
            for key, submetrics in metrics.items():
                results[key].append(submetrics)

        # Create ROC curve plots for train, validation, and test sets
        roc_figures = self.plot_roc_curves(all_y_true, all_y_proba, k_folds)

        # Log ROC plots to WandB
        for dataset_type, fig in roc_figures.items():
            wandb.log({f"roc_curve_{dataset_type}": wandb.Image(fig)})
            plt.close(fig)

        # Mean and std of metrics
        flat_metrics = {}
        for key, submetrics in results.items():
            mean_metrics = {f"{key}_{subkey}": np.mean([m[subkey] for m in submetrics]) for subkey in submetrics[0].keys()}
            std_metrics = {f"{key}_{subkey}_std": np.std([m[subkey] for m in submetrics]) for subkey in submetrics[0].keys()}
            flat_metrics.update(mean_metrics)
            flat_metrics.update(std_metrics)


        # Log metrics
        wandb.log(flat_metrics)

        wandb.finish()
        return flat_metrics
        