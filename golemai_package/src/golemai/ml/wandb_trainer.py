from matplotlib import pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import wandb.sklearn

class WandbTrainer:
    SEED = 42
    CROSSVALIDATION_DATASETS = {
        'LOOKBACK_LENS': ['nq', 'cnndm'],
        'QA': ['nq', 'bioask', 'hotpotqa_en', 'hotpotqa_pl', 'polqa', 'poquad_v2'],
        'SUM': ['cnndm', 'xsum'],
    }

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
            "auc": roc_auc_score(y_true, y_pred),
            "auprc": average_precision_score(y_true, y_pred),
        }

    def _prepare_datasets(self, train_ds, test_ds):
        """
        Prepare datasets by splitting features and labels.

        Args:
            train_ds (pd.DataFrame): Training dataset.
            test_ds (pd.DataFrame): Test dataset.

        Returns:
            tuple: Prepared training and test datasets with features and labels.
        """

        # Split data into features and labels
        X_train, y_train = train_ds.drop(columns=['label', 'dataset']), train_ds['label']
        X_test, y_test = test_ds.drop(columns=['label', 'dataset']), test_ds['label']

        return X_train, y_train, X_test, y_test

    def _initialize_wandb_run(self, model, dataset_dict, description, group='test', job_type=None, **kwargs):
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
            group=group,
            job_type=job_type,
            name=f'{group}_{job_type}_{wandb.util.generate_id()}',
            config={
                **model.get_params(),
                "dataset_description": dataset_dict,
                "description": description,
                **kwargs,
            },
        )

    def plot_roc_curves(self, y_true_dict, y_proba_dict):
        """
        Create ROC curves for K-Fold cross-validation across test, train, and validation sets.

        Args:
            y_true_dict (dict): Dictionary of true labels for each dataset type and fold.
            y_proba_dict (dict): Dictionary of predicted probabilities for each dataset type and fold.
        Returns:
            dict: Dictionary of matplotlib figure objects for each dataset type.
        """
        # Dataset types to plot
        dataset_types = ['train', 'validation', 'test']
        
        # Store figures
        roc_figures = {}


        for dataset_type in dataset_types:
            aucs = []
            plt.figure(figsize=(10, 8))
            plt.axes().set_aspect('equal', 'datalim')

            tprs = []
            base_fpr = np.linspace(0, 1, 101)

            # Plot individual fold ROC curves
            for i in range(len(y_true_dict[dataset_type])):
                y_true = y_true_dict[dataset_type][i]
                y_proba = y_proba_dict[dataset_type][i]
                
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                plt.plot(fpr, tpr, 'b', alpha=0.15)
                
                # Interpolate ROC curve
                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

                aucs.append(roc_auc_score(y_true, y_proba))

            # Calculate mean and standard deviation of ROC curves
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            std = tprs.std(axis=0)

            tprs_upper = np.minimum(mean_tprs + std, 1)
            tprs_lower = mean_tprs - std

            # Calculate AUC
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

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
            plt.title(f'ROC Curves for - {dataset_type.capitalize()} Set\n'
                  f'Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
            plt.legend(loc='lower right')

            # Store the figure
            roc_figures[dataset_type] = plt.gcf()

        return roc_figures

    def train_model_and_evaluate_kfold(self, model, X_train, y_train, X_test, y_test, k_folds=5):
        """
        Train the specified model and evaluate it using KFold cross-validation.

        Args:
            model: Machine learning model or pipeline.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.
            k_folds (int, optional): Number of folds for cross-validation.
        Returns:
            dict: Aggregated metrics for train, validation, and test sets.
            dict: True labels for each set.
            dict: Predicted probabilities for each set.
        """

        sets = ["train", "validation", "test"]

        results = {set_type: [] for set_type in sets}
        all_y_true = {set_type: [] for set_type in sets}
        all_y_proba = {set_type: [] for set_type in sets}

        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.SEED)

        for train_index, val_index in kfold.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            predictions = {
                "train": model.predict_proba(X_train_fold)[:, 1],
                "validation": model.predict_proba(X_val_fold)[:, 1],
                "test": model.predict_proba(X_test)[:, 1],
            }

            for set_type, y_true in zip(
                sets,
                [y_train_fold, y_val_fold, y_test]
                ):

                results[set_type].append(self.compute_metrics(y_true, predictions[set_type]))

                all_y_proba[set_type].append(predictions[set_type])
                all_y_true[set_type].append(y_true)



        return results, all_y_true, all_y_proba
        

    def evaluate(self, model, dataset: pd.DataFrame, description: str, group_name: str = 'test', k_folds=5, **kwargs):
        """
        Evaluate the specified model on the provided dataset.

        Args:
            model: Machine learning model or pipeline.
            dataset (pd.DataFrame): Dataset to evaluate the model on.
            description (str): Description of the evaluation run.
            run_name (str, optional): Specific name for the WandB run.

        Returns:
            dict: Metrics logged to WandB.
        """
        group_name = f'{group_name}_{wandb.util.generate_id()}'
        
        for evaluation_type, datasets in WandbTrainer.CROSSVALIDATION_DATASETS.items():
            all_results, all_y_true, all_y_proba = [], [], []

            dataset_dict = {
                f'{dataset_name}':{
                    "size": len(dataset[dataset['dataset'] == dataset_name]),
                    'positive_ratio': dataset[dataset['dataset'] == dataset_name]['label'].mean(),
                    'negative_ratio': 1 - dataset[dataset['dataset'] == dataset_name]['label'].mean()
                } 
                for dataset_name in datasets
            }


            self._initialize_wandb_run(
                    model, 
                    dataset_dict, 
                    group=group_name, 
                    job_type=evaluation_type,
                    description=description, 
                    k_folds=k_folds, 
                    **kwargs
            )
            
            for test_dataset in datasets:
                train_ds = dataset[(dataset['dataset'] != test_dataset) & (dataset['dataset'].isin(datasets))]
                test_ds = dataset[dataset['dataset'] == test_dataset]

                if train_ds.empty or test_ds.empty:
                    print(f"Skipping {evaluation_type} {test_dataset} as it is not present in the dataset")
                    continue

                X_train, y_train, X_test, y_test= self._prepare_datasets(train_ds, test_ds)

                results, y_true, y_proba = self.train_model_and_evaluate_kfold(
                    model, 
                    X_train, 
                    y_train, 
                    X_test, 
                    y_test, 
                    k_folds=k_folds
                )

                # Log metrics to WandB
                flattened_results = {f"{set_type}_{metric}": [result[metric] for result in results[set_type]] for set_type in results for metric in results[set_type][0]}

                all_results.append(flattened_results)
                all_y_true.append(y_true)
                all_y_proba.append(y_proba)

            
            # Mean+std of metrics

            if not all_results:
                print(f"No results for {evaluation_type}")
                wandb.finish()
                continue

            mean = {f'{metric}_mean': np.mean([result[metric] for result in all_results]) for metric in all_results[0]}
            std = {f'{metric}_std': np.std([result[metric] for result in all_results]) for metric in all_results[0]}

            wandb.log({**mean, **std})


            # merge all y_true and y_proba for plotting

            all_y_true_dict = {set_type: [ y_true_dict[set_type] for y_true_dict in all_y_true] for set_type in all_y_true[0]}
            all_y_proba_dict = {set_type: [ y_proba_dict[set_type] for y_proba_dict in all_y_proba] for set_type in all_y_proba[0]}

            all_y_true_dict = {set_type: [ y_true for y_true_list in all_y_true_dict[set_type] for y_true in y_true_list] for set_type in all_y_true_dict}
            all_y_proba_dict = {set_type: [ y_proba for y_proba_list in all_y_proba_dict[set_type] for y_proba in y_proba_list] for set_type in all_y_proba_dict}

            roc_figures = self.plot_roc_curves(all_y_true_dict, all_y_proba_dict)
            for set_type, roc_figure in roc_figures.items():
                wandb.log({f'{set_type}_roc': wandb.Image(roc_figure)})

            wandb.finish()
            


