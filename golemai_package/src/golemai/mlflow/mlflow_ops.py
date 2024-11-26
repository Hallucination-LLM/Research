import os

import mlflow

MLFLOW_URI = "http://172.16.2.203:5000"


def get_next_run_name(experiment_id: str, experiment_name) -> int:
    """
    Get the next run name for the experiment

    Args:
        experiment_id (str): The experiment id
        experiment_name (str): The experiment name

    Returns:
        str: The next run name
    """
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    run_numbers = []
    for run in runs:
        try:
            if "mlflow.runName" in run.data.tags:
                run_number = int(
                    run.data.tags.get("mlflow.runName").split("_")[-1]
                )
                run_numbers.append(run_number)
        except (ValueError, AttributeError, IndexError) as e:
            # Handle the exception (e.g., log it, ignore it, etc.)
            print(f"Error processing run: {e}")

    if run_numbers:
        num = max(run_numbers) + 1
    else:
        num = 1

    return f"{experiment_name}_{num}"


def get_experiment_id(exp_name: str, artifact_location: str) -> int:
    """
    Get the experiment ID by name. If the experiment does not exist, create it.

    Args:
        exp_name (str): Name of the experiment.
        artifact_location (str): Location of the artifacts.
    Returns:
        int: Experiment ID.
    """

    if experiment := mlflow.get_experiment_by_name(exp_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(
            exp_name,
            artifact_location=artifact_location,
        )


def set_mlflow(
    tracking_uri: str = "http://172.16.2.203:5000",
    exp_name: str = None,
    artifact_location: str = None,
) -> int:
    """
    Set the MLflow tracking URI and create an experiment if it does not exist.
    """

    mlflow.set_tracking_uri(tracking_uri)

    return get_experiment_id(exp_name, artifact_location)


def enable_artifacts(
    artifact_root: str = "/home/mlflow/mlflow/artifacts",
) -> None:
    """
    Enable artifacts by setting the permissions for the artifact root directory

    Args:
        artifact_root (str): The artifact root directory
    """

    if not os.path.exists(artifact_root):
        raise FileNotFoundError(f"Directory {artifact_root} does not exist")

    try:
        os.system(f"sudo chmod -R 775 {artifact_root}")
        print("Command executed successfully")
    except Exception as e:
        print(f"An error occurred: {e}")
