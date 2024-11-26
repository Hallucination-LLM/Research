import os

def setup_environment(
    device_num: int = 0, tokenizers_parallelism: bool = False
) -> str:
    """
    Setup environment for GPU operations

    Args:
        device_num (int): GPU device number
        tokenizers_parallelism (bool): Tokenizers parallelism flag

    Returns:
        str: Device name
    """

    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_num}"
    # os.environ["TOKENIZERS_PARALLELISM"] = f"{tokenizers_parallelism}".lower()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        raise RuntimeError("No GPU available. Exiting...")
    
    
    device = device if device_num == "auto" else f"cuda"
    print(f"Using device: {device}, {device_num = }")

    return device
