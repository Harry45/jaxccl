import os
import pickle


# This defines a few utility functions
def z2a(z):
    """converts from redshift to scale factor"""
    return 1.0 / (1.0 + z)


def a2z(a):
    """converts from scale factor to  redshift"""
    return 1.0 / a - 1.0


def save_pkl(file: dict, folder_name: str, file_name: str) -> None:
    """Stores a list/dict/class in a folder.

    Args:
        file (list): Quantity to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    os.makedirs(folder_name, exist_ok=True)

    # use compressed format to store data
    with open(folder_name + "/" + file_name + ".pkl", "wb") as f:
        pickle.dump(file, f)


def load_pkl(folder_name: str, file_name: str) -> dict:
    """Reads a list from a folder.

    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.

    Returns:
        dict: the dictionary with all the quantities.
    """
    with open(folder_name + "/" + file_name + ".pkl", "rb") as f:
        file = pickle.load(f)

    return file
