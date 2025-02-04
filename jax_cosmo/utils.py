import os
import pickle
import dill
from typing import Any

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

# def save_dill(file: dict, folder_name: str, file_name: str) -> None:
#     """Stores a list/dict/class in a folder.

#     Args:
#         file (list): Quantity to store.
#         folder_name (str): The name of the folder.
#         file_name (str): The name of the file.
#     """

#     # create the folder if it does not exist
#     os.makedirs(folder_name, exist_ok=True)

#     # use compressed format to store data
#     with open(os.path.join(folder_name, file_name), "wb") as f:
#         dill.dump(file, f)


# def load_dill(folder_name: str, file_name: str) -> dict:
#     """Reads a list from a folder.

#     Args:
#         folder_name (str): The name of the folder.
#         file_name (str): The name of the file.

#     Returns:
#         dict: the dictionary with all the quantities.
#     """
#     with open(os.path.join(folder_name, file_name), "rb") as f:
#         file = dill.load(f)
#     return file

# class CustomPickler(pickle.Pickler):
#     def persistent_id(self, obj):
#         """
#         Define custom behavior for persistent_id.
#         This method is called during pickling to assign persistent IDs to specific objects.
#         """
#         # Example: Handle custom serialization for certain object types
#         if hasattr(obj, "custom_id"):  # Replace with your specific logic
#             return obj.custom_id  # Assign a unique identifier for the object
#         return None  # Default behavior (no special handling)


# class CustomUnpickler(pickle.Unpickler):
#     def persistent_load(self, pid):
#         """
#         Define custom behavior for persistent_load.
#         This method is called during unpickling to load objects based on their persistent ID.
#         """
#         # Example: Resolve persistent IDs back to objects
#         # Replace with your specific logic for loading objects
#         return {"custom_id": pid}  # Example: Restore an object from its ID


# def save_pkl(file: Any, folder_name: str, file_name: str) -> None:
#     """
#     Stores an object (list/dict/class/etc.) in a folder as a pickle file, using a custom pickler.

#     Args:
#         file (Any): Object to store.
#         folder_name (str): The name of the folder.
#         file_name (str): The name of the file (without extension).
#     """
#     # Create the folder if it does not exist
#     os.makedirs(folder_name, exist_ok=True)

#     # Build the file path
#     file_path = os.path.join(folder_name, file_name + ".pkl")

#     try:
#         # Use the custom pickler to save the file
#         with open(file_path, "wb") as f:
#             CustomPickler(f).dump(file)
#         print(f"File saved successfully: {file_path}")
#     except Exception as e:
#         print(f"Error saving file to {file_path}: {e}")


# def load_pkl(folder_name: str, file_name: str) -> Any:
#     """
#     Reads an object (list/dict/class/etc.) from a pickle file in a folder, using a custom unpickler.

#     Args:
#         folder_name (str): The name of the folder.
#         file_name (str): The name of the file (without extension).

#     Returns:
#         Any: The loaded object.
#     """
#     # Build the file path
#     file_path = os.path.join(folder_name, file_name + ".pkl")

#     try:
#         # Use the custom unpickler to load the file
#         with open(file_path, "rb") as f:
#             file = CustomUnpickler(f).load()
#         print(f"File loaded successfully: {file_path}")
#         return file
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#     except Exception as e:
#         print(f"Error loading file from {file_path}: {e}")
