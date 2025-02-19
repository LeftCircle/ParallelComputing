"""
Utility functions for path manipulation.
"""

import os


def get_file_extension(file_path: str) -> str:
	"""
	Get the file extension from a file path.

	Args:
		file_path (str): The file path to extract the extension from.

	Returns:
		str: The file extension.
	"""
	return os.path.splitext(file_path)[1]


def get_file_name_without_extension(file_path: str) -> str:
	"""
	Get the file name without the extension.
	ex: C:/path/to/file.txt -> file

	Args:
		file_path (str): The file path to extract the name from.

	Returns:
		str: The file name without the extension.
	"""
	return os.path.splitext(os.path.basename(file_path))[0]


def create_folder_if_not_exists(folder_path: str) -> None:
	"""
	Create a folder if it does not already exist.

	Args:
		folder_path (str): The folder path to create.
	"""
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)


def get_file_number_suffix(file_path: str) -> int:
	"""
	Get the number suffix for a file.

	Args:
		file_path (str): The path to the file to get the number suffix from.

	Returns:
		int: The number suffix.
	"""
	name : str = os.path.basename(file_path)
	extension : str = get_file_extension(name)
	name_munus_extension : str = name.replace(extension, "")
	number_str : str = name_munus_extension.split("_")[-1]
	if number_str.isdigit():
		return int(number_str)
	return 0


def remove_file_number_suffix(file_path: str) -> str:
	"""
	Remove the number suffix from a file name.

	Args:
		file_path (str): The path to the file to remove the number suffix from.

	Returns:
		str: The new file path with the number suffix removed.
	"""
	name : str = get_file_name_without_extension(file_path)
	extension : str = get_file_extension(file_path)
	number_str : str = name.split("_")[-1]
	if number_str.isdigit():
		tokens: list[str] = name.split("_")[:-1]
		name = "_".join(tokens)
	new_path : str = os.path.join(os.path.dirname(file_path), name + extension)
	return new_path


def _rename_until_new_number(file_path: str, count : int = 0) -> str:
	"""
	Rename a file until a new number is found.

	Args:
		file_path (str): The path to the file to rename.
		count (int): The number suffix to start with.
	
	Returns:
		str: The new path with the new name.
	"""
	dir : str = os.path.dirname(file_path)
	name : str = get_file_name_without_extension(file_path)
	extension : str = get_file_extension(file_path)
	count += 1
	new_name : str = f"{name}_{count:02}{extension}"
	new_path : str = os.path.join(dir, new_name)
	while os.path.exists(new_path):
		assert(count < 100, "Too many files with the same name.")
		new_name = f"{name}_{count:02}{extension}"
		count += 1
		new_path = os.path.join(dir, new_name)
	return new_path


def add_numbers_if_already_exists(file_path: str) -> str:
	"""
	Get the new name for a file based on the file path.
	If a file with the same name exists, the new name will have 
	a numeric suffix.

	Args:
		file_path (str): The path to the file to rename.

	Returns:
		str: the new path for the file.
	"""
	if not os.path.exists(file_path):
		return file_path
	file_number : int = get_file_number_suffix(file_path)
	new_path : str = _rename_until_new_number(file_path, file_number)
	return new_path


def add_file_to_subdir(file_path: str, base_dir: str, subpath: str) -> None:
	"""
	Adds a file to a subdirectory.

	Args:
		file_path (str): The path to the file to move.
		subpath (str): The subdirectory to move the file to.
	"""
	name : str = os.path.basename(file_path)
	folder_dir : str = os.path.join(base_dir, subpath)
	new_file_path : str = os.path.join(base_dir, subpath, name)
	create_folder_if_not_exists(folder_dir)
	new_file_path = add_numbers_if_already_exists(new_file_path)
	os.rename(file_path, new_file_path)


def get_file_name_for_subdir(file_path: str,
							 base_dir: str, subpath: str) -> str:
	"""
	Get the new file name for a file in a subdirectory.

	Args:
		file_path (str): The path to the file to rename.
		base_dir (str): The base directory for the file.
		subpath (str): The subdirectory for the file.

	Returns:
		str: The new path for the file.
	"""
	name : str = os.path.basename(file_path)
	new_file_path : str = os.path.join(base_dir, subpath, name)
	new_file_path = add_numbers_if_already_exists(new_file_path)
	return new_file_path


def add_prefix_from_extension(file_path: str,
							  extension_prefix: dict) -> str:
	"""
	Add a prefix to a file based on the extension.

	Args:
		file_path (str): The path to the file to rename.
		extension_prefix (dict): The extension prefix dictionary.

	Returns:
		str: The new path for the file.
	"""
	extension : str = get_file_extension(file_path)
	if extension in extension_prefix:
		prefix : str = extension_prefix[extension]
		new_name : str = f"{prefix}_{os.path.basename(file_path)}"
		dir : str = os.path.dirname(file_path)
		return os.path.join(dir, new_name)
	return file_path


def remove_strings_from_name(file_path: str, remove_str: list[str]) -> str:
	"""
	Remove strings from a file name that are not allowed.
	ex: if 'small' is in the remove_str list, 
	    C:/path/to_small/file_small_diffuse.png -> 
	    C:/path/to_small/file_diffuse.png

	Args:
		file_path (str): The path to the file to rename.
		remove_str (list): The list of strings to remove.

	Returns:
		str: The new path for the file.
	"""
	name : str = get_file_name_without_extension(file_path)
	extension : str = get_file_extension(file_path)
	name_parts : list[str] = name.split("_")
	name_parts = [
		part for part in name_parts if part.lower() not in remove_str
	]
	name = "_".join(name_parts)
	return os.path.join(os.path.dirname(file_path), f"{name}{extension}")


def replace_strings_in_name(file_path: str, replace_str: dict) -> str:
	"""
	Replace strings in a file name.
	ex: if {'diffuse' : "C"} is in the replace_str dictionary,
		C:/path/to_diffuse/file_diffuse.png -> C:/path/to_diffuse/file_C.png

	Args:
		file_path (str): The path to the file to rename.
		replace_str (dict): The dictionary of strings to replace.

	Returns:
		str: The new path for the file.
	"""
	name : str = get_file_name_without_extension(file_path)
	extension : str = get_file_extension(file_path)
	name_parts : list[str] = name.split("_")
	for i, part in enumerate(name_parts):
		if part.lower() in replace_str:
			name_parts[i] = replace_str[part.lower()]
	name = "_".join(name_parts)
	dir : str = os.path.dirname(file_path)
	return os.path.join(dir, f"{name}{extension}")


def clean_underscores_from_name(file_path: str) -> str:
	"""
	Clean up underscores from a file name.
	ex: C:/path/to__file__diffuse.png -> C:/path/to/file_diffuse.png

	Args:
		file_path (str): The path to the file to rename.

	Returns:
		str: The new path for the file.
	"""
	name : str = get_file_name_without_extension(file_path)
	extension : str = get_file_extension(file_path)
	while "__" in name:
		name = name.replace("__", "_")
	if name.startswith("_"):
		name = name[1:]
	if name.endswith("_"):
		name = name[:-1]
	dir : str = os.path.dirname(file_path)
	return os.path.join(dir, f"{name}{extension}")

def get_all_files_of_suffix(folder_path: str, suffix: str) -> list[str]:
	"""
	Get all files in a folder with a specific suffix.

	Args:
		folder_path (str): The path to the folder to search.
		suffix (str): The suffix to search for.

	Returns:
		list: The list of files with the specified suffix.
	"""
	files : list[str] = []
	for root, _, file_names in os.walk(folder_path):
		for file_name in file_names:
			if file_name.endswith(suffix):
				files.append(os.path.join(root, file_name))
	return files

def get_current_folder() -> str:
	"""
	Get the current folder.

	Returns:
		str: The current folder.
	"""
	return os.path.dirname(os.path.abspath(__file__))