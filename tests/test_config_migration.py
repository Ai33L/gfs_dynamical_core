import os
import pytest
import re
try:
    import tomllib
except ImportError:
    import tomli as tomllib

def get_setup_py_metadata():
    with open("setup.py", "r") as f:
        content = f.read()
    
    metadata = {}
    patterns = {
        "name": r'name=["\'](.*?)["\']',
        "version": r'version=["\'](.*?)["\']',
        "description": r'description=["\'](.*?)["\']',
        "author": r'author=["\'](.*?)["\']',
        "author_email": r'author_email=["\'](.*?)["\']',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metadata[key] = match.group(1)
    
    return metadata

def test_pyproject_toml_exists():
    assert os.path.exists("pyproject.toml")

def test_pyproject_metadata_matches_setup_py():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    
    setup_metadata = get_setup_py_metadata()
    project_info = pyproject["project"]
    
    assert project_info["name"] == setup_metadata["name"]
    assert project_info["version"] == setup_metadata["version"]
    assert project_info["description"] == setup_metadata["description"]
    assert project_info["authors"][0]["name"] == setup_metadata["author"]
    assert project_info["authors"][0]["email"] == setup_metadata["author_email"]

def test_pyproject_readme_includes_history():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    
    readme = pyproject["project"].get("readme")
    # In pyproject.toml, readme can be a string or a table
    if isinstance(readme, dict):
        assert "README.rst" in str(readme.get("file", ""))
        assert "HISTORY.rst" in str(readme.get("file", ""))
    else:
        # If it's a string, it must point to both or be dynamic
        # But for now, let's just assert it includes history
        assert "HISTORY.rst" in str(readme)

def test_pyproject_urls_match_setup_py():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    
    with open("setup.py", "r") as f:
        content = f.read()
    
    match = re.search(r'url=["\'](.*?)["\']', content)
    setup_url = match.group(1) if match else None
    
    project_urls = pyproject["project"].get("urls", {})
    assert project_urls.get("Homepage") == setup_url

def test_ruff_configured():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert "ruff" in pyproject.get("tool", {})
