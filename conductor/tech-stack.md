# Tech Stack: gfs-dynamical-core

## Languages & Core Runtime
- **Python**: Primary language for the API and high-level logic.
- **Cython**: Used for performance-critical bridging between Python and Fortran.
- **Fortran**: The legacy GFS dynamical core implementation.

## Data & Array Backends
- **NumPy**: The default array backend.
- **unyt**: Unit handling and metadata provider for the `unyt` array backend.
- **pint**: Alternative unit handling (legacy support).
- **xarray**: Data structures for multi-dimensional scientific data.

## Domain Frameworks
- **sympl**: System for Modelling Planets - provides the component architecture.
- **climt**: Climate Modelling Toolkit - the parent project for this core.

## Development & Build Tools
- **pyproject.toml**: Unified configuration for project metadata, dependencies, and tools.
- **setuptools**: Build backend for package distribution.
- **Cython**: Compilation of Python extensions.
- **GNU Make**: Build automation for the Fortran/Cython components.

## Testing & Quality Assurance
- **pytest**: Primary testing framework.
- **Ruff**: Fast, unified linter and formatter for Python.
- **tox**: Test automation across different environments.

## Documentation
- **Sphinx**: Documentation generator.
- **ReadTheDocs**: Hosting platform for documentation.
