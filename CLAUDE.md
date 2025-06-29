# OCR Compare Project Guidelines

## Build & Test Commands
- Install: `pip install -e .`
- Run tests: `pytest`
- Run single test: `pytest tests/path_to_test.py::test_function_name -v`
- Lint: `flake8 src/`
- Type check: `mypy src/`

## Code Style Guidelines
- **Imports**: Group imports: stdlib → third-party → local. Sort alphabetically within groups.
- **Type Annotations**: Use static typing with all function parameters and return values.
- **Naming**: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_SNAKE_CASE
- **Documentation**: Docstrings for all public functions and classes (""" style).
- **Error Handling**: Use specific exceptions, handle explicitly. Abstract classes should raise NotImplementedError.
- **Class Structure**: Follow abstract base class pattern for interfaces like StorageHandler.
- **Path Handling**: Use pathlib.Path over string paths.