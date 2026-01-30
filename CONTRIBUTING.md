# Contributing to OmniNexus

Thank you for your interest in contributing to OmniNexus! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (Python version, OS, etc.)
   - Relevant code snippets or error messages

### Suggesting Features

1. Check existing issues and discussions for similar suggestions
2. Create a new issue with:
   - Clear description of the feature
   - Use case / motivation
   - Proposed implementation (if applicable)
   - Potential impact on existing functionality

### Submitting Code

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/NEURALMORPHIC-FIELDS/OmniNexus_Fragmergent_Adaptive_Systems.git
cd OmniNexus_Fragmergent_Adaptive_Systems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

#### Development Workflow

1. **Fork the repository** on GitHub

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the code style guidelines (below)
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests**:
   ```bash
   pytest tests/
   ```

5. **Run code quality checks**:
   ```bash
   # Formatting
   black omninexus/ tests/

   # Linting
   flake8 omninexus/ tests/

   # Type checking
   mypy omninexus/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Wait for review

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use Black for formatting (line length: 88)
- Use meaningful variable and function names
- Maximum line length: 88 characters

### Documentation

- All public functions must have docstrings (Google style)
- Include type hints for function parameters and return values
- Update README.md for significant feature changes

### Example Docstring

```python
def calculate_coherence(history: List[float], window: int = 20) -> float:
    """
    Calculate temporal coherence of signal history.

    Coherence measures the predictability of recent values based on
    standard deviation within a sliding window.

    Args:
        history: List of historical signal values.
        window: Number of recent samples to analyze. Defaults to 20.

    Returns:
        Coherence value in range [0, 1], where 1 indicates
        perfectly stable signal and 0 indicates high variability.

    Raises:
        ValueError: If window is larger than history length.

    Example:
        >>> history = [0.1, 0.2, 0.15, 0.18, 0.12]
        >>> coherence = calculate_coherence(history, window=5)
        >>> print(f"Coherence: {coherence:.3f}")
    """
    pass
```

### Commit Message Format

We follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(oscillator): add frequency modulation support
fix(avatar): correct energy calculation boundary
docs(readme): update installation instructions
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `tests/test_oscillator.py`)
- Use descriptive test names
- Test both success and failure cases

### Test Example

```python
import pytest
from omninexus.components.oscillator import FragmergentOscillator


class TestFragmergentOscillator:
    """Tests for FragmergentOscillator class."""

    def test_step_returns_float(self):
        """step() should return a float value."""
        osc = FragmergentOscillator()
        phi = osc.step()
        assert isinstance(phi, float)

    def test_invalid_harmonic_layers_raises(self):
        """Invalid harmonic_layers should raise ValueError."""
        with pytest.raises(ValueError):
            FragmergentOscillator(harmonic_layers=10)

    def test_coherence_range(self):
        """Coherence should always be in [0, 1]."""
        osc = FragmergentOscillator()
        for _ in range(100):
            osc.step()
        coherence = osc.get_phase_coherence()
        assert 0.0 <= coherence <= 1.0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=omninexus --cov-report=html

# Run specific test file
pytest tests/test_oscillator.py

# Run with verbose output
pytest -v
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] No unrelated changes included

### PR Description Template

```markdown
## Description
Brief description of changes.

## Related Issues
Fixes #123

## Changes Made
- Change 1
- Change 2

## Testing
How were the changes tested?

## Screenshots (if applicable)
```

## Questions?

- Open a GitHub Discussion for general questions
- Create an Issue for bugs or specific feature requests
- Contact the maintainers for sensitive matters

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section

Thank you for helping improve OmniNexus!
