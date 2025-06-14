# SuSSE
Sub-Saharan Solar Estimation

## Building & Install Package  

To install the project use pip
```
pip install .
```


If you want to install the package in developer mode, which means that the changes you make in source code will directly be reflected in the package, then use

```
pip install -e .
```


## Tox and Testing

We use Tox for managing different testing and formatting environments, ensuring that the codebase remains clean, consistent, and properly tested. Tox allows us to automate tests, static analysis, and formatting checks both locally and in our continuous integration (CI) setup on GitHub.

### Local Testing with Tox

To test your changes locally, you can use the following Tox commands:

- **Formatting Code**: 
  Ensures the code follows the correct formatting standards.
  ```bash
  tox -e format-code
  ```

- **Formatting Imports**:  
  Organizes and formats the import statements according to the project style guide.
  ```bash
  tox -e format-imports
  ```

- **Check Format**:
  This runs checks to ensure that the code and import formatting are correct, without making any changes.
  ```bash
  tox -e check-format
  ```

- **Static Analysis**: 
  Runs static analysis using MyPy to catch type errors or other issues before running tests.
  ```bash
  tox -e static-analysis
  ```

- **Running Tests**: 
  Runs the tests located in the tests folder and generates a coverage report.
  ```bash
  tox -e py312
  ```

Each of these steps is configured in the `tox.ini` file located at the root of the project.

### GitHub Workflow and Continuous Integration

When you push your code to GitHub, a series of automated checks will run through GitHub Actions. The `.github/workflows/tox.yml` file defines the steps to be performed:

1. **Tox CI**: This job runs on every push or pull request to the main branch. It installs Tox and runs all the tests, static analysis, and formatting checks using Tox.

   - It uses the same commands listed above to ensure that your branch is compliant with the project’s guidelines.

2. **Format Check**: This job ensures that the code is correctly formatted.

3. **Static Analysis**: This job runs MyPy to ensure there are no type errors in the code.

### How to Contribute: Git Workflow

When making changes, **do not push directly to the `main` branch**. Follow these steps instead:

1. **Create a New Branch**: Your branch name should start with your initials followed by a descriptive name (e.g., `jm/new_feature`).

    ```bash
    git checkout -b jm/new_feature
   ```

2. **Make Your Changes**: After you’ve made changes, ensure your branch is compliant by running all the necessary Tox commands locally:

    ```bash
    tox -e format-code
    tox -e format-imports
    tox -e check-format
    tox -e static-analysis
    tox -e py312  # to run the tests
   ```

3. **Push Your Branch**: Push the branch to GitHub.

    ```bash
    git push origin jm/new_feature
   ```
   
4. **Create a Pull Request**: Once the branch is pushed, create a pull request and assign me (your supervisor) as a reviewer. Make sure that all checks on GitHub pass before requesting a review.

This workflow ensures that the code remains high-quality, consistent, and error-free before merging into the main branch.