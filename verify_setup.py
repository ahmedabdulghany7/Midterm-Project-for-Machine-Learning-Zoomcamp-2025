#!/usr/bin/env python
"""
Setup Verification Script
Checks if all required files and dependencies are in place.
"""

import os
import sys


def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking Python Dependencies...")
    print("-" * 60)

    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn',
        'flask',
        'openpyxl'
    ]

    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False

    return all_installed


def check_project_structure():
    """Check if all required files exist."""
    print("\nChecking Project Structure...")
    print("-" * 60)

    checks = [
        ('Concrete_Data.xls', 'Dataset file'),
        ('train.py', 'Training script'),
        ('predict.py', 'Prediction API script'),
        ('requirements.txt', 'Requirements file'),
        ('Dockerfile', 'Dockerfile'),
        ('README.md', 'README file'),
        ('QUICKSTART.md', 'Quick start guide'),
        ('.gitignore', 'Git ignore file'),
        ('test_api.py', 'API test script'),
        ('deploy/docker-compose.yml', 'Docker Compose config'),
        ('deploy/aws_deploy.sh', 'AWS deployment script'),
        ('deploy/gcp_deploy.sh', 'GCP deployment script'),
        ('deploy/heroku_deploy.sh', 'Heroku deployment script'),
        ('deploy/README.md', 'Deployment README'),
    ]

    all_exist = True
    for filepath, description in checks:
        if not check_file(filepath, description):
            all_exist = False

    return all_exist


def check_model_trained():
    """Check if model has been trained."""
    print("\nChecking Model Artifacts...")
    print("-" * 60)

    model_files = [
        'models/best_model.pkl',
        'models/scaler.pkl',
        'models/feature_names.pkl'
    ]

    all_exist = True
    for filepath in model_files:
        if not os.path.exists(filepath):
            print(f"✗ {filepath} - NOT FOUND")
            all_exist = False
        else:
            print(f"✓ {filepath}")

    if not all_exist:
        print("\n⚠ Model not trained yet. Run: python train.py")

    return all_exist


def check_jupyter_notebook():
    """Check if Jupyter notebook exists."""
    print("\nChecking Jupyter Notebook...")
    print("-" * 60)

    notebooks = [
        f for f in os.listdir('.')
        if f.endswith('.ipynb') and 'checkpoint' not in f
    ]

    if notebooks:
        for nb in notebooks:
            print(f"✓ Found notebook: {nb}")
        return True
    else:
        print("✗ No Jupyter notebook found")
        return False


def check_docker():
    """Check if Docker is available."""
    print("\nChecking Docker...")
    print("-" * 60)

    try:
        import subprocess
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Docker is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Docker command failed")
            return False
    except FileNotFoundError:
        print("⚠ Docker not found (optional)")
        return False


def print_next_steps(model_trained):
    """Print next steps based on verification results."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if not model_trained:
        print("\n1. Train the model:")
        print("   python train.py")
        print("\n2. Start the API server:")
        print("   python predict.py")
        print("\n3. Test the API:")
        print("   python test_api.py")
    else:
        print("\n✓ Everything is set up!")
        print("\nTo start using the application:")
        print("   python predict.py")
        print("\nTo test the API:")
        print("   python test_api.py")

    print("\n" + "="*60)


def main():
    """Run all checks."""
    print("="*60)
    print("CONCRETE STRENGTH PREDICTION - SETUP VERIFICATION")
    print("="*60)

    # Run checks
    deps_ok = check_dependencies()
    structure_ok = check_project_structure()
    model_ok = check_model_trained()
    notebook_ok = check_jupyter_notebook()
    docker_ok = check_docker()

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    checks = {
        'Dependencies': deps_ok,
        'Project Structure': structure_ok,
        'Model Trained': model_ok,
        'Jupyter Notebook': notebook_ok,
        'Docker (Optional)': docker_ok
    }

    for check_name, status in checks.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check_name:.<40} {status_str}")

    # Overall status
    critical_checks = [deps_ok, structure_ok]
    if all(critical_checks):
        print("\n✅ SETUP VERIFICATION PASSED!")
        if not model_ok:
            print("⚠ Note: Model not trained yet (run python train.py)")
    else:
        print("\n❌ SETUP VERIFICATION FAILED!")
        print("\nPlease install missing dependencies:")
        print("   pip install -r requirements.txt")

    # Print next steps
    print_next_steps(model_ok)


if __name__ == "__main__":
    main()
