#!/usr/bin/env python3
"""
Pre-commit configuration for medication adherence monitoring project.

This script sets up pre-commit hooks for code quality and formatting.
"""

import subprocess
import sys
from pathlib import Path


def install_pre_commit():
    """Install pre-commit hooks."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        print("✓ Pre-commit installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install pre-commit")
        return False
    return True


def setup_hooks():
    """Set up pre-commit hooks."""
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        print("✓ Pre-commit hooks installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install pre-commit hooks")
        return False
    return True


def create_pre_commit_config():
    """Create pre-commit configuration file."""
    config_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.280
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
"""
    
    config_path = Path(".pre-commit-config.yaml")
    config_path.write_text(config_content)
    print("✓ Pre-commit configuration created")


def main():
    """Main setup function."""
    print("Setting up pre-commit hooks for medication adherence monitoring...")
    
    # Install pre-commit
    if not install_pre_commit():
        return
    
    # Create configuration
    create_pre_commit_config()
    
    # Setup hooks
    if not setup_hooks():
        return
    
    print("\n✓ Pre-commit setup completed successfully!")
    print("\nTo run pre-commit hooks manually:")
    print("  pre-commit run --all-files")
    print("\nTo update hooks:")
    print("  pre-commit autoupdate")


if __name__ == "__main__":
    main()
