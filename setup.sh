#!/bin/bash

# Ensure the script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced, not executed. Please run it with: 'source ./setup.sh'."
    return 1
fi

echo "Preparing setup for $OSTYPE..."

# Detect and handle OS-specific setup
case "$OSTYPE" in
    linux-gnu*)
        echo "Detected Linux Operating System..."

        # Check if Python is installed
        if ! command -v python3 &>/dev/null; then
            echo "Python3 is not installed! Please install it: 'sudo apt install -y python3'"
            return 1
        fi

        # Check if 'venv' is available
        if ! python3 -m venv --help &> /dev/null; then
            echo "venv is not installed! Please install it: 'sudo apt install -y python3-venv'"
            return 1
        fi
        
        # Install 'pip' if not available
        if ! python3 -m pip --version &> /dev/null; then
            echo "Installing pip..."
            python3 -m ensurepip || { echo "Failed to install pip!"; return 1; }
        fi

        echo "Setting up the virtual environment..."
        python3 -m venv env || { echo "Failed to create virtual environment!"; return 1; }
        source env/bin/activate || { echo "Failed to activate virtual environment!"; return 1; }

        if [[ -f "requirements.txt" ]]; then
            echo "Installing dependencies..."
                #python3 -m pip install --upgrade pip || { echo "Failed to upgrade pip!"; return 1; }
                pip install -r requirements.txt || { echo "Failed to install dependencies!"; return 1; }
        else
            echo "requirements.txt not found! You must provide the necessary requirements.txt file."
            return 1
        fi

        echo "Setup complete. Virtual environment is activated. To deactivate, run: 'deactivate'."
        echo "To reactivate the environment later, run: 'source env/bin/activate'."
        ;;

    darwin*)
        echo "Detected macOS Operating System..."

        # Ensure Homebrew is installed
        if ! command -v brew &>/dev/null; then
            echo "Homebrew not found. Please install it first: https://brew.sh/"
            return 1
        fi

        # Check if Python is installed
        if ! command -v python3 &>/dev/null; then
            echo "Python3 is not installed! Please install it: 'brew install python3'"
            return 1
        fi

        # Check if venv is available
        if ! python3 -m venv --help &> /dev/null; then
            echo "venv is not installed. Please reinstall Python3 to include venv support."
            return 1
        fi
        
        # Install 'pip' if not available
        if ! python3 -m pip --version &> /dev/null; then
            echo "Installing pip..."
            python3 -m ensurepip || { echo "Failed to install pip!"; return 1; }
        fi

        echo "Setting up the virtual environment..."
        python3 -m venv env || { echo "Failed to create virtual environment!"; return 1; }
        source env/bin/activate || { echo "Failed to activate virtual environment!"; return 1; }

        if [[ -f "requirements.txt" ]]; then
            echo "Installing dependencies..."
            #python3 -m pip install --upgrade pip || { echo "Failed to upgrade pip!"; return 1; }
            python3 -m pip install -r requirements.txt || { echo "Failed to install dependencies!"; return 1; }
        else
            echo "requirements.txt not found! You must provide the necessary requirements.txt file."
            return 1
        fi

        echo "Setup complete. Virtual environment is activated. To deactivate, run: 'deactivate'."
        echo "To reactivate the environment later, run: 'source env/bin/activate'."
        ;;

    msys* | win32*)
        echo "Detected Windows Operating System..."

        # Check if Python is installed
        if ! command -v python &>/dev/null; then
            echo "Python is not installed. Please install it: https://www.python.org/downloads/"
            return 1
        fi

        # Check if 'venv' is available
        if ! python -m venv --help &> /dev/null; then
            echo "venv is not installed. Please reinstall Python for Windows with venv support!"
            return 1
        fi
	
        # Ensure 'pip' is available
        if ! python -m pip --version &> /dev/null; then
            echo "Installing pip..."
            python -m ensurepip || { echo "Failed to install pip!"; return 1; }
        fi

        echo "Setting up the virtual environment..."
        python -m venv env || { echo "Failed to create virtual environment!"; return 1; }
        source env/Scripts/activate || { echo "Failed to activate virtual environment!"; return 1; }
        
        if [[ -f "requirements.txt" ]]; then
            echo "Installing dependencies..."
            #python -m pip install --upgrade pip || { echo "Failed to upgrade pip!"; return 1; }
            python -m pip install -r requirements.txt || { echo "Failed to install dependencies!"; return 1; }
        else
            echo "requirements.txt not found! You must provide the necessary requirements.txt file."
            return 1
        fi

        echo "Setup complete. Virtual environment is activated. To deactivate, run: 'deactivate'."
        echo "To reactivate the environment later, run: 'source env/Scripts/activate'."
        ;;

    *)
        echo "Unsupported OS: $OSTYPE. Please manually install Python3 and set up a virtual environment."
        return 1
        ;;
esac
