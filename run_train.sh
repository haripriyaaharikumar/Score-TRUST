#!/bin/bash
PYTHON_EXE="/../bin/python"
PROJECT_PATH="/../Ur_project_name"

export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"

$PYTHON_EXE ./run_train.py