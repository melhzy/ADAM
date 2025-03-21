#!/bin/bash

# Prompt for EXCLUDE_NUM
read -p "Enter number to exclude (e.g., 01, 02, 03, etc.): " EXCLUDE_NUM

# Prompt for BASE_NAME
read -p "Enter base name (e.g., machine_learning, xgboost, random_forest, logistic_regression, etc.): " BASE_NAME

# Prompt for SOURCE_NUM (the number to copy from, e.g., 01)
read -p "Enter source experiment number to copy from (e.g., 01, 02, 03, etc.): " SOURCE_NUM

# Loop and copy files, excluding the specified number
for i in {01..10}; do
    if [ "$i" != "$EXCLUDE_NUM" ]; then
        SOURCE_FILE="${BASE_NAME}_experiment${SOURCE_NUM}.ipynb"
        DEST_FILE="${BASE_NAME}_experiment$i.ipynb"

        # Check if source file exists
        if [ -f "$SOURCE_FILE" ]; then
            cp "$SOURCE_FILE" "$DEST_FILE"
            echo "Copied $SOURCE_FILE to $DEST_FILE"
        else
            echo "Source file $SOURCE_FILE does not exist. Aborting."
            exit 1
        fi
    fi
done

echo "Operation completed."