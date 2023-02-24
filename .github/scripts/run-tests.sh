#!/bin/sh
set -e

echo "Running tests..."
python -m pytest
echo "Tests passed!"