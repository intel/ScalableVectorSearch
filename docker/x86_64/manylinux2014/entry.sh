#!/bin/sh

# setup MKL environment variables
source /opt/intel/oneapi/setvars.sh
# Run the requested command
exec "$@"
