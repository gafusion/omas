# This script runs the OMAS tests without usinig the `-m unittest` option
# which is useful to debug when `-m unittest` gives cryptic error messages
set -e
for testfile in omas/tests/test_*.py ; do
    python3 $testfile
done