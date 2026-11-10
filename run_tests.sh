#!/usr/bin/env bash
# Test entry point (library_notes/09_improvement_plan.md WS1).
#
#   ./run_tests.sh             fast gate: unit tests only (~10 s)
#   ./run_tests.sh validation  statistical validation on the Gaussian toy (~2 min)
#   ./run_tests.sh mpi         mpiexec-driven integration/deadlock tests (~1.5 min, needs mpiexec)
#   ./run_tests.sh all         everything (~5 min)
#
# PYTHON overrides the interpreter (default: /dolfinx-env/bin/python3 if present, else python3).
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x /dolfinx-env/bin/python3 ]]; then PYTHON=/dolfinx-env/bin/python3; else PYTHON=python3; fi
fi

case "${1:-unit}" in
    unit)        exec "$PYTHON" -m pytest tests -q -m unit ;;
    validation)  exec "$PYTHON" -m pytest tests -q -m validation ;;
    mpi)         exec "$PYTHON" -m pytest tests/mpi -q -m mpi -rxX ;;
    all)         "$PYTHON" -m pytest tests -q -m "not mpi" && exec "$PYTHON" -m pytest tests/mpi -q -m mpi -rxX ;;
    *)           echo "usage: $0 [unit|validation|mpi|all]" >&2; exit 2 ;;
esac
