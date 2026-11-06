# CS6886 -- Optimising attention on a CPU
#
# Path to your C compiler. Use the LATEST clang, for example:
#   Linux:  /usr/bin/clang-23        macOS (Homebrew):  /opt/homebrew/opt/llvm/bin/clang
CC = clang

CFLAGS = -O3 -march=native
OMP    = -fopenmp
# Install the selected compiler's OpenMP development package system-wide.
# Requires make, a compiler with AVX2/FMA support, libm, and Python 3 (stdlib only).
# No experiment scripts, project-local libraries, or prebuilt binaries are needed.
PYTHON = python3
INPUTS = $(wildcard input/*.bin)

# make <variant>  : compile src/attn_<variant>.c and report its time on every input
# make all        : the same for every variant
# make check      : check every variant's output against naive's output
#                   run make all first; missing/invalid outputs fail the check
# make clean      : delete build/ and output/

all: naive loops tiled fused best best_v2 omp

naive loops tiled fused best best_v2:
	@mkdir -p build output; rm -f build/$@ output/$@_*
	$(CC) $(CFLAGS) -o build/$@ src/attn_$@.c -lm
	@set -e; for f in $(INPUTS); do ./build/$@ "$$f" "output/$@_$${f##*/}"; done

omp:
	@mkdir -p build output; rm -f build/$@ output/$@_*
	$(CC) $(CFLAGS) $(OMP) -o build/$@ src/attn_$@.c -lm
	@set -e; for f in $(INPUTS); do ./build/$@ "$$f" "output/$@_$${f##*/}"; done

# Passes if max|O - O_naive| / max|O_naive| <= 1e-4. Outputs never match exactly:
# different loop orders round differently.
check:
	@$(PYTHON) -c "$$CHECK"

clean:
	rm -rf build output

.PHONY: all naive loops tiled fused best best_v2 omp check clean

define CHECK
import glob, math, os, struct, sys
from array import array
def load(p):
    with open(p, "rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8: raise ValueError("invalid header")
        n, d = struct.unpack("<ii", hdr)
        if n <= 0 or d <= 0: raise ValueError("invalid dimensions")
        a = array("f"); a.fromfile(f, n * d)
        if sys.byteorder != "little": a.byteswap()
        if f.read(1): raise ValueError("unexpected trailing data")
    if not all(math.isfinite(x) for x in a): raise ValueError("non-finite output")
    return (n, d), a
variants = ["loops", "tiled", "fused", "best", "best_v2", "omp"]
inputs = sorted(glob.glob("input/*.bin"))
if not inputs: raise SystemExit("no inputs: expected input/*.bin")
failures = 0
for input_path in inputs:
    inp = os.path.basename(input_path)
    try:
        with open(input_path, "rb") as f: expected = struct.unpack("<ii", f.read(8))
        shape, ref = load("output/naive_" + inp)
        if shape != expected: raise ValueError("dimensions differ from input")
    except (OSError, ValueError, EOFError, struct.error) as e:
        print("%-20s %-12s FAIL (%s; run make all)" % (inp, "naive", e))
        failures += 1
        continue
    scale = max(abs(x) for x in ref)
    for v in variants:
        path = "output/" + v + "_" + inp
        try:
            out_shape, out = load(path)
            if out_shape != shape: raise ValueError("dimensions differ from reference")
            absolute = max(abs(a - b) for a, b in zip(out, ref))
            err = absolute / scale if scale else (0.0 if absolute == 0.0 else float("inf"))
        except (OSError, ValueError, EOFError, struct.error) as e:
            print("%-20s %-12s FAIL (%s; run make all)" % (inp, v, e))
            failures += 1
            continue
        print("%-20s %-12s %s  (error %.1e)" % (inp, v, "PASS" if err <= 1e-4 else "FAIL", err))
        failures += err > 1e-4
raise SystemExit(1 if failures else 0)
endef
export CHECK
