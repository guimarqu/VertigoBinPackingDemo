#!/bin/sh
# Download the Falkenauer instances of BPPLIB into instances/.
# BPPLIB is CC BY-NC-ND 4.0, so the files are fetched, not redistributed here.
set -eu

BPPLIB_COMMIT=fc3aa784a3b14282fc591d710f056abc1b1451c3
URL="https://raw.githubusercontent.com/mdelorme2/BPPLIB/$BPPLIB_COMMIT/Instances/Benchmarks/1_Falkenauer.rar"

cd "$(dirname "$0")/.."
mkdir -p instances
curl -fsSL -o instances/falkenauer.rar "$URL"
# The archive holds Falkenauer/Falkenauer {U,T}/<name>.txt; keep the files only.
bsdtar -xf instances/falkenauer.rar -C instances --strip-components 2
rm instances/falkenauer.rar
ls instances | wc -l | xargs printf '%s instances in instances/\n'
