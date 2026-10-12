#!/bin/bash
#SBATCH --job-name=run_multistart
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=charlotte.holt@lshtm.ac.uk
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/multistart_%j.log
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CPU_TARGET="generic;znver1,clone_all;znver4,clone_all;icelake-server,clone_all"
julia --project=. --compiled-modules=existing scripts/multistart.jl
