#!/bin/bash
#SBATCH --job-name=run_symbolic_regression
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=outputs/out_%j.log
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CPU_TARGET="generic;znver1,clone_all;znver4,clone_all;icelake-server,clone_all"
julia --project=. --compiled-modules=existing -t $SLURM_CPUS_PER_TASK scripts/ude_symbolic_regression.jl "$@"
