#!/bin/bash
#SBATCH --job-name=ude_setup
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=outputs/setup_%j.log
export JULIA_CPU_TARGET="generic;znver1,clone_all;znver4,clone_all;icelake-server,clone_all"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'