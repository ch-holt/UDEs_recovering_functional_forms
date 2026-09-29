#!/bin/bash
#SBATCH --job-name=run_single_trajectories
#SBATCH --array=1-51
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=charlotte.holt@lshtm.ac.uk
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=outputs/out_%A_%a.log
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CPU_TARGET="generic;znver1,clone_all;znver4,clone_all;icelake-server,clone_all"
julia --project=. --compiled-modules=existing -t $SLURM_CPUS_PER_TASK scripts/ude_model_single_traj.jl "$@"
