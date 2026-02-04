#!/bin/bash
#SBATCH --job-name=kappa_array
#SBATCH --account=def-jbpoline
#SBATCH --output=/home/jpfarr/projects/def-jbpoline/jpfarr/enigma-pd_questionnaire-harmonization/logs/kappa_%a.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-6

# SLURM Array Job for Multi-Kappa Analysis
# Each array task processes one construct in parallel
# ENIGMA-PD Harmonization Study

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate Python environment
module load python/3.11.5
source ../envs/enigma-pd/bin/activate

echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Virtualenv: $VIRTUAL_ENV"
echo ""

# Set threading environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using $SLURM_CPUS_PER_TASK cores for parallel processing"
echo ""

echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la *.py 2>/dev/null | head -5
echo ""

# Create directories
mkdir -p logs
mkdir -p simulation_results_multikappa

# Map task ID to construct
case $SLURM_ARRAY_TASK_ID in
    1) CONSTRUCT="Depression" ;;
    2) CONSTRUCT="Anxiety" ;;
    3) CONSTRUCT="Psychosis" ;;
    4) CONSTRUCT="Apathy" ;;
    5) CONSTRUCT="ICD" ;;
    6) CONSTRUCT="Sleep" ;;
    *) echo "Invalid task ID"; exit 1 ;;
esac

echo "Processing construct: $CONSTRUCT"
echo ""

# Run analysis for specific construct
python multi-kappa_corrected.py --construct $CONSTRUCT

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Task $SLURM_ARRAY_TASK_ID ($CONSTRUCT) completed successfully!"
else
    echo "✗ Task $SLURM_ARRAY_TASK_ID ($CONSTRUCT) failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "Memory used: $(sacct -j $SLURM_JOB_ID --format=MaxRSS --noheader | head -1)"
echo "=========================================="

exit $EXIT_CODE