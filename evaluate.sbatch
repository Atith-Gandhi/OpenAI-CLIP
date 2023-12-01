#!/bin/bash
#SBATCH -JSlurmPythonExample1                   # Job name
#SBATCH --account=gts-svarma39               # charge account
#SBATCH --mem-per-gpu=700G                        # Memory per core
#SBATCH --gres=gpu:A100:1
#SBATCH -t18:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oResNetSpecializedBert_No_Sampling.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=agandhi98@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

# module load anaconda3/2022.05                       # Load module dependencies
# conda activate stylegan2
python main.py --sampling-function no_sampling
wait