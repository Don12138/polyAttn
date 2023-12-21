#!/bin/bash

#SBATCH -p intel32
#SBATCH -J jupyter
#SBATCH --nodes=1
#SBATCH -t 20-00:00
#SBATCH --mem=20G

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/pub/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/pub/opt/intel/oneapi/intelpython/latest/etc/profile.d/conda.sh" ]; then
        . "/home/pub/opt/intel/oneapi/intelpython/latest/etc/profile.d/conda.sh"
    else
        export PATH="/home/pub/opt/intel/oneapi/intelpython/latest/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate py_38_torch_113_pyg

jupyter notebook --no-browser  --port=12138 --ip=0.0.0.0 --NotebookApp.password='argon2:$argon2id$v=19$m=10240,t=10,p=8$IMiigJ3sa/bIWixmbFh6ww$9l8Ju9YQ7HWoa4KJXBadLA'

