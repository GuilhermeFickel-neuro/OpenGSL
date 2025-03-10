conda create -n OpenGSL python==3.10
conda activate OpenGSL
conda install pytorch==1.13.1 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 'dgl<2'
python -m pip install -e . 'torch<2'