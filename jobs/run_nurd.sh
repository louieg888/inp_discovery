# Base experiment for INPs on sinusoids
# ======================================
# Train NP without knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1
python models/train.py

python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1
python models/train.py

python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True
python models/train.py

# With critic stuff, normal

python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True
python models/train.py

### multi task experiments
#
#

# no reweighting, no bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 
python models/train.py

# no reweighting, bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 3 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type b --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 
python models/train.py

# reweighting, no bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True
python models/train.py

# reweighting, bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 3 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type b --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True
python models/train.py