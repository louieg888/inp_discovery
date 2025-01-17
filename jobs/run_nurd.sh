# Base experiment for INPs on sinusoids
# ======================================
# Train NP without knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --task single
python models/train_reweighting.py

python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --task single
python models/train_reweighting.py

python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True --task single --use-optimal-rep True
python models/train_reweighting.py

# With critic stuff, normal

python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 10 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True --task single
python models/train_reweighting.py

### multi task experiments
#
#

# no reweighting, no bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 100 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --task multi
python models/train_reweighting.py

# no reweighting, bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 3 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type b --noise 0 --min-num-context 0 --max-num-context 100 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --task multi
python models/train_reweighting.py

# reweighting, no bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 100 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True --task multi
python models/train_reweighting.py

# reweighting, bc knowledge
python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 3 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type b --noise 0 --min-num-context 0 --max-num-context 100 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --xy-encoder-num-hidden 1 --xy-encoder-hidden-dim 16 --latent-encoder-num-hidden 1 --decoder-hidden-dim 16 --decoder-num-hidden 1 --x-encoder-num-hidden 1 --reweight True --task multi
python models/train_reweighting.py





# ### testing to see if we are fucking ourselves by underparametrizing
# python config.py  --project-name nurd_validation --dataset nurd  --input-dim 1 --output-dim 1 --hidden-dim 3 --run-name-prefix nurd_validation --use-knowledge True --knowledge-type b --noise 0 --min-num-context 0 --max-num-context 100 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 --reweight True

# ### ok now i'm properly confused, let's try this to see if we can get a gap between train/test 
# python config.py  --project-name nurd_validation --dataset nurd  --input-dim 2 --output-dim 1 --hidden-dim 1 --run-name-prefix nurd_validation --use-knowledge False --knowledge-type z --noise 0 --min-num-context 0 --max-num-context 100 --num-targets 1000 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --seed 1 