This folder contains the adapters for each model. To use each model, setup according to their respective instructions:

## MINT
1. Clone MINT repo from [MINT](https://github.com/VarunUllanat/mint/tree/main) into a folder adjacent to pareto-prot.
2. Follow installation instructions, but use `environment_revised.yml` to setup the environment instead. Activate the environment.
3. Replace `extract.py`, `finetune_general.py`, `embeddings_mint.py`, and `SKEMPI_v2/prepare_data.py` with those provided in this repo. Edit all except `extract.py` to update with the output paths of your choosing.
4. Download the MINT base model checkpoint from [HuggingFace](https://huggingface.co/varunullanat2012/mint/tree/main).
5. Run `embeddings_mint.py` followed by `finetune_general.py` to create your own set of ddG prediction model weights.
```
cd mint/downstream/GeneralPPI && \
python embeddings_mint.py \
    --task SKEMPI \
    --model_name mint \
    --checkpoint_path [base model checkpoint path] \
    --devices 0 \
    --bs 1
```
```
python finetune_general.py \
    --task SKEMPI \
    --model mint \
    --use_mlp_for_cv
```
6. Move the checkpoint files to `pareto-prot/model/mint`. Ensure the mint base checkpoint is named `mint.ckpt` and SKEMPI_v2 finetuned (only need one) is `SKEMPI_v2.joblib`.