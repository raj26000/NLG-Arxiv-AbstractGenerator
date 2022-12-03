# NLG-Arxiv-AbstractGenerator
Natural Language Generation Application to produce Abstracts of ArXiv papers given their titles by finetuning the GPT-2 LM on titles and abstracts of papers in **cs.CL** category. This can be extended to more diverse categories, with more compute time and resources. This repo contains the training script for the same and the data that was used for model training, along with an Inference module and an interactive **Streamlit** Web App to play around with. Follow the next steps to reproduce the training, inference and app locally else head over to the [**deployed app**](https://raj26000-nlg-arxiv-abstractgenerator-app-e183zy.streamlit.app/) to have fun. 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://raj26000-nlg-arxiv-abstractgenerator-app-e183zy.streamlit.app/)

## Setup for Local Training and Inference
- First clone the repository with `git clone https://github.com/raj26000/NLG-Arxiv-AbstractGenerator.git`
- Use the [`environment.yml`](environment.yml) file to create a conda virtual environment with all the required dependencies installed:
```
conda env create -f environment.yml
```
- Activate the environment with `conda activate env_name`. Remember, the first line of the yml file has the name of the environment that will be created. Feel free to change.

## Model Training 
- The Dataset for this task was sourced from [**Kaggle**](https://www.kaggle.com/datasets/Cornell-University/arxiv) and narrowed down to papers belonging to Computation And Language category.
- A **GPT-2** model was finetuned on this data using **PyTorch**, the script for which is provided - [`train.py`](train.py). The input sequence for training was modelled as follows:
```
input = <|startoftext|> title <|SEP|> abstract <|endoftext|>
```
- In order to perform training, first check the appropriate hyperparameters in ['config.json'](config.json) and alter them if you wish to. Then execute the training with the following terminal command: `python train.py`. The best model and tokenizer checkpoints will be saved automatically in the path provided in config.
- The trained model and tokenizer checkpoints I obtained were deployed on HuggingFace Hub and can be directly used to perform inference - in case one doesn't want to run the training again. Here is the [**HF model repository**](https://huggingface.co/raj26000/gpt2-arxiv-cs.CL) for the same. 

## Performing Inference
- At inference time, we provide the title as prompt to the model and expect it to generate the abstract. The prompt looks like this:
```
prompt = <|startoftext|> title <|SEP|>
```
- If you have trained the model again, consider deploying the checkpoints on HuggingFace hub and provide its path in the config (**hf_finetuned_checkpoint**). Else, my default trained checkpoint will be used for inference. Follow [**this**](https://huggingface.co/docs/transformers/model_sharing#upload-with-the-web-interface) to share your models to the Hub simply using the web interface.
- The [`app.py`](app.py) file contains the Streamlit app. To run the app locally with your checkpoint, use this command in terminal. The app should start running on localhost. :rocket:
```
python -m streamlit run app.py
```
- The app has also been deployed on **Streamlit Cloud** with my model checkpoint if one wants to use that. You don't have to reproduce it locally then. Check it out here:
[**Deployed App**](https://raj26000-nlg-arxiv-abstractgenerator-app-e183zy.streamlit.app/)

## Using the App
- Once you have the app running, local or deployed version, its time to test it out.
- Enter the title of the cs.CL category paper. (Or any other based on what you have finetuned the model for).
- Select the decoding strategy for text generation - currently the project supports **Greedy Search**, **Beam Search**, **Stochastic Sampling**, **Contrastive Search**. 
- With each, you may be prompted to further select some associated hyper-parameters. You can choose to stick with what I've provided or enter your own. 
- Click the button to generate the Abstract. It **might** take a few seconds, so please be patient.