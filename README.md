# LLM-based chatbot

Instructions to run locally:

- Key point: Download preferred LLM through HuggingFace, this app uses OpenHermés
- Make sure you have Git LFS installed for the command to work.

  ```
  git clone https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-7b-v2.5
  mv openhermes-7b-v2.5 models/
  ```
Edit the llm_app.py file, and replace the 'DESIRED_QUANTISED_MODEL_HERE' placeholder with the correct name of the model.
You can also change the hyperparameters when creating an instance of the model.

```
pip install -r requirements.txt
```
```
streamlit run llm_app.py --server.address 0.0.0.0 --server.port 8000
```
