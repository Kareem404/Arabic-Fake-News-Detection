# Arabic News Credibility Checker
🚀 In the repo, we present the code for training a fake news detection model using the [AFND dataset](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd). The model is given an Arabic article in text and returns whether an article came from a credible source or not. 

## 1. 🤖 Model Description
We fine-tuned the AraBERTv02 pretrained transformer using Low-Rank Adaptation (LoRA). The reason LoRA was used is because we did not have enough compute to run fully fine-tune the model.

### 1.1. AraBERT
AraBERT is an Arabic pretrained language model based on the Google's BERT architechture. We specifically use `bert-base-arabertv02` which is the smallest AraBERT model. The model size is 543MB/136M and it was trained on 8.6B words. 

### 1.2. LoRA
LoRA is a parameter-efficient fine-tuning method for large pre-trained models (like BERT, GPT). Instead of updating all the model weights, LoRA inserts small trainable low-rank matrices into certain layers (usually the attention or feedforward layers).

## 2. </> Running our Code
To run our code, start by cloning this repo: 
`git clone https://github.com/Kareem404/Arabic-Fake-News-Detection.git`
### 2.1 Install Dataset
You first need to install the dataset which is found in this [link](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd).

After that, place the dataset in the `./AFND` directory. The directory structure should look like this:
```
AFND
│   README.md
│   sources.json  
└───Dataset
│   └───source1
│   └───source2
│   │   ...
│   └───sourceN

```

### 2.2 Install Dependencies
First, run this command ```pip install -r requirements.txt```. This installs all the required dependencies excluding PyTorch because we are going to use a GPU-enabled version. To install Pytorch, run this command: 

```pip install torch==2.5.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html```

Please note that this will not work if you do not have the correct CUDA version. 

### 2.3 Preproccess/Tokenize Dataset
To be able to run our model, we need to first tokenize the dataset. To do so, run ````python preproccess.py````. The ```preproccess.py``` script creates a new directory `./Split` which containes a tokenized version of the dataset saved as a bunch of `.pt` tensors. The directory contains training and testing splits saved as tokenized tensors. This is done individually for the "text" feature and the "title" feature. Note in our model, we only use the "text" feature.

⚠️ This script **SHOULD** only be ran once.

### 2.4 Fine-tune AraBERTv02
To train the model, run ```python train.py```. The script by default will run the model for 8 epochs with a batch size of 16. To change this, modify `TrainingArguments` in the `train.py` file:

```
TrainingArguments(output_dir="./Split_test/results", 
                    eval_strategy="epoch",
                    num_train_epochs=8, # modify this
                    per_device_train_batch_size=16  # modify this
)
```

The script also checkpoints every specific amount of steps of running the model. This will be shown in `./AFND/results`. You can later resume training from a specific checkpoint or use the checkpoint to evaluate the model. 

# 📈 3. Evaluation
To the best of our knowledge, the results obtained by our model are the best using the dataset! The confusion matrix is shown by this figure:

![alt text](figs/confusion%20matrix.png)

The classification report is shown by this table: 
<center>

|             | Precision   | Recall        |f1-score |
| :---        |    :----:   |   :----:      |   :----:   |
| Not credible      | 0.92       | 0.91   |0.92 |
| credible   | 0.93        | 0.94      | 0.93|

</center>

The loss curve is shown curve is shown here:

![alt text](figs/loss%20curve.png)

As seen, the model has not converged yet which means it is possible to have a better preforming model if training was contunied. However, due to limited compute resources, we had to stop here.

# 👨‍💻 4. API
A simple API and client application was made to better showcase the model. The API was created using [FastAPI](https://fastapi.tiangolo.com/). The API was then deployed using Docker (🐋) and Microsoft Azure ACI's service. The container was made public in dockerhub and is available in this [link](https://hub.docker.com/repository/docker/karim129/afnd/general). Feel free to pull it and run the API locally using `docker pull karim129/afnd:latest`. After pulling the repo, you could run it locally using `docker run --name afnd -p 8001:80 karim129/afnd:latest`. The API will be available at `http://localhost:8001`. 

The code for the API is available in this directory `./API`. In the `./API` directory we should have another folder that is a model checkpoint `./ckpt`. You can extract this folder after the `./train.py` file was ran. The folder structure of `./ckpt` should be as follows:
```
./ckpt
│   adapter_config.json
│   adapter_model.safetensors
│   optimizer.pt
│   rng_state.pth
│   scheduler.pt
│   trainer_state.JSON
│   training_args.bin
```
To access the API (if it is still running ;/), use the following link: `http://4.239.91.120`. You can access the documentation in this route `./docs`.

The client application is available at this [URL](https://kmvx1.pythonanywhere.com/).

# 📝 5. Citations
@inproceedings{antoun2020arabert,
  title={AraBERT: Transformer-based Model for Arabic Language Understanding},
  author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
  booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
  pages={9}
}
