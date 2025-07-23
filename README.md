# Arabic Fake News Detection
🚀 In the repo, we present the code for training a fake news detection model using the [AFND dataset](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd). 

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

As seen, the model has not converged yet which means it is possible to have a better preforming model if training was contunied. However, due to limited compute resources, we had to stop here
