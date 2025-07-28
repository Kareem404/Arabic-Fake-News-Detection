from transformers import AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer
from peft import PeftModel

def predict_credibility(news: list[str]) -> list[list[dict]]:
    """Function to predict the credibility of a news source"""

    # load model
    model_name="aubmindlab/bert-base-arabertv02" # AraBERT model
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    trained_model = PeftModel.from_pretrained(base_model, './ckpt') # get the model
    # preprocess news 
    news_preprocess = list(map(lambda x: arabert_prep.preprocess(x), news))
    # init pipeline
    pipe = TextClassificationPipeline(model=trained_model, tokenizer=arabert_tokenizer, return_all_scores=True)
    # classifiy
    outputs = pipe(news_preprocess)
    credibility_score = outputs[0][1]['score']

    return credibility_score


# LABEL_1: credible
# LABEL_0: not credible
# sample output: [[{'label': 'LABEL_0', 'score': 0.7900840640068054}, {'label': 'LABEL_1', 'score': 0.2099159061908722}]]