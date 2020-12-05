# python
# author: scc_hy
# Func: Text Classification with SpaCy

# Bag of Words
# ----------------------------------------
import spacy
# Create an empty model
nlp = spacy.blank('en')
# Create the TextCategorizer with exclusive classes and 'bow' architecture
textcat = nlp.create_pipe(
  'textcat',
  config = {
    'exclusive_classes':True,
    'architecture':'bow'
  }
)

nlp.add_pip(textcat)

# add labels to text classifier
textcat.add_label('ham')
textcat.add_label('spam')

# Training a Text Categorizer Model
# 做成指定数据结构
# ------------------------------------------------
train_texts = spam['text'].values
train_labels = [
  {
  'cats' : {
      'ham' : label == 'ham',
      'spam' : label == 'spam'
    }
  } for label in spam['label']
]
train_data = list(zip(train_texts, train_labels))

# train the model
# ---------------------------------------------------
from spacy.util import minibatch
import random

spacy.util.fix_random_seed(1)
opt_ = nlp.begin_training() # 模型迭代
batches = minibatch(train_data, size=8)
  
rd_ = random.Random()
rd_.random_seed(12)

losses ={}
for epoch in range(10):
  rd_.shuffle(train_data)
  for bath in batches:
    texts, labels = batch
    nlp.update(texts, labels, sgd=opt_, losses = losses)
  print(losses)

# Making Predictions
# ---------------------------------------------------
texts = ['texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]
docs = [
  nlp.tokenizer(text) for text in texts
]

textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])

# 综合应用的简单函数
# ------------------------------------------------------

from spacy.util import minibatch
import random

def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    
    # Shuffle data
    train_data = data.sample(frac=1, random_state=7)
    
    texts = train_data.text.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}
              for y in train_data.sentiment.values]
    split = int(len(train_data) * split)
    
    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]
    
    return texts[:split], train_labels, texts[split:], val_labels

train_texts, train_labels, val_texts, val_labels = load_data('../input/nlp-course/yelp_ratings.csv')



def train(model, train_data, optimizer):
    losses = {}
    rd_ = random.Random()
    rd_.random_seed(12)
    rd_.shuffle(train_data)
    
    batches = minibatch(train_data, size=8)
    for batch in batches:
        texts, labels = zip(*batch)
        # Update model with texts and labels
        model.update(texts, labels, sgd=optimizer, losses=losses)
        
    return losses


def predict(nlp, texts): 
    # Use the model's tokenizer to tokenize each input text
    docs = [nlp.tokenizer(t) for t in texts]
    
    # Use textcat to get the scores for each doc
    textcat = nlp.get_pipe('textcat')
    scores, _ = textcat.predict(docs)
    
    # From the scores, find the class with the highest score/probability
    label_idx = scores.argmax(axis=1)
    predicted_class = label_idx #[textcat.labels[label] for label in label_idx]
    
    return predicted_class


def evaluate(model, texts, labels):
    """ Returns the accuracy of a TextCategorizer model. 
    
        Arguments
        ---------
        model: ScaPy model with a TextCategorizer
        texts: Text samples, from load_data function
        labels: True labels, from load_data function
    
    """
    # Get predictions from textcat model (using your predict method)
    predicted_class = predict(model, texts)
    
    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]
    
    # A boolean or int array indicating correct predictions
    correct_predictions = predicted_class == true_class
    
    # The accuracy, number of correct predictions divided by all predictions
    accuracy = correct_predictions.mean()
    
    return accuracy


  # This may take a while to run!
n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")



