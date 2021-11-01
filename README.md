# VeeSmart - AI
<h1 align="center">Sentiment Analysis on Businesses Reviews</h1>

<h4 align="center">Abstract</h4>
<p>Our goal during this internship is to create a deep learning model that can predict the 
sentiment (positive/negative) of a review. The steps we took to achieve that are:<br>
&nbsp&nbsp&nbsp&nbsp1- Web Scraping & Data Cleaning<br>
&nbsp&nbsp&nbsp&nbsp2- Data Visualization<br>
&nbsp&nbsp&nbsp&nbsp3- Data Preparation & LSTM model<br>
&nbsp&nbsp&nbsp&nbsp4- BERT Transformers model<br>
&nbsp&nbsp&nbsp&nbsp5- LSTM model VS BERT Transformers model<br>
&nbsp&nbsp&nbsp&nbsp6- Named Entity Recognition & Sentiment Analysis</p>

[<h2 align="center">Web Scraping & Data Cleaning</h2>](https://github.com/HadiAbouDaya/VeeSmart/tree/main/WebScraping%20%2B%20DataPreprocessing)
<h3>Web scraping</h3>
<p>I used the <b>selenium</b> and <b>BeautifulSoup</b> libraries to scrape Zomato’s website. I did it in a 3-step
process which are: getting the html source code of the pages I am willing to scrape, extracting the data 
needed and finally scraping Zomato again to get the profile picture for each user.
</p>
<h3>Data Cleaning
</h3>
<p>reated a new dataframe containing the scraped data without invalid or duplicated rows, or non-English sentences and showed a few samples of the data along with a few helpful graphs to check for data 
imbalance.</p>

[<h2 align="center">Data Visualization</h2>](https://github.com/HadiAbouDaya/VeeSmart/blob/main/LSTM%20-%20Binary.ipynb)
<h3>Data related
</h3>
<p>The data was comprised of 100 000 reviews with an average rating of 3.78 stars, after feature engineering 
and joining the businesses table to the reviews table I had 16 columns (<br>
0 user_id (Id of the User)<br>
1 business_id (Id of the business)<br>
2 stars (Star rating)<br>
3 useful <br>
4 funny <br>
5 cool <br>
6 text (The review itself)<br>
7 year <br>
8 month <br>
9 day <br>
10 hour <br>
11 text_length_char (Num of characters in review)<br>
12 categories (Category of the business)<br>
13 clean_text (Review after cleaning)<br>
14 clean_text_length (Num of words in clean_text)<br>
15 labels (0 if stars <3 else 1)<br>
).<br>
Also, I displayed correlation between correlation between columns, correlation between stars and the rest 
of the columns, reviews/stars, reviews/year, stars/length of review. Also a few interesting charts show that 
the higher the rating, the less users wrote on average.</p>
<h3>NLP related</h3>
<p>“the” is the most used stopword followed by “and”, then “a” … Conversely, “-“ is the most used 
punctuation, followed by “&”, then “.” then ”,”…
The number of unique words used in this dataset is 88062 words.</p>

[<h2 align="center">Data Preparation & LSTM model</h2>](https://github.com/HadiAbouDaya/VeeSmart/blob/main/LSTM%20-%20Binary.ipynb)
<h3>Data Preparation</h3>
<p>Preparing the text in a specific manner is key to get the most out of your LSTM model for NLP. 
<b>Reducing the amount of data analyzed</b> is essential, that is why we start by lower casing all words and
removing all html tags, URLs, emojis, stopwords and punctuation from our reviews because their effect 
on the sentiment of the whole sentence in minimal. <b>Stemming</b> is the next step in the process, it reduces
different forms of the same word into their root. Ex: eating, eats, eaten is eat. Maybe the most important 
step of them all is word embedding, you first need to choose the maximum sequence length which may be 
a tradeoff between accuracy and training resources, normally it is chosen as the average length of the 
clean review but I chose it using the formula below
<b>max_length = int((maximum_length + mean_length)//2)</b> for better accuracy and I 
didn’t mind waiting a bit longer when training the LSTM model. The next step is <b>word tokenization</b>, 
which assigns a number to each unique word in our vocabulary and also an out of vocabulary token 
“OOV” for words that where never seen before by the tokenizer when it was being fitted to the training
data. Now we finally get to the embedding part, this is where we encode each sentence into numbers by 
matching each word into its correspondent number set by the tokenizer and we add zeros at the end of 
each sentence that is shorter that the <b>max_length</b> previously calculated and trim sentences that are 
longer than that. After applying all previously stated steps and transformations we split you data into 
training and validation while previously having a test set that we have never touched.</p>
<h3>LSTM model</h3>
<p>LSTM stands for Long Short-Term Memory. LSTM is a Gated Recurrent Neural Network, and a 
bidirectional LSTM is just an extension of that model. An important factor is that such networks 
can store information that can be used to be processed by future cells. We can think of LSTM as 
an RNN with a memory pool with two main vectors:<br>
&nbsp&nbsp&nbsp&nbsp(1) Short-term state: keeps the output at the current time step.<br>
&nbsp&nbsp&nbsp&nbsp(2) Long-term state: retains, reads, and rejects long-term objects while passing through a network.<br>
These characteristics make the bidirectional LSTMs great for sequence analysis. After deciding which 
model architecture I will be using, I tuned its hyper parameters through grid search cross validation. 
When you are working with unbalanced data, which is the case here, it is helpful to calculated relative 
class weights to reduce the bias of your model. I placed a callback for my training with a training 
accuracy threshold of 90% which was spot on as it stopped training right when the model started 
overfitting. The model performed well on the test set with accuracy and F1 score > 90%, further testing 
results will be discussed later.</p>

[<h2 align="center">BERT Transformers model
</h2>](https://github.com/HadiAbouDaya/VeeSmart/blob/main/BERT%20-%20Binary.ipynb)
<h3>Data Preparation
</h3>
<p>Actually, very little data preparation is required to make the reviews ready for BERT Transformers. One 
essential thing is to trim all of your sentences to the maximum length of the BERT Tokenizer you are 
using (mine was 512 words/sentences) after that, you are all set. You just need to import the BERT 
Tokenizer, set max_length=262, truncation=True, padding="max_length". Then you create a tensorflow 
dataset from the tokenized data tf.data.Dataset.from_tensor_slices((dict(train_encodings), 
training_labels)) and the tf dataset is ready to fine-tune your pretrained BERT Transformers model for 
sequence classification.
</p>
<h3>BERT Transformers model</h3>
<p>I chose the TFDistilBertForSequenceClassification because the distil BERT has less trainable parameters 
than other BERT Transformers models but with really good results. The same class weights where 
applied and the model kept on learning and did not overfit, I stopped the training with a callback that has 
a 90% accuracy threshold because it was enough accuracy for my use case, I did not come close to 
overfitting. The results on the test set were great, I got 93% accuracy and 95% F1 score, more analysis of 
the test result will come later.</p>

[<h2 align="center">LSTM model VS BERT Transformers model</h2>](https://github.com/HadiAbouDaya/VeeSmart/blob/main/LSTM_VS_BERT.ipynb)
<h3>LSTM model</h3>
<p>Testing results of this model are the following:<br>
Accuracy: 0.912168<br>
Precision: 0.926072<br>
Recall: 0.946541<br>
F1 score: 0.936195</p>
<h3>BERT Transformers model</h3>
<p>Testing results of this model are the following:<br>
Accuracy: 0.932000<br>
Precision: 0.936950<br>
Recall: 0.962349<br>
F1 score: 0.949480</p>
<h3>Conclusion</h3>
<p>The LSTM model starts overfitting early during the training process compared to the BERT model, 
which means the BERT model has even more potential. But in our case, BERT performed a little better 
than LSTM but no significant difference, partly because I did not reach the point right before the BERT 
starts overfitting but mostly because they both performed well on this particular dataset.<br>
Using the LSTM model is more demanding than BERT. To use LSTM effectively, the following process 
is necessary, you need to clean the text (remove stopwords, URLs, emojis, punctuation, HTML tags), 
then do word stemming (replace words by their root form), then fit the tokenizer to your resulting 
vocabulary in the intention of converting words to numbers, finally you encode your sentences. Now, 
the data is ready to train your LSTM model.<br>
Whereas for the BERT model you just need to apply the BERT tokenizer to your sentences, and yes, 
your data is ready to fine-tune the pretrained BERT model.
So, using the BERT model is overall a better option than LSTM!
</p>

[<h2 align="center">Named Entity Recognition & Sentiment Analysis</h2>](https://github.com/HadiAbouDaya/VeeSmart/blob/main/NER%20and%20Sentiment%20Analysis.ipynb)
<p>My goal in this part, I need to make a function that is fed with a sentence such as “The food is as 
always good and the waiters are very friendly. Plus our waiter Izzat B is super nice. Will come 
again very soon. Also the deserts are super yummy!” and output this: 
{'Review': 'The food is as always good and the waiters are very friendly. Plus our waiter Izzat B 
is super nice. Will come again very soon. Also the deserts are super yummy!', 'Date': '2014-10-11 
03:34:02', 'opinionMining': [{'SubReview': 'The food is as always good and the waiters are very 
friendly.', 'aspects': {'food', 'waiter'}, 'sentiment': ['pos']}, {'SubReview': 'Plus our waiter Izzat B 
is super nice.', 'aspects': {'waiter'}, 'sentiment': ['pos']}, {'SubReview': 'Will come again very 
soon.', 'aspects': '', 'sentiment': ['pos']}, {'SubReview': 'Also the deserts are super yummy!', 
'aspects': {'desert'}, 'sentiment': ['pos']}], 'WholeReviewSentiment': ['pos']}
I achieved this by creating the following dictionary: {"Review":review, "Date":date, 
"opinionMining":opinionMining, "WholeReviewSentiment":WholeReviewSentiment} where 
review is the text review, date is the date of publishing the review, WholeReviewSentiment is 1 
if it is 4 or 5 star review, else it is a 0, and finally opinionMining divides the review into sub 
sentences and extracts the nouns from it, predicts its sentiment (0 or 1) by using the fine-tuned 
BERT Transformers model. This function can loop through the rows of a dataframe and produce 
these JSON like format which can be useful for businesses trying to analyze the sentiment of a 
large number of reviews.</p>
