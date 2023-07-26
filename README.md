# NLP

## AUTOMATED TEXT SUMMARIZATION



# Introduction

 
Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the intitial text while at the same time preserving key information elements and the meaning of content.

Automatic Text summaraization is one of Natural Language processing Fields, a subject area utilizing Linguistic, computer science and statistics. The main goal is taking information that comes on document, extract its content and present to the user its mainly important content in a reduced form that satisfies the user’s desires. In simple terms, it is a process of generating a concise and meaniful summary of text from multiple text resources such as books, news articles, blog posts, research papers, emails and tweets

 There are two main paradigms to summarization: extractive and abstractive. Extractive approaches form summaries by extracting and concatenating the most important spans from the source text, while abstractive methods generate candidate summaries that contain novel words and phrases not featured in the source text, usually requiring additional rewriting operations. An extractive method has the benefit of maintaining reasonable levels of grammaticality and accuracy. Conversely, the ability to generalize, paraphrase, and introduce additional knowledge are key features in an abstractive framework.


We (humans ) are generally good at this type of task as it involves first understanding the meaning of the source document and the distiling the meaning and capturing salient details in the new description.as such result our goal is to have the resulting summaries as good as those written by human. It is not enough to just generate words and phrases that capture the gist of the source document, the summary should be accurate and should read fluently as new standalone this is what makes this task difficult.

With extremely incresing availlable information and the limited time people have retrieving information is a fasterway is very desirable, the huge amount of all the availlable information that exist today is in form of unstructured data, so called textual data. Textual information is embedded in textual data. By having this tool ,the textual information can be processed faster by human for decision making process. This tool is benefial for all types of people in all kinds of domain.



## Description of the dataset : 

The dataset that we have choosen for this task is medical related dataset. It is an excel sheet containing 1000 records containing details of various types of medicine in the form of  long texts.

We are also using the pre-trained Glove word embeddings to solve a text classification problem using this technique.


## Approach :

The approach for this project was extracting sentence from our input file, transforming it into an array, then identify the most important keywords in the source document and combining them together to produce what is likely to be a readable summary (with all important information).


## Methodology : 
Sentence segmentation is the first step we applied in our automated text summarization project. Word embedding sentence segmentation was applied inorder to group words of simillar context together and dissimilar word are positioned far away from each other.

 Normally, to Parse a paragraph of text, a simple and limted way of diving it into sentences would be to use (.) to obtain thier ends. The dificulty is for the computer to understand the punctuantion characters that appear in the middle of sentence, where it does not mean the end of sentence. We used split method for this challenge.


The Following are all Methods followed in completing the project :

## Split method 
Split method was used (line. Split). Using the split method on the input returned a List of string after Breaking the given string by specified separator. Then we were able to treat each characters (, .‘’ !?) as potential rather definite end of sentence markers.


## Text prepocessing 
After a paragraph has been segmented into sentences, then we applied text preprocessing to each sentence. Preprocessing input text simply means putting the data into a predictable and analazable form.  Stop Word removal and tokenisation were implemented for This task. 

•	Tokenization :
It’s the process of breaking a stream of textual data into words, terms, sentences, symbols, or some other meaningful elements called tokens. For tokenization we used sent_tokenize from nltk  library which Is used to split text and return a List of sentences.

•	Stop word removal : 
Stop words are the most common words in any language that do not carry any meaning and are usually ignored by NLP. In English, examples of stop words are “a”, “and”, “the” and “of”. We will used the nltk package to filter the stop words.


Part of speech tagging : 

Each token was attached with its part of speech. The challenging issues in assigning a part of speech to  a Word   that may have more than one part of speech.for example,the word ‘place’ can belong to a verb or noun.to resolve this problem ,we found a disambiguaration rule .



## Making vectors out of sentences :

After tokenization, we made vectors out of  sentences. Having vector representations of words helped to analyze the semantics of textual contents better. Text vectorization is converting text into vectors of values to represent their meanings. we create a function to vectorize each data point. The sentence is the mean representation of each word. For this project we used Glove in python. Finally, we vectorize the whole dataset and save the vectorized NumPy array as a file, so we don’t have to go through this process again every time running the code.


## Finding similarity between words :

 Cosine Similarity Approach was applied. The main reason we applied this rule was to find simar words and to avoid repetition of words. Cosine Similarity measures the similarity between two sentences or documents in terms of the value within the range of [-1, 1] whichever you want to measure.


## Keyword identification :

Keyword was extracted from the token List, we removed punctuations and special characters. Since puctuation affects interpretation of meaning of structure and its important. Deleting punctuation reduces the ability of the follow on semantic parsing functionality.



## Sentence extraction :

The sentence extraction was conducted on sentence that has high probality value. Convert the content into a dictionary. Finally extracted the sentences as summary and saved in a csv file using pandas library
 


## Important metric parameters used for this project


## Tokenize

It gives tuples of integers having same semantics as string slices. Tokenization is breaking the raw text into small chunks. Tokenization breaks the raw text into words, sentences called tokens. These tokens help in understanding the context or developing the model for the NLP.


## Corpus

It is a collection of pre-written texts. A corpus is a large and structured set of machine-readable texts that have been produced in a natural communicative setting. Its plural is corpora. They can be derived in different ways like text that was originally electronic, transcripts of spoken language and optical character recognition, etc.


Sklearn.metrics.pairwise

Compute cosine similarity between samples in X and Y.


## Cosine similarity 

Cosine similarity is one of the metric to measure the text-similarity between two documents irrespective of their size in Natural language Processing. A word is represented into a vector form. The text documents are represented in n-dimensional vector space.


## NetworkX

NetworkX is a package for the Python programming language that's used to create, manipulate, and study the structure, dynamics, and functions of complex graph networks.
 
## Conclusion

The amont of textual information is growing exponentialy as the Need for automatic text summarization tools. Both extracting and abstracting text summarization Methods might bring us solutions for keeping up with the growing amount of information. The challenging issues we encounterd in automated text summarization, is to produce a Shorter version of text without loosing any valuable information from the original texts.In the beginning, we applied supervised self-created learning algorithms , but it was not showing the desired result so we used unsupervised learning algorithm and connected pre-trained Glove model with our solution, which made it an accurate solution. We have attached a snap jut to show a glimpse of result obtained at the end.


 









