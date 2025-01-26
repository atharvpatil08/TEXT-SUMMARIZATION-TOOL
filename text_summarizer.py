from transformers import pipeline



# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")


# Example text
text = """
Natural Language Toolkit, or NLTK, is a leading platform for building Python programs to work with human language data. 
It provides easy-to-use interfaces to over 50 corpora and lexical resources. 
NLTK includes tools for text processing, such as tokenization, tagging, and parsing. 
This library is widely used in academic and industrial research projects.
"""

# Generate abstractive summary
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)

# Print the summary
print("Abstractive Summary:")
print(summary[0]['summary_text'])
