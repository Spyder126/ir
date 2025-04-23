from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define the corpus
corpus = [
    "India has the second-largest population in the world.",
    "It is surrounded by oceans from three sides which are Bay of Bengal in the east, the Arabian Sea in the west and Indian Ocean in the south.",
    "Tiger is the national animal of India.",
    "Peacock is the national bird of India.",
    "Mango is the national fruit of India."
]

# Step 2: Preprocess (convert to lowercase)
processed_corpus = [sentence.lower() for sentence in corpus]

# Step 3: Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_corpus)

# Step 4: Input question
question = "Which is the national bird of India?"
question_lower = question.lower()

# Step 5: Vectorize the question
question_vector = vectorizer.transform([question_lower])

# Step 6: Compute cosine similarity
similarities = cosine_similarity(question_vector, X).flatten()

# Step 7: Get the most relevant answer
best_match_index = similarities.argmax()
answer = corpus[best_match_index]

# Step 8: Output the result
print("Question:", question)
print("Answer:", answer)
