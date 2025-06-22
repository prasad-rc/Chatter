from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

# Load the Sentence Transformer model (lightweight and fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample course catalog (can be replaced by DB or file later)
courses = [
    {
        "title": "Python for Beginners",
        "description": "A beginner-friendly course to learn Python programming from scratch.",
    },
    {
        "title": "AI for Everyone",
        "description": "An introduction to artificial intelligence for non-programmers.",
    },
    {
        "title": "Advanced Machine Learning",
        "description": "Covers algorithms like XGBoost, SVM, and deep learning architectures.",
    },
    {
        "title": "Web Development Bootcamp",
        "description": "HTML, CSS, JavaScript, and full-stack web development skills.",
    },
    {
        "title": "Data Science with Python",
        "description": "Data analysis, visualization, and machine learning with Python.",
    },
    {
        "title": "Deep Learning with PyTorch",
        "description": "Learn how to build and train deep neural networks using PyTorch.",
    }
]

# Precompute embeddings for course descriptions
course_descriptions = [course["description"] for course in courses]
course_embeddings = model.encode(course_descriptions, convert_to_tensor=True)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        user_input = request.form["interest"]
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(user_embedding, course_embeddings)[0]
        
        # Get top 3 most similar courses
        top_results = np.argsort(-cosine_scores)[:3]
        recommendations = [courses[idx]["title"] for idx in top_results]

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
