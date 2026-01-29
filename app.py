import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
new_df = pd.read_csv("new_df.csv")

@app.route('/')
def index():
    all_movies = new_df['title'].to_list()
    return render_template('index.html', movie_list=all_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    select_movie = request.form['selected_movie']
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tag']).toarray()
    similarity = cosine_similarity(vectors)
    
    try:
        movie_index = new_df[new_df['title'] == select_movie].index[0]
        distances = similarity[movie_index]
        movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in movie_indices:
            recommended_movies.append(new_df.iloc[i[0]].title)
            
    except Exception as e:
        recommended_movies = ["Movie not found!"]

    # Wapis index.html par bhejein par results ke saath
    all_movies = new_df['title'].to_list()
    return render_template('index.html', 
                           movie_list=all_movies, 
                           recommendations=recommended_movies, 
                           chosen_movie=select_movie)

if __name__ == '__main__':
    app.run(debug=True)