🛒 Electronics Recommendation System

A hybrid Recommendation System built using Collaborative Filtering (ALS) and Content-Based Filtering (TF-IDF) on Amazon Electronics reviews.
The project demonstrates real-world recommender system challenges such as sparsity, cold-start, and dataset alignment, with a simple Flask UI for interaction.

🚀 Features

✅ Collaborative Filtering using Alternating Least Squares (ALS)

✅ Content-Based Recommendations using TF-IDF

✅ Hybrid Recommendation Strategy

✅ Cold-start handling for new users

✅ Simple Flask web interface   


                     
📊 Dataset

Source: Amazon Electronics Reviews

Fields used:

reviewerID – user identifier

asin – product identifier

reviewText – review text

overall – rating

Due to size constraints, only a processed subset is used for UI display, while the collaborative model is trained on a larger interaction matrix.

🧠 Recommendation Approaches
1️⃣ Collaborative Filtering (ALS)

Uses implicit feedback (ratings)

Learns latent user–item embeddings

Handles large sparse matrices efficiently

Best for existing users

2️⃣ Content-Based Filtering

TF-IDF over review text

Recommends similar items based on product descriptions

Useful for item similarity & cold-start

3️⃣ Hybrid Recommender

Combines collaborative + content signals

Falls back to popularity for unseen users

🗂️ Project Structure           

Electronics_Recommendation_System/
│
├── app.py                 # Flask application
├── src/
│   ├── recommender_cf.py
│   ├── recommender_content.py
│   └── recommender_hybrid.py
│
├── notebooks/             # EDA & model development
├── data/
│   └── processed/         # Processed datasets (not tracked in git)
├── models/                # Trained models (not tracked in git)
│
├── app/
│   ├── templates/
│   └── static/
│
├── requirements.txt
└── README.md


🖥️ Running Locally               

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the app
python app.py

3️⃣ Open browser
http://127.0.0.1:5000       

🧪 How to Use

Enter a user ID

If user exists → personalized recommendations

If user is new → popular items shown

Each recommendation displays:

Product ASIN

Short review text (if available)     

📈 Evaluation

Precision@K used for offline evaluation         
Sparse interaction matrix (~99.9% sparsity)         
Demonstrates realistic recommender performance tradeoffs                   

📌 Future Improvements

Store full ASIN → metadata mapping

Add product titles/images        
Use approximate nearest neighbors (FAISS)       
Improve hybrid weighting strategy        
Add online evaluation metrics         
