# Music Recommendation System

A machine learning-based music recommendation system inspired by Spotify's approach. 
This system predicts a user's likelihood of repeatedly listening to a song within a 
set timeframe and generates personalized song recommendations.

## Project Overview

This project implements a recommendation system that:
- Predicts whether a user will listen to a specific song again within a month
- Uses historical listening data to identify patterns in user behavior
- Generates personalized song recommendations based on listening patterns
- Provides user listening reports and insights

## Features

- **Data Processing**: Handles both real and synthetic music listening data
- **Machine Learning**: Uses Random Forest classifier for prediction
- **Feature Engineering**: Extracts temporal patterns and user behaviors
- **Model Evaluation**: Includes comprehensive metrics and visualization
- **Recommendation Engine**: Recommends songs based on predicted preferences
- **User Reports**: Generates listening statistics and preferences

## Installation

```bash
git clone https://github.com/farazahmed2048/codealpha_tasks.git
cd Task_1 (music-recommendation-system)
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Usage

### Basic Usage

```python
from music_recommendation import MusicRecommendationSystem

# Initialize the system
recommender = MusicRecommendationSystem()

# Run the complete workflow with synthetic data
recommender.run()

# Get recommendations for a specific user
user_id = "user_42"
recommendations = recommender.recommend_songs(user_id, data, top_n=5)
print(recommendations)

# Generate user listening report
report = recommender.generate_user_report(user_id, data)
print(report)
```

### Using Your Own Dataset

```python
# Initialize with your data
recommender = MusicRecommendationSystem(data_path='your_dataset.csv')

# Run the complete workflow
recommender.run()
```

### Dataset Format

The system works with datasets containing the following columns:
- `user_id`: Unique identifier for each user
- `song_id`: Unique identifier for each song
- `listen_count`: Number of times the user has listened to the song
- `listen_time`: Duration of listening in seconds
- `skip_count`: Number of times the user skipped the song
- `time_of_day`: When the song was played (morning, afternoon, evening, night)
- `day_of_week`: Day of the week when the song was played
- `genre`: Music genre
- `artist_popularity`: Popularity score of the artist
- `song_age_days`: Age of the song in days
- `timestamp`: Timestamp of when the song was played
- `listened_again`: Target variable (1 if user listened to the song again within a month, 0 otherwise)

## Project Structure

```
music-recommendation-system/
│
├── music_recommendation.py    # Main implementation
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
│
├── models/                    # Saved models
│   └── music_recommendation_model.pkl
│
├── data/                      # Data directory
│   └── example_data.csv       # Sample dataset (optional)
│
└── notebooks/                 # Jupyter notebooks (optional)
    └── exploration.ipynb      # Data exploration example
```

## How It Works

1. **Data Loading**: The system loads music listening data or generates synthetic data if none is provided
2. **Preprocessing**: Features are processed and transformed for machine learning
3. **Model Training**: A Random Forest classifier is trained to predict repeat listening
4. **Evaluation**: Model performance is evaluated using accuracy, precision, recall, and F1 score
5. **Recommendation**: Songs are ranked based on predicted probability of repeat listening
6. **Reporting**: User listening patterns and preferences are analyzed

## Future Improvements

- Implement collaborative filtering for enhanced recommendations
- Add content-based filtering using audio features
- Develop A/B testing framework for recommendation strategies
- Add support for real-time recommendation updates
- Implement deep learning models for improved prediction

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Spotify's recommendation system
- Built as part of a machine learning internship project

## Author

Faraz Ahmed
