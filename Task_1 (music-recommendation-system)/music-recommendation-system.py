"""
Music Recommendation System

This system predicts whether a user will listen to a song again within a month
based on their listening history. It uses machine learning to analyze patterns in
user listening behavior and recommends songs accordingly.

Author: Faraz Ahmed

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib
import os

class MusicRecommendationSystem:
    def __init__(self, data_path=None):
        """
        Initialize the music recommendation system.
        
        Args:
            data_path (str): Path to the dataset file.
        """
        self.data_path = data_path
        self.model = None
        self.preprocessor = None
        self.features = None
        self.target = None
        
    def load_data(self, data_path=None):
        """
        Load and preprocess the dataset.
        
        Args:
            data_path (str, optional): Path to the dataset file.
            
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if data_path:
            self.data_path = data_path
            
        if self.data_path is None:
            # If no data path is provided, generate synthetic data for demonstration
            print("No data path provided. Generating synthetic data for demonstration...")
            return self._generate_synthetic_data()
        
        try:
            # Try to load the data from the provided path
            data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Generating synthetic data instead...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_users=1000, n_songs=500, n_samples=10000):
        """
        Generate synthetic data for demonstration purposes.
        
        Args:
            n_users (int): Number of unique users
            n_songs (int): Number of unique songs
            n_samples (int): Number of listening events
            
        Returns:
            pd.DataFrame: Synthetic dataset
        """
        print(f"Generating synthetic dataset with {n_samples} samples...")
        
        # Generate user IDs
        user_ids = [f"user_{i}" for i in range(n_users)]
        
        # Generate song IDs
        song_ids = [f"song_{i}" for i in range(n_songs)]
        
        # Generate synthetic data
        np.random.seed(42)
        data = {
            'user_id': np.random.choice(user_ids, n_samples),
            'song_id': np.random.choice(song_ids, n_samples),
            'listen_count': np.random.randint(1, 20, n_samples),
            'listen_time': np.random.randint(30, 300, n_samples),  # listening time in seconds
            'skip_count': np.random.randint(0, 5, n_samples),
            'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
            'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples),
            'genre': np.random.choice(['pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical', 'country'], n_samples),
            'artist_popularity': np.random.randint(1, 100, n_samples),
            'song_age_days': np.random.randint(1, 1000, n_samples),
            'timestamp': [(datetime.datetime.now() - datetime.timedelta(days=np.random.randint(1, 60))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(n_samples)]
        }
        
        df = pd.DataFrame(data)
        
        # Create the target variable: 1 if the user listened to the same song within a month, 0 otherwise
        # For simulation, we'll say users with higher listen counts are more likely to listen again
        listen_prob = 0.2 + (df['listen_count'] / 20) * 0.6 - (df['skip_count'] / 5) * 0.3
        df['listened_again'] = (np.random.random(n_samples) < listen_prob).astype(int)
        
        print("Synthetic data generated successfully!")
        return df
    
    def explore_data(self, data):
        """
        Explore the dataset and print basic statistics.
        
        Args:
            data (pd.DataFrame): The dataset to explore.
        """
        print("\n--- Dataset Exploration ---")
        print(f"Dataset shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        
        print("\nData types:")
        print(data.dtypes)
        
        print("\nBasic statistics:")
        print(data.describe())
        
        print("\nTarget variable distribution:")
        target_counts = data['listened_again'].value_counts(normalize=True) * 100
        print(f"Listened again: {target_counts[1]:.2f}%")
        print(f"Did not listen again: {target_counts[0]:.2f}%")
        
        # Convert timestamp to datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day'] = data['timestamp'].dt.day
            data['month'] = data['timestamp'].dt.month
            data['weekday'] = data['timestamp'].dt.weekday
            
            print("\nTemporal distribution:")
            print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data for model training.
        
        Args:
            data (pd.DataFrame): The dataset to preprocess.
            
        Returns:
            tuple: Preprocessed features and target variable.
        """
        print("\n--- Data Preprocessing ---")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['weekday'] = df['timestamp'].dt.weekday
        
        # Features to use for the model
        categorical_features = [col for col in ['user_id', 'song_id', 'genre', 'time_of_day', 'day_of_week'] 
                                if col in df.columns]
        
        numerical_features = [col for col in ['listen_count', 'listen_time', 'skip_count', 'artist_popularity', 
                                            'song_age_days', 'hour', 'day', 'month', 'weekday'] 
                             if col in df.columns]
        
        # Target variable
        target = 'listened_again'
        
        # Store feature and target names
        self.features = categorical_features + numerical_features
        self.target = target
        
        # Split data into features and target
        X = df[self.features]
        y = df[self.target]
        
        print(f"Selected features: {self.features}")
        print(f"Target variable: {self.target}")
        
        return X, y
    
    def create_preprocessor(self):
        """
        Create a preprocessor for the features.
        
        Returns:
            ColumnTransformer: The preprocessor pipeline.
        """
        # Identify categorical and numerical features
        categorical_features = [col for col in self.features 
                               if col in ['user_id', 'song_id', 'genre', 'time_of_day', 'day_of_week']]
        
        numerical_features = [col for col in self.features 
                             if col not in categorical_features]
        
        # Create transformers for each type of feature
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers into a preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def build_model(self):
        """
        Build the machine learning model pipeline.
        
        Returns:
            Pipeline: The complete model pipeline.
        """
        print("\n--- Building Model ---")
        
        # Create preprocessor if not already created
        if self.preprocessor is None:
            self.create_preprocessor()
        
        # Create the full pipeline with preprocessor and model
        model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model = model
        print("Model pipeline built successfully!")
        return model
    
    def train_model(self, X, y):
        """
        Train the machine learning model.
        
        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target variable.
            
        Returns:
            Pipeline: The trained model.
        """
        print("\n--- Training Model ---")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        
        # Build the model if not already built
        if self.model is None:
            self.build_model()
        
        # Train the model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Store the test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        print("Model trained successfully!")
        return self.model
    
    def save_model(self, model_path='models', model_name='music_recommendation_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Directory to save the model.
            model_name (str): Name of the model file.
            
        Returns:
            str: Path to the saved model.
        """
        if self.model is None:
            print("No model to save. Please train the model first.")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Save the model
        model_file = os.path.join(model_path, model_name)
        joblib.dump(self.model, model_file)
        print(f"Model saved to {model_file}")
        
        return model_file
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model.
            
        Returns:
            Pipeline: The loaded model.
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def recommend_songs(self, user_id, data, top_n=10):
        """
        Recommend songs for a user.
        
        Args:
            user_id (str): The ID of the user.
            data (pd.DataFrame): The dataset containing song information.
            top_n (int): Number of songs to recommend.
            
        Returns:
            pd.DataFrame: Top N recommended songs.
        """
        if self.model is None:
            print("No model available for predictions. Please train or load a model first.")
            return None
        
        # Filter songs that the user hasn't listened to yet
        user_listened_songs = set(data[data['user_id'] == user_id]['song_id'])
        new_songs = data[~data['song_id'].isin(user_listened_songs)].drop_duplicates('song_id')
        
        if new_songs.empty:
            print(f"No new songs found for user {user_id}.")
            return pd.DataFrame()
        
        # Prepare data for prediction
        X_pred = new_songs[self.features]
        
        # Predict the probability of listening again
        try:
            listen_prob = self.model.predict_proba(X_pred)[:, 1]
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
        
        # Add prediction probabilities to the dataframe
        new_songs['listen_probability'] = listen_prob
        
        # Sort by probability and get top N recommendations
        recommendations = new_songs.sort_values('listen_probability', ascending=False).head(top_n)
        
        return recommendations[['song_id', 'genre', 'listen_probability']]
    
    def generate_user_report(self, user_id, data):
        """
        Generate a listening report for a specific user.
        
        Args:
            user_id (str): The ID of the user.
            data (pd.DataFrame): The dataset containing listening history.
            
        Returns:
            dict: User listening report.
        """
        user_data = data[data['user_id'] == user_id]
        
        if user_data.empty:
            return {"error": f"No data found for user {user_id}"}
        
        # Basic listening stats
        total_listens = len(user_data)
        unique_songs = user_data['song_id'].nunique()
        
        # Calculate average listening time
        avg_listen_time = user_data['listen_time'].mean() if 'listen_time' in user_data.columns else 0
        
        # Calculate most listened genres
        genre_counts = user_data['genre'].value_counts() if 'genre' in user_data.columns else pd.Series()
        top_genres = genre_counts.to_dict() if not genre_counts.empty else {}
        
        # Most listened songs
        song_counts = user_data['song_id'].value_counts().head(5).to_dict()
        
        # Time of day preferences
        time_of_day_counts = user_data['time_of_day'].value_counts().to_dict() if 'time_of_day' in user_data.columns else {}
        
        # Compile the report
        report = {
            "user_id": user_id,
            "total_listens": total_listens,
            "unique_songs": unique_songs,
            "avg_listen_time_seconds": avg_listen_time,
            "top_genres": top_genres,
            "top_songs": song_counts,
            "time_of_day_preferences": time_of_day_counts
        }
        
        return report
    
    def run(self, data_path=None, model_path=None, save_model=True):
        """
        Run the complete workflow: load data, preprocess, train model, and save model.
        
        Args:
            data_path (str, optional): Path to the dataset file.
            model_path (str, optional): Path to load a pre-trained model.
            save_model (bool): Whether to save the trained model.
            
        Returns:
            MusicRecommendationSystem: Self instance for method chaining.
        """
        # Load data
        data = self.load_data(data_path)
        
        # Explore data
        data = self.explore_data(data)
        
        # If model path is provided, try to load the model
        if model_path:
            loaded_model = self.load_model(model_path)
            if loaded_model:
                self.model = loaded_model
                return self
        
        # Preprocess data
        X, y = self.preprocess_data(data)
        
        # Build and train model
        self.train_model(X, y)
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        return self



if __name__ == "__main__":

    recommender = MusicRecommendationSystem()
    

    recommender.run()
    

    data = recommender._generate_synthetic_data(n_samples=2000)
    
    # Get recommendations for a specific user
    user_id = "user_42"  # Example user ID
    recommendations = recommender.recommend_songs(user_id, data, top_n=5)
    
    # Print recommendations
    if recommendations is not None and not recommendations.empty:
        print(f"\nTop 5 song recommendations for {user_id}:")
        print(recommendations)
    
    # Generate user report
    report = recommender.generate_user_report(user_id, data)
    print(f"\nUser report for {user_id}:")
    for key, value in report.items():
        if isinstance(value, dict) and len(value) > 5:
            # For dictionaries with many entries, just show the top 5
            print(f"{key}:")
            for k, v in list(value.items())[:5]:
                print(f"  {k}: {v}")
            print("  ...")
        else:
            print(f"{key}: {value}")
