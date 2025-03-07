import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Union, List
import logging
import pickle
from datetime import datetime, timedelta

class DataLoader:
    """
    Utility class to load and manage March Madness data.
    """
    
    def __init__(self, data_dir: str = "../data", cache_dir: str = "../cache"):
        """Initialize DataLoader with data and cache directories"""
        self.data_dir = self._validate_path(data_dir)
        self.cache_dir = self._validate_path(cache_dir)
        self.cache_duration = timedelta(hours=24)  # Cache expires after 24 hours
        self.logger = self._setup_logger()

    def _validate_path(self, path: str) -> str:
        """Validate and create directory if it doesn't exist"""
        abs_path = os.path.abspath(path)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path

    def _cache_key(self, data_type: str) -> str:
        """Generate cache key for data type"""
        return os.path.join(self.cache_dir, f"{data_type}_cache.pkl")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid"""
        if not os.path.exists(cache_path):
            return False
        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - modified_time < self.cache_duration

    def _load_from_cache(self, data_type: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load data from cache if valid"""
        cache_path = self._cache_key(data_type)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.logger.info(f"Loading {data_type} from cache")
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
        return None

    def _save_to_cache(self, data: Dict[str, pd.DataFrame], data_type: str):
        """Save data to cache"""
        try:
            cache_path = self._cache_key(data_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved {data_type} to cache")
        except Exception as e:
            self.logger.error(f"Error saving to cache: {str(e)}")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DataLoader")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def load_tournament_data(self, gender: str = 'M') -> Dict[str, pd.DataFrame]:
        """
        Load tournament data with caching.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing tournament data
        """
        # Try loading from cache first
        cache_key = f'tournament_{gender}'
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        # If not in cache, load from files
        files = {
            'results': f'{gender}NCAATourneyCompactResults.csv',
            'detailed': f'{gender}NCAATourneyDetailedResults.csv',
            'seeds': f'{gender}NCAATourneySeeds.csv',
            'slots': f'{gender}NCAATourneySlots.csv'
        }

        # Load each file individually to better handle errors
        data = {}
        for key, filename in files.items():
            df = self._load_file(filename)
            if df is None:
                self.logger.error(f"Failed to load {filename}")
                return None
            data[key] = df
            
        if self._validate_tournament_data(data):
            # Save to cache if validation passes
            self._save_to_cache(data, cache_key)
            return data
        return None

    def _load_file(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a single CSV file with error handling"""
        try:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {str(e)}")
            return None

    def _validate_tournament_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate loaded tournament data"""
        try:
            for key, df in data.items():
                if df is None:
                    self.logger.error(f"Missing data for {key}")
                    return False
                if df.empty:
                    self.logger.error(f"Empty dataframe for {key}")
                    return False
                
            # Basic validation - check if seeds dataframe has expected columns
            if 'Season' not in data['seeds'].columns or 'TeamID' not in data['seeds'].columns:
                self.logger.error("Seeds dataframe missing required columns")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False

    def load_regular_season(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load regular season data with caching.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            pd.DataFrame: Regular season results
        """
        cache_key = f'regular_season_{gender}'
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data.get('data')

        try:
            filename = f"{gender}RegularSeasonCompactResults.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self._save_to_cache({'data': df}, cache_key)
            return df
        except Exception as e:
            self.logger.error(f"Error loading regular season data: {e}")
            return None
            
    def load_regular_season_detailed(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load detailed regular season data.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            pd.DataFrame: Detailed regular season results
        """
        try:
            filename = f"{gender}RegularSeasonDetailedResults.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading detailed regular season data: {e}")
            return None
            
    def load_teams(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load teams data.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            pd.DataFrame: Teams data
        """
        try:
            filename = f"{gender}Teams.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading teams data: {e}")
            return None
            
    def load_team_conferences(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load team conferences data.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            pd.DataFrame: Team conferences data
        """
        try:
            filename = f"{gender}TeamConferences.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading team conferences data: {e}")
            return None
            
    def load_conferences(self) -> Optional[pd.DataFrame]:
        """
        Load conferences data.
        
        Returns:
            pd.DataFrame: Conferences data
        """
        try:
            filename = "Conferences.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading conferences data: {e}")
            return None
            
    def load_massey_ordinals(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load Massey Ordinals rankings data.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data (note: only available for men)
            
        Returns:
            pd.DataFrame: Massey Ordinals data
        """
        if gender != 'M':
            self.logger.warning("Massey Ordinals data only available for men's teams")
            return None
            
        try:
            filename = "MMasseyOrdinals.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Massey Ordinals data: {e}")
            return None
            
    def load_sample_submission(self, stage: int = 1) -> Optional[pd.DataFrame]:
        """
        Load sample submission file.
        
        Args:
            stage (int): 1 or 2 for different submission stages
            
        Returns:
            pd.DataFrame: Sample submission data
        """
        try:
            filename = f"SampleSubmissionStage{stage}.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading sample submission data: {e}")
            return None
            
    def load_cities(self) -> Optional[pd.DataFrame]:
        """
        Load cities data.
        
        Returns:
            pd.DataFrame: Cities data
        """
        try:
            filename = "Cities.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading cities data: {e}")
            return None
            
    def load_game_cities(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load game cities data.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            pd.DataFrame: Game cities data
        """
        try:
            filename = f"{gender}GameCities.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading game cities data: {e}")
            return None
            
    def load_seasons(self, gender: str = 'M') -> Optional[pd.DataFrame]:
        """
        Load seasons data.
        
        Args:
            gender (str): 'M' for men's data, 'W' for women's data
            
        Returns:
            pd.DataFrame: Seasons data
        """
        try:
            filename = f"{gender}Seasons.csv"
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            self.logger.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading seasons data: {e}")
            return None

# Create singleton instance
data_loader = DataLoader()