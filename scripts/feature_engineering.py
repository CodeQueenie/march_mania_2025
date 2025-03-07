import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sklearn.preprocessing import StandardScaler
from utils.logger import app_logger as logger
from utils.data_loader import DataLoader

# Initialize data loader with correct paths
data_dir = os.path.join(project_root, 'data')
cache_dir = os.path.join(project_root, 'cache')
data_loader = DataLoader(data_dir=data_dir, cache_dir=cache_dir)

def calculate_team_season_stats(df_regular_season, team_id, season):
    """
    Calculate team statistics for a specific season.
    
    Args:
        df_regular_season (pd.DataFrame): Regular season results
        team_id (int): Team ID
        season (int): Season year
        
    Returns:
        dict: Dictionary with team statistics
    """
    # Filter games where the team played
    team_games = df_regular_season[(df_regular_season['Season'] == season) & 
                                  ((df_regular_season['WTeamID'] == team_id) | 
                                   (df_regular_season['LTeamID'] == team_id))]
    
    if team_games.empty:
        return None
    
    # Calculate basic stats
    wins = len(team_games[team_games['WTeamID'] == team_id])
    losses = len(team_games[team_games['LTeamID'] == team_id])
    games_played = wins + losses
    
    if games_played == 0:
        return None
    
    win_pct = wins / games_played
    
    # Calculate points scored and allowed
    points_scored = 0
    points_allowed = 0
    
    for _, game in team_games.iterrows():
        if game['WTeamID'] == team_id:
            points_scored += game['WScore']
            points_allowed += game['LScore']
        else:
            points_scored += game['LScore']
            points_allowed += game['WScore']
            
    avg_points_scored = points_scored / games_played
    avg_points_allowed = points_allowed / games_played
    point_diff = avg_points_scored - avg_points_allowed
    
    # Return dictionary with stats
    return {
        'TeamID': team_id,
        'Season': season,
        'Wins': wins,
        'Losses': losses,
        'GamesPlayed': games_played,
        'WinPct': win_pct,
        'AvgPointsScored': avg_points_scored,
        'AvgPointsAllowed': avg_points_allowed,
        'PointDiff': point_diff
    }

def extract_seed_number(seed_str):
    """Extract numeric part from seed string (e.g., 'W01' -> 1)"""
    if pd.isna(seed_str):
        return None
    try:
        return int(seed_str[1:3])
    except:
        return None

def generate_team_seasons_data(gender='M'):
    """
    Generate team statistics for all seasons.
    
    Args:
        gender (str): 'M' for men's data, 'W' for women's data
        
    Returns:
        pd.DataFrame: DataFrame with team season statistics
    """
    logger.info(f"Generating team season statistics for {gender} data")
    
    # Load regular season data
    df_regular_season = data_loader.load_regular_season(gender)
    
    if df_regular_season is None:
        logger.error(f"Failed to load regular season data for {gender}")
        return None
    
    # Load teams data
    df_teams = data_loader.load_teams(gender)
    
    if df_teams is None:
        logger.error(f"Failed to load teams data for {gender}")
        return None
    
    # Load tournament seeds data
    tournament_data = data_loader.load_tournament_data(gender)
    
    if tournament_data is None:
        logger.error(f"Failed to load tournament data for {gender}")
        return None
    
    df_seeds = tournament_data['seeds']
    
    # Get unique team and season combinations
    all_teams = df_teams['TeamID'].unique()
    all_seasons = df_regular_season['Season'].unique()
    
    # Calculate team season stats
    team_season_stats = []
    
    for season in tqdm(all_seasons, desc=f"Processing {gender} seasons"):
        for team_id in all_teams:
            stats = calculate_team_season_stats(df_regular_season, team_id, season)
            if stats:
                team_season_stats.append(stats)
    
    # Convert to DataFrame
    df_team_season_stats = pd.DataFrame(team_season_stats)
    
    # Add tournament seed information
    df_seeds_processed = df_seeds.copy()
    df_seeds_processed['SeedNumber'] = df_seeds_processed['Seed'].apply(extract_seed_number)
    
    # Merge tournament seed info
    df_team_season_stats = pd.merge(
        df_team_season_stats,
        df_seeds_processed[['Season', 'TeamID', 'Seed', 'SeedNumber']],
        on=['Season', 'TeamID'],
        how='left'
    )
    
    logger.info(f"Generated {len(df_team_season_stats)} team season records for {gender}")
    
    return df_team_season_stats

def generate_matchup_features(df_team_stats, team_id_1, team_id_2, season):
    """
    Generate features for a specific matchup between two teams.
    
    Args:
        df_team_stats (pd.DataFrame): Team season statistics
        team_id_1 (int): First team ID
        team_id_2 (int): Second team ID
        season (int): Season year
        
    Returns:
        dict: Dictionary with matchup features or None if data is missing
        
    Raises:
        ValueError: If input parameters are invalid
        KeyError: If required statistics columns are missing
    """
    try:
        # Validate inputs
        if not isinstance(df_team_stats, pd.DataFrame) or df_team_stats.empty:
            return None
            
        if not isinstance(team_id_1, (int, np.integer)) or not isinstance(team_id_2, (int, np.integer)):
            raise ValueError("Team IDs must be integers")
            
        if not isinstance(season, (int, np.integer)):
            raise ValueError("Season must be an integer")
        
        # Get team stats for the given season
        team1_stats = df_team_stats[(df_team_stats["TeamID"] == team_id_1) & 
                                  (df_team_stats["Season"] == season)]
        team2_stats = df_team_stats[(df_team_stats["TeamID"] == team_id_2) & 
                                  (df_team_stats["Season"] == season)]
        
        # If either team doesn't have stats, return None
        if team1_stats.empty or team2_stats.empty:
            return None
        
        team1_stats = team1_stats.iloc[0]
        team2_stats = team2_stats.iloc[0]
        
        # Check for required columns
        required_columns = ["WinPct", "PointDiff", "AvgPointsScored", "AvgPointsAllowed", "GamesPlayed"]
        for column in required_columns:
            if column not in team1_stats or column not in team2_stats:
                raise KeyError(f"Required column '{column}' missing from team statistics")
        
        # Calculate feature differences
        features = {
            "Season": season,
            "Team1ID": team_id_1,
            "Team2ID": team_id_2,
            "WinPctDiff": team1_stats["WinPct"] - team2_stats["WinPct"],
            "PointDiffDiff": team1_stats["PointDiff"] - team2_stats["PointDiff"],
            "AvgPointsScoredDiff": team1_stats["AvgPointsScored"] - team2_stats["AvgPointsScored"],
            "AvgPointsAllowedDiff": team1_stats["AvgPointsAllowed"] - team2_stats["AvgPointsAllowed"],
            "GamesPlayedDiff": team1_stats["GamesPlayed"] - team2_stats["GamesPlayed"],
        }
        
        # Add seed information if available
        if "SeedNumber" in team1_stats and "SeedNumber" in team2_stats and not pd.isna(team1_stats["SeedNumber"]) and not pd.isna(team2_stats["SeedNumber"]):
            features["SeedNumberDiff"] = team1_stats["SeedNumber"] - team2_stats["SeedNumber"]
        else:
            features["SeedNumberDiff"] = np.nan
        
        return features
    except Exception as e:
        print(f"Error generating matchup features for teams {team_id_1} vs {team_id_2}: {e}")
        return None

def generate_submission_features(gender='M'):
    """
    Generate features for submission predictions.
    
    Args:
        gender (str): 'M' for men's data, 'W' for women's data
        
    Returns:
        pd.DataFrame: DataFrame with features for all possible matchups
    """
    logger.info(f"Generating submission features for {gender} data")
    
    # Generate team season stats
    df_team_stats = generate_team_seasons_data(gender)
    
    if df_team_stats is None:
        logger.error(f"Failed to generate team stats for {gender}")
        return None
    
    # Load submission file to get matchups
    df_sample = data_loader.load_sample_submission()
    
    if df_sample is None:
        logger.error("Failed to load sample submission file")
        return None
    
    # Filter by gender (Men's teams are 1000-1999, Women's teams are 3000-3999)
    if gender == 'M':
        df_sample = df_sample[df_sample['ID'].str.contains('_1')]
    else:
        df_sample = df_sample[df_sample['ID'].str.contains('_3')]
    
    # Process each matchup
    features_list = []
    
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc=f"Processing {gender} matchups"):
        # Parse ID to get season and teams
        id_parts = row['ID'].split('_')
        season = int(id_parts[0])
        team1_id = int(id_parts[1])
        team2_id = int(id_parts[2])
        
        # Generate matchup features
        features = generate_matchup_features(df_team_stats, team1_id, team2_id, season)
        
        if features:
            features['ID'] = row['ID']
            features_list.append(features)
    
    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)
    
    logger.info(f"Generated {len(df_features)} matchup features for {gender}")
    
    return df_features

def main():
    """Generate features for both men's and women's tournaments and save to CSV"""
    output_dir = os.path.join(project_root, 'data', 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate features for men's tournament
    df_men_features = generate_submission_features('M')
    if df_men_features is not None:
        men_output_path = os.path.join(output_dir, 'men_features.csv')
        df_men_features.to_csv(men_output_path, index=False)
        logger.info(f"Saved men's features to {men_output_path}")
    
    # Generate features for women's tournament
    df_women_features = generate_submission_features('W')
    if df_women_features is not None:
        women_output_path = os.path.join(output_dir, 'women_features.csv')
        df_women_features.to_csv(women_output_path, index=False)
        logger.info(f"Saved women's features to {women_output_path}")

if __name__ == "__main__":
    main()