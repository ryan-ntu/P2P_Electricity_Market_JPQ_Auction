"""
Example configuration for generation disruption testing in P2P bidding environment.

This file demonstrates how to configure the generation disruption feature
for robustness testing of the microgrid P2P bidding system.
"""

# Example environment configuration with generation disruption
example_env_config = {
    # Basic environment parameters
    'max_steps': 1000,
    'market_mechanism': 'mrda',
    'common_price': 0.5,
    'emergency_price': 2.0,
    'feed_in_price': 0.3,
    
    # Generation disruption configuration for robustness testing
    'generation_disruption': {
        'enabled': True,  # Enable/disable disruption simulation
        
        # Disruption probability (0.0 to 1.0)
        # Higher values mean more frequent disruptions
        'probability': 0.15,  # 15% chance per agent per hour
        
        # Disruption severity range (0.0 to 1.0)
        # Represents the percentage reduction in generation
        'min_severity': 0.2,  # Minimum 20% reduction
        'max_severity': 0.8,  # Maximum 80% reduction
        
        # Types of disruptions and their probabilities
        'disruption_types': [
            'sudden_drop',      # Immediate significant reduction (most common)
            'gradual_decline',  # Smaller reduction that might persist
            'complete_failure'  # Near-zero generation (rare)
        ],
        'type_weights': [0.6, 0.3, 0.1],  # Probability weights for each type
        
        # Persistent effects configuration
        'persistence_probability': 0.3,  # 30% chance of persistent effects
        'max_persistence_hours': 3,      # Maximum hours for persistent effects
        
        # Logging configuration
        'verbose_logging': True  # Enable detailed logging of disruptions
    }
}

# Conservative disruption configuration (less aggressive)
conservative_config = {
    'generation_disruption': {
        'enabled': True,
        'probability': 0.05,  # 5% chance - less frequent
        'min_severity': 0.1,  # 10% minimum reduction
        'max_severity': 0.5,  # 50% maximum reduction
        'disruption_types': ['sudden_drop', 'gradual_decline'],
        'type_weights': [0.7, 0.3],  # No complete failures
        'persistence_probability': 0.2,
        'max_persistence_hours': 2,
        'verbose_logging': False
    }
}

# Aggressive disruption configuration (more challenging)
aggressive_config = {
    'generation_disruption': {
        'enabled': True,
        'probability': 0.25,  # 25% chance - more frequent
        'min_severity': 0.3,  # 30% minimum reduction
        'max_severity': 0.95, # 95% maximum reduction
        'disruption_types': ['sudden_drop', 'gradual_decline', 'complete_failure'],
        'type_weights': [0.4, 0.4, 0.2],  # More complete failures
        'persistence_probability': 0.5,
        'max_persistence_hours': 5,
        'verbose_logging': True
    }
}

# Disable disruption configuration
disabled_config = {
    'generation_disruption': {
        'enabled': False  # Disable disruption simulation
    }
}

def get_disruption_config(config_type='default'):
    """
    Get a specific disruption configuration.
    
    Args:
        config_type: 'default', 'conservative', 'aggressive', or 'disabled'
        
    Returns:
        dict: Configuration dictionary
    """
    configs = {
        'default': example_env_config['generation_disruption'],
        'conservative': conservative_config['generation_disruption'],
        'aggressive': aggressive_config['generation_disruption'],
        'disabled': disabled_config['generation_disruption']
    }
    
    return configs.get(config_type, configs['default'])

# Usage example:
if __name__ == "__main__":
    # Example of how to use the configuration
    env_config = {
        'max_steps': 1000,
        'market_mechanism': 'mrda',
        'common_price': 0.5,
        'emergency_price': 2.0,
        'feed_in_price': 0.3,
        'generation_disruption': get_disruption_config('aggressive')
    }
    
    print("Example configuration with aggressive disruption:")
    print(env_config['generation_disruption'])
