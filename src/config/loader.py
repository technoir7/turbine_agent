import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found at {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process Env Var Overrides
    # Format: TURBINE_SECTION_KEY (e.g. TURBINE_EXCHANGE_API_KEY)
    for section, values in config.items():
        if isinstance(values, dict):
            for key in values:
                env_var = f"TURBINE_{section.upper()}_{key.upper()}"
                if env_var in os.environ:
                    val = os.environ[env_var]
                    # Simple type casting
                    if isinstance(values[key], bool):
                        config[section][key] = val.lower() == 'true'
                    elif isinstance(values[key], int):
                        config[section][key] = int(val)
                    elif isinstance(values[key], float):
                        config[section][key] = float(val)
                    else:
                        config[section][key] = val
                    logger.info(f"Config: Overrode {section}.{key} from env var")
                    
    return config
