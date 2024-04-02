class DataConfig:
    """
    Configuration class for handling file paths and data-related settings.
    """
    model_path = "../model/3.tflite"


class DetectorConfig:
    """
    Configuration class for handling detector-related settings.
    """
    EDGES = {(0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm',
             (0, 6): 'c', (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c',
             (5, 6): 'y', (5, 11): 'm', (6, 12): 'c', (11, 12): 'y', (11, 13): 'm',
             (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'}
    confidence_threshold = 0.5
