from .classifier import Classifier
from .dataset import Dataset, DatasetEntry


# supress sklearn annoying warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
