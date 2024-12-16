from ultralytics.engine.model import Model
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationPredictor, ClassificationValidator
from ultralytics.nn.tasks import ClassificationModel


class EffcientNet(Model):

    def __init__(self, ver="v1", ref="b0"):
        # check version
        if ver not in ["v1", "v2"]:
            LOGGER.error("parameter \"ver\" must be rather v1 or v2", exc_info=1)
        else:
            self.ver = ver
        
        # check reference
        if self.ver == "v1" and ref not in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            LOGGER.error("parameter \"ref\" must among b0-7", exc_info=1)
        elif self.ver == "v2" and ref not in ["s", "m", "l"]:
            LOGGER.error("parameter \"ref\" must among \"s\", \"m\" and \"l\"", exc_info=1)
        else:
            self.ref = ref
        
        super().__init__(model=f'efficientnet{self.ver}_{self.ref}.yaml', task='classify', verbose=True)

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for EffcientNet, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the EffcientNet model.
        """
        return {
            "classify": {
                "model": ClassificationModel,
                "predictor": ClassificationPredictor,
                "validator": ClassificationValidator,
                "trainer": ClassificationTrainer,
            }
        }
