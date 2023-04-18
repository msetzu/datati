from typing import List

from models import Modeler


class Pipeline(List[Modeler]):
    """Apply a sequence of transformation to a Dataset"""
    def __init__(self, *transformations):
        super().__init__(list(transformations))

    def __call__(self, *args, **kwargs):
        dataset = args[0]

        for transformation in self:
            dataset = transformation.process(dataset, **kwargs)

        return dataset
