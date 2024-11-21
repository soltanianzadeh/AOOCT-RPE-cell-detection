from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .densenet.densenet_factory import get_densenet_backbone



def get_backbone_architecture(backbone_name):
    if "densent" in backbone_name:
        return get_densenet_backbone(backbone_name)
