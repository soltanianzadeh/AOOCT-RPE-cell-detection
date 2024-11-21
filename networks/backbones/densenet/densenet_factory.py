
# *coding:utf-8 *


from .build_densenet import get_densenet_121


_densenet_backbone = {
    'densenet121': get_densenet_121,

}


def get_densenet_backbone(model_name):
    support_models = ['densenet121']
    assert model_name in support_models, "We just support the following models: {}".format(support_models)

    model = _densenet_backbone[model_name]

    return model

if __name__ == '__main__':
    str1 = 'densenet121'
    model = get_densenet_backbone(str1)

