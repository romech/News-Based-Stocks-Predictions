import os
import yaml


class Config:
    def __init__(self, dict_like):
        self.dict_like = dict_like

    @classmethod
    def open(cls, path):
        if not os.path.exists(path):
            path = "config.yml"
        with open(path, "r") as cfg_file:
            return Config(yaml.full_load(cfg_file))

    def __call__(self, name):
        node = self.dict_like
        for prefix in name.split('.'):
            node = node[prefix]

        if isinstance(node, dict):
            return Config(node)
        else:
            return node
