"""
Miniature test implementation of the config system, and a corresponding test.

The key idea of this config loading system:
- The main ConfigExample class has a BaseDemoCfg which can be either RedDemoCfg or GreenDemoCfg.
- demo_factory attribute of BaseDemoCfg specifies which class to instantiate.
- ConfigExample is first loaded with only the fields from BaseDemoCfg (to avoid unknown field error
  when given e.g. red_field attribute, since BaseDemoCfg does not have that field).
- Then the child class e.g. RedDemoCfg is instantiated, now including the red_field attribute.
- The instantiated RedDemoCfg overwrites the BaseDemoCfg demo attribute of ConfigExample.
"""

from pprint import pprint

from attr import asdict, define
from omegaconf import OmegaConf

from packg import Const
from packg.iotools import dumps_yaml
from typedparser import attrs_from_dict

from entitynet.config.config_factory import load_sub_config


@define(auto_attribs=True, kw_only=True)
class BaseDemoCfg:
    demo_factory: str = None
    demo_name: str = None


@define(auto_attribs=True, kw_only=True)
class GreenDemoCfg(BaseDemoCfg):
    green_field: bool = False


@define(auto_attribs=True, kw_only=True)
class RedDemoCfg(BaseDemoCfg):
    red_field: bool = False


class DemoFactoryC(Const):
    GREEN_DEMO = "green_demo"
    RED_DEMO = "red_demo"


DemoFactoryConfigs = {
    DemoFactoryC.GREEN_DEMO: GreenDemoCfg,
    DemoFactoryC.RED_DEMO: RedDemoCfg,
}


@define(auto_attribs=True, kw_only=True)
class ConfigExample:
    demo: BaseDemoCfg = None
    number: int = 8

    @classmethod
    def from_dict(cls, config_dict: dict, override_dict: dict | None = None) -> "ConfigExample":
        if override_dict is not None:
            config_omegaconf = OmegaConf.create(config_dict)
            updated_omegaconf = OmegaConf.merge(config_omegaconf, override_dict)
            config_dict: dict = OmegaConf.to_container(updated_omegaconf, resolve=True)

        # Extract the demo config and simplify config_dict to match the base class
        config_dict, demo_cfg = load_sub_config(
            config_dict, "demo", "demo_factory", DemoFactoryConfigs, BaseDemoCfg
        )

        # Create main config instance with only base fields in the dict
        config: ConfigExample = attrs_from_dict(ConfigExample, config_dict)

        # Override with the actual demo config
        config.demo = demo_cfg
        return config

    def __repr__(self):
        return f"{dumps_yaml(asdict(self), standard_format=False)}"


example_yaml_demo_dict = {
    "demo": {
        "demo_factory": DemoFactoryC.GREEN_DEMO,
        "demo_name": "test_demo",
        "green_field": True,
    },
    "number": 8,
}


def test_config_demo():
    config = ConfigExample.from_dict(example_yaml_demo_dict, override_dict={"number": 10})
    pprint(config)
    config_demo: GreenDemoCfg = config.demo
    assert config_demo.demo_factory == DemoFactoryC.GREEN_DEMO
    assert config_demo.green_field
    assert config.number == 10
