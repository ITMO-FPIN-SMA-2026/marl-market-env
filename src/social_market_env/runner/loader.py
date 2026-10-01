"""Import pluggable components by 'package.module:ClassName' strings."""

import importlib
import inspect
from typing import Any

from social_market_env.config import ComponentSpec, ConfigError


def import_object(path: str) -> Any:
    """Import the object named by 'package.module:Name'."""
    module_name, _, attribute = path.partition(':')
    if not module_name or not attribute:
        raise ConfigError(f'Import path must look like package.module:Name, got {path!r}')
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except ImportError as error:
        raise ConfigError(f'Cannot import module {module_name!r} for {path!r}: {error}') from error
    except AttributeError as error:
        raise ConfigError(f'Module {module_name!r} has no attribute {attribute!r} (from {path!r})') from error


def instantiate(*, spec: ComponentSpec, defaults: dict[str, Any]) -> Any:
    """Create a component from its spec, filling constructor parameters it accepts from defaults."""
    factory = import_object(spec.impl)
    accepted = inspect.signature(factory).parameters
    injected = {name: value for name, value in defaults.items() if name in accepted and name not in spec.kwargs}
    return factory(**injected, **spec.kwargs)
