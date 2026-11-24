"""Load the static asset and optional fixed annotations for a runtime session."""
from perception.environment.storage import StaticEnvironmentMap


def load_static_map(directory, catalog_path=None):
    if catalog_path is None:
        return StaticEnvironmentMap(directory)

    from decision.agent.adapters.shadow import load_catalog_map
    from decision.agent.tools.places import Catalog

    catalog = Catalog.load(catalog_path)
    catalog.require_complete()
    return load_catalog_map(catalog, directory=directory)
