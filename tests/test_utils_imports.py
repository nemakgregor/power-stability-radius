def test_utils_submodule_import_works():
    # `stability_radius/utils.py` must not break submodule imports.
    from stability_radius.utils import setup_logging
    from stability_radius.utils.download import download_ieee_case
