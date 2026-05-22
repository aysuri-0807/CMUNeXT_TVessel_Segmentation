from research_formal.models.sota_unetpp import build_sota_unetpp


def test_sota_builder_is_callable():
    assert callable(build_sota_unetpp)
