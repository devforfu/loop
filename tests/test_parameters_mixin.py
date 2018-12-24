import pytest

from loop.mixins import ParametersMixin


def test_parameters_mixin_attach_set_get_methods(mock):
    assert hasattr(mock, 'set_params')
    assert hasattr(mock, 'get_params')


def test_get_params_method_returns_init_parameters(mock):
    assert mock.get_params() == {'number': 1, 'string': 'value', 'flag': True}


def test_set_params_overrides_object_parameters(mock):
    mock.set_params(number=0, string='', flag=False)

    assert mock.number == 0
    assert not mock.string
    assert not mock.flag


def test_setting_invalid_parameter_raises_value_exception(mock):
    with pytest.raises(ValueError):
        mock.set_params(unknown='unknown')


def test_clean_params_method_removes_irrelevant_parameters_from_dictionary(mock):
    params = {'first': 1, 'second': 2, 'flag': True}

    updated = mock.clean_params(params)

    assert updated == {'flag': True}


@pytest.fixture()
def mock():
    return MockClass()


class MockClass(ParametersMixin):

    def __init__(self, number=1, string='value', flag=True):
        self.number = number
        self.string = string
        self.flag = flag
