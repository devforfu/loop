from loop.collections import NamedList


def test_named_list_access_with_numerical_indexes():
    nl = NamedList([('a', 1), ('b', 2), ('c', 3)])

    assert nl[0] == 1
    assert nl[1] == 2
    assert nl[2] == 3


def test_named_list_access_with_string_keys():
    nl = NamedList([('a', 1), ('b', 2), ('c', 3)])

    assert nl['a'] == 1
    assert nl['b'] == 2
    assert nl['c'] == 3


def test_named_list_iterable_in_order():
    nl = NamedList([('a', 1), ('b', 2), ('c', 3)])

    items = list(nl)

    assert items == [1, 2, 3]
