# from turtle import update
# import pytest

# from src.misc import update_namespace

# def test_update_dict_1():
#     orig = {
#         "model": {
#             "lr": 10,
#             "num_layers": 5,
#             "sequence_len": 5
#         },
#         "data": {
#             "sequence_len": 5
#         }
#     }
#     updates = {"lr": 0.1, "sequence_len":10}

#     new = update_namespace(orig, updates)

#     expected = {
#         "model": {
#             "lr": 0.1,
#             "num_layers": 5,
#             "sequence_len": 10
#         },
#         "data": {
#             "sequence_len": 10
#         }
#     }

#     assert new == expected

# def test_update_dict_2():
#     orig = {
#         "model": {
#             "lr": 10,
#             "num_layers": 5,
#             "sequence_len": 5
#         },
#         "data": {
#             "sequence_len": 5
#         }
#     }
#     updates = {"model": {"sequence_len": 10}, "lr":1}
#     new = update_namespace(orig, updates)

#     expected = {
#         "model": {
#             "lr": 1,
#             "num_layers": 5,
#             "sequence_len": 10
#         },
#         "data": {
#             "sequence_len": 5
#         }
#     }
#     assert new == expected


# def test_update_dict_3():
#     orig = {
#         "model": {
#             "lr": 10,
#             "num_layers": 5,
#             "sequence_len": 5,
#             "a" : 0,
#         },
#         "data": {
#             "a" : 0,
#             "sequence_len": 5
#         }
#     }
#     updates = {"model": {"sequence_len": 10}, "a":1}
#     new = update_namespace(orig, updates)

#     expected = {
#         "model": {
#             "lr": 10,
#             "num_layers": 5,
#             "sequence_len": 10,
#             "a" : 1,
#         },
#         "data": {
#             "a" : 1,
#             "sequence_len": 5
#         }
#     }
#     assert new == expected
