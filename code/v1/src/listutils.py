#!/usr/bin/env python3
"""Auxiliary functions to process lists"""

def normalized(vect):
	"""
	Normalize the given vector.
	:param vect: Vector represented as a list of floats.
	:return: Normalized vector represented as a list of floats.
	"""
	factor = 1 / sum(vect)
	return [element * factor for element in vect]


def flatten_lists(lists):
	"""
	Construct a single list from a list of lists.
	:param lists: List of lists.
	:return: Single list that contains all the elements of all the lists, in the same order.
	"""  
	return [element for inner in lists for element in inner]

	
def flatten_state(state):
	"""
	Construct a single list from the state.
	:param lists: State of the MDP.
	:return: Single list that contains all the elements of state.N, state.M and state.R.
	"""
	N_list = flatten_lists(state.N)
	M_list = flatten_lists(state.M)
	R_list = flatten_lists(flatten_lists(state.R))
	state_list = N_list + M_list + R_list
	return state_list

def unflatten_list(lst, dim):
	"""
	Construct a list of lists from a single list.
	:param lst: List of elements, size must be a multiple of dim.
	:param dim: Number of elements in each inner list.
	:return: List of lists that contain all the elements of the list.
	"""
	##print(len(lst))
	##print(dim)
	assert((len(lst) % dim) == 0)
	lists = []
	for i in range(len(lst) // dim):
		lists.append([lst[j] for j in range(i * dim, (i + 1) * dim)])
	return lists