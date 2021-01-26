import sys


def minfringe(lat,Q):
	target = Q[0]
	temp_lat = sys.maxsize
	for node_name in Q:
		if lat[node_name] < temp_lat:
			target = node_name
			temp_lat = lat[node_name]

	return target


def dijkstra(nodes, nodes_connected_links, s, t):
	lat = {}
	Q = []
	prev = {}
	for node in nodes:
		lat[node.name] = sys.maxsize
		prev[node.name] = None
		Q.append(node.name)

	lat[s] = 0

	while len(Q) > 0:
		u = minfringe(lat, Q)
		Q.remove(u)
		for to_link, to_node_name in nodes_connected_links[u]:
			if to_node_name in Q:
				if (to_link.lat + lat[u]) < lat[to_node_name]:
					lat[to_node_name] = to_link.lat + lat[u]
					prev[to_node_name] = u

	while prev[t] != s:
		t = prev[t]

	action = 0
	for to_link, to_node_name in nodes_connected_links[s]:
		if to_node_name == t:
			return action
		else:
			action += 1

	return False

