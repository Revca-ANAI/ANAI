def unsorted_set(seq):
    placeholder_set = set()
    return [x for x in seq if not (x in placeholder_set or placeholder_set.add(x))]
