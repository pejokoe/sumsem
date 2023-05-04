#!/usr/bin/python3
from functools import reduce
# (a)
# data = list(range(11))
# squares = list(map(lambda x: x**2, range(1, 11)))
# squares2 = list(map(lambda x: x**2, range(1, 21)))
# print(squares)
# print(squares2)

# (b)
# even = list(filter(lambda x: not x%2, range(1, 11)))
# even2 = list(filter(lambda x: not x%2, range(1, 21)))
# print(even)
# print(even2)

# (c)
# print(list(map(lambda x: x**2, filter(lambda x: not x%2, range(1, 11)))))

# (d)

words = ["test", "peter", "jurekk", "einsehrlangeswort"]
# def find_longest_word(data): 
#     return len(reduce(lambda x, y: x if len(x) > len(y) else y, words))
# print(find_longest_word(words))

# (e)
# def filter_long_words(data, n):
#     return list(filter(lambda x: len(x) > n, data))
# print(filter_long_words(words, 4))

# (f)

# d = {"merry":"god", "christmas":"jul", "and":"och", "happy":"gott", "new":"nytt",
# "year":"ar"}
# def translate(card):
#     return ' '.join(list(map(lambda x: d[x] if x in d else x, card)))
                
# print(translate("merry christmas and happy new year".split()))

# (g)

# l1 = []
# for word in words:
#     l1.append(len(word))
# l2 = list(map(lambda x: len(x), words))
# l3 = [len(word) for word in words]
# print(l1, "\n", l2, "\n", l3)

# (h)

def max_in_list(l):
    return reduce(lambda x, y: x if x > y else y, l)

l = [4, 1, 546, 7, 4,243 ,76 , 7534, 521, 19824]
print(max_in_list(l))
max = reduce(lambda x, y: x if x > y else y, l)
print(max)
