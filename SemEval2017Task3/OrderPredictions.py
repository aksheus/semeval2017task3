import functools
import sys
import os

"""
     USAGE: python OrderPredicitons path/input.predictions  path/output.predictions
"""

def compare(item1,item2):
    if item1[0] == item2[0]:
        x = int(item1[2])
        y = int(item2[2])
        if x<y:
           return -1
        elif x==y:
            return 0
        else:
            return 1
    else:
        if item1[0] > item2[0]:
            return 1
        else:
            return -1

def order_predictions(content):
    return sorted(content,key=functools.cmp_to_key(compare))

if __name__ == '__main__':
    if os.path.isfile(sys.argv[1]):
        content = []
        with open(sys.argv[1]) as inp:
            for line in inp:
                content.append(line)
        seen = set()
        #content = [ tuple(z.split()) for z in content]
        pruned_content = []
        for z in content:
            to_add = tuple(z.split())
            if to_add[1] not in seen:
                pruned_content.append(to_add)
            seen.add(to_add[1])

        proper_content = order_predictions(pruned_content)
        with open(sys.argv[2],'w',encoding='utf-8') as out:
            for column in proper_content:
                out.write('\t'.join(col for col in column))
                out.write('\n')


