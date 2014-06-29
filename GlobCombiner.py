__author__ = 'davidnola'

import glob
import cPickle


def begin():
    results = []
    names = []
    for f in glob.glob('Results/*.pkl'):
        pkl = cPickle.load(open(f, 'rb'))
        for p, v in pkl:
            (name, result) = (p, v)
            results.append(result)
            names.append(name)


    final_text = ""
    riter = iter(results)

    for name in names:
        next = str(round(riter.next(), 5))
        final_text += name + "," + next + ", " + next + "\n"


    print final_text

    f = open('DistributedSubmitSingle.csv', 'a')
    f.write(final_text)
    f.close()