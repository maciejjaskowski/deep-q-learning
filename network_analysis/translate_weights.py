import numpy
import sys

npzfile = numpy.load(sys.argv[1])

f = open(sys.argv[2], 'w')
for key, item in npzfile.iteritems():
    f.write(key + " - " + str(item.shape))
    f.write(str(list(item)))
    f.write("\r\n")

    #numpy.savetxt(key+".txt", item)
f.close()
