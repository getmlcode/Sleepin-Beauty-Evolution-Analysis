import numpy
import random

class web:
    def __init__(self,n):
        self.size = n
        self.in_links = {}
        self.number_out_links = {}
        self.dangling_pages = {}
        for j in xrange(n):
            self.in_links[j] = []
            self.number_out_links[j] = 0
            self.dangling_pages[j] = True

def step(g,p,s=0.85):
    '''Performs a single step in the PageRank computation,
    with web g and parameter s.  Applies the corresponding M
    matrix to the vector p, and returns the resulting
    vector.'''
    n = g.size
    v = numpy.matrix(numpy.zeros((n,1)))
    inner_product = sum([p[j] for j in g.dangling_pages.keys()])
    for j in xrange(n):
        if j % 10000 == 0:
            print j,
        v[j] = s*sum([p[k]/g.number_out_links[k]
                      for k in g.in_links[j]])+s*inner_product/n+(1-s)/n
    # We rescale the return vector, so it remains a
    # probability distribution even with floating point
    # roundoff.
    return v/numpy.sum(v)

def pagerank(g,s=0.85,tolerance=0.0000001):
    '''Returns the PageRank vector for the web g and
    parameter s, where the criterion for convergence is that
    we stop when M^(j+1)P-M^jP has length less than
    tolerance, in l1 norm.'''
    n = g.size
    p = numpy.matrix(numpy.ones((n,1)))/n
    iteration = 1
    change = 2
    while change > tolerance:
        print "Iteration: %s" % iteration
        new_p = step(g,p,s)
        change = numpy.sum(numpy.abs(p-new_p))
        print "\nChange in l1 norm: %s" % change
        p = new_p
        iteration += 1
    return p

if __name__ == '__main__':
    print 'Constructing citation network...'
    citations = open('/home/anubhav/Desktop/Datasets/PubMed_subset-master/paper_paper.txt', 'r')
    output = open('/home/anubhav/Desktop/Datasets/PubMed_subset-master/PaperRanks_PageRank.txt', 'w')
    g = web(99214)
    for line in citations:
        # Here j is citing paper and k is cited paper
        j, k = map(int, line.split('\t'))
        g.number_out_links[j-1] += 1
        g.in_links[k-1].append(j-1)
        g.dangling_pages[j-1] = False
    print '\nComputing PageRank...\n'
    pr = pagerank(g, 0.95, 0.000001)
    rank = numpy.array(map(float, pr))
    idx = rank.argsort()
    ans = "pid\trank\n"
    for i in range(1, 99214):
        ans += str(idx[-i]+1) + "\t" + str(rank[idx[-i]]) + '\n'
    output.write(ans)
    citations.close()
    output.close()
