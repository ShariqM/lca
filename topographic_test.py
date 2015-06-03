


sz = 10

def sort(a,b):
    if a < b:
        return (a,b)
    return (b,a)

def acquire(neighbors, i, j, k, l, sz):
    if k == 0 and l == 0:
        return
    ni = i+k
    nj = j+l
    if ni < 0 or ni >= sz:
        return
    if nj < 0 or nj >= sz:
        return
    neighbors.append(sort(i * sz + j, ni * sz + nj))

def get_neighbors(n,i, j, sz):
    neighbors = []
    if n == 4:
        for (k,l) in ((-1,0), (1, 0), (0, -1), (0, 1)):
            acquire(neighbors, i, j, k, l, sz)
    else:
        for k in (-1, 0, 1):
            for l in (-1, 0, 1):
                acquire(neighbors, i, j, k, l, sz)
    return neighbors

for sz in range(1,20):
    pv = set([])
    n = 4
    for i in range(sz):
        for j in range(sz):
            neighbors = get_neighbors(n, i,j,sz)
            pv = pv.union(neighbors)

    print 'N=%d SZ=%d NVars=%d' % (n, sz, len(pv))
    #print 'SZ=%d NVars=%d Vars=%s' % (sz, len(pv), pv)

# On the order of 4 * (n-2) ** 2 (not great...)
