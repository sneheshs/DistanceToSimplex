from distanceToSimplex import *
import time

global time_taken
time_taken = {}

def test(svecs):
    vecs = np.array(svecs).astype(float)
    distance, projection = distanceToSimplex(vecs[0], vecs[1:])
    return distance, projection

def test_multi_onebyone(svecs):
    vecs = np.array(svecs).astype(float)
    distances = []
    projections = []
    num_of_items = 5
    for i in range(0, num_of_items):
        distance, projection = distanceToSimplex(vecs[i], vecs[num_of_items:])
        distances.append(distance)
        projections.append(projection)
    
    return distances, projections

def test_multi(svecs):
    vecs = np.array(svecs).astype(float)
    points = []
    num_of_items = 5
    for i in range(0, num_of_items):
        points.append(vecs[i])
    distances, projections = distancesToSimplex(points, vecs[num_of_items:])
    return distances, projections

def load_vecs(filename):
    f = open(filename, 'r')
    lines = f.read().split('\n')
    f.close()
    svecs = []
    for line in lines:
        if line.strip() == "":
            continue
        vec = line.split(',')[1].split(' ')[:-1]
        svecs.append(vec)

    return svecs

def test_case1():
    # SANITY TEST
    all_test_cases = []
    all_test_cases.append([[.5,.5,.5],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    all_test_cases.append(load_vecs('test_set1.csv'))
    all_test_cases.append(load_vecs('test_set2.csv'))

    for sv in all_test_cases:
        time_taken['rref'] = 0
        time_taken['frref'] = 0
        time_taken['total'] = 0

        STARTTIME = time.time()
        distance, _ = test(sv)
        print('Simplex Distance', distance)
        time_taken['total'] = time.time()-STARTTIME
        
        # # Uncomment this to display timing values
        print(time_taken)
        # print('RREF % time taken', time_taken['rref']/time_taken['total'])
        # print('FRREF % time taken', time_taken['frref']/time_taken['total'], '\n\n')


def test_case2():
    time1 = []
    time2 = []

    all_test_cases = []
    all_test_cases.append(load_vecs('test_set2.csv'))

    for sv in all_test_cases:
        time_taken['rref'] = 0
        time_taken['frref'] = 0
        time_taken['total'] = 0

        STARTTIME = time.time()
        distances_one, _ = test_multi_onebyone(sv)
        #print('Simplex Distance One by One', distances_one)
        time_taken['total'] = time.time()-STARTTIME
        
        time1.append(time_taken['total'])


    for sv in all_test_cases:
        time_taken['rref'] = 0
        time_taken['frref'] = 0
        time_taken['total'] = 0

        STARTTIME = time.time()
        distances_multi, _ = test_multi(sv)
        #print('Simplex Distance Multi', distances_multi)
        time_taken['total'] = time.time()-STARTTIME
        
        time2.append(time_taken['total'])

    return np.mean(time1), np.mean(time2), np.array(distances_multi)-np.array(distances_one)


t1 = []
t2 = []
for repeat in range(0, 100):
    _t1, _t2, _error = test_case2()
    t1.append(_t1)
    t2.append(_t2)

print(np.mean(t1), np.mean(t2), _error)

