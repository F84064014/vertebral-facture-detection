import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='NO_VALUE', help='model.pt path(s)')
    opt = parser.parse_args()
    # print(type(opt.name))

    if opt.weights == 'w1':
        print('w1')
    elif opt.weights == 'w2':
        print('w2')
    elif opt.weights == 'NO_VALUE':
        print('no')


    ys = [98, 102, 154, 133, 190, 175]
    # samples = [
    #     [1,3,5],
    #     [1,3,6],
    #     [1,4,5],
    #     [1,4,6],
    #     [2,3,5],
    #     [2,3,6],
    #     [2,4,5],
    #     [2,4,6],
    # ]

    # Ps = [1/8 for i in range(len(samples))]

    samples = [
        [1,4,6],
        [2,3,6],
        [1,3,5],
    ]

    Ps = [1/4, 1/2, 1/4]

    avg = 0
    std = 0
    ms = []
    for sample in samples:
        a,b,c = sample
        m = (ys[a-1] + ys[b-1] + ys[c-1])/3
        # print(m)
        ms.append(m)

    for i in range(len(Ps)):
        avg += ms[i] * Ps[i]

    for i in range(len(Ps)):
        std += (ms[i] - avg)**2 * Ps[i]
        print((ms[i] - avg)**2)

    print('avg = {}'.format(avg))
    print('std = {}'.format(std))

    # print(sum(ys)/len(ys))