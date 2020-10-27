import matplotlib.pyplot as plt
import os


def load_tsv(path):
    def deal_line(line):
        i, j, v = line.split()
        return int(i), int(j), float(v)

    ia = []
    ja = []
    val = []
    m, n, nnz = 0, 0, 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.strip().startswith('%'):
                continue
            if not m:
                m, n, nnz = deal_line(line)
                continue
            i, j, v = deal_line(line)
            ia.append(i)
            ja.append(j)
            val.append(v)
    return m, n, nnz, ia, ja, val


def draw(path):
    fig = plt.figure()
    title = path.split('/')[-1].split('.')[0]
    m, n, nnz, x, y, val = load_tsv(path)
    plt.scatter(x, y, c=val, s=0.1)
    plt.title(title + ': ' + str(nnz / (m * n)))
    plt.savefig('img/%s.png' % title, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    task_ls = os.listdir('neuron1024')
    for i in task_ls:
        draw('neuron1024/' + i)
