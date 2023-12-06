from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def parallelize(func, it, n_w=None):
    if n_w is None:
        n_w = cpu_count()
    with Pool(n_w) as p:
        for r in tqdm(p.imap_unordered(func, it)):
            print(r)
