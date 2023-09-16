
# joblib is a cross-platform multiprocessing lib.
# evaluate joblib's overhead for using various methods of calling large dataframe.
# conclusion is that joblib's overhead costs nearly nothing when compared to usual serial programming.

import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed

def main():
    
    # Create a nx5 DataFrame and fill with random ints.
    df = pd.DataFrame(np.random.randint(1, 100, (10000000, 5)), columns=list('ABCDE'))
    print(df)
    
    # Define four UDFs. Each one is called for dataframe processing using differing methods.

    # UDF 1: map double the value. one item at a time.
    def map_double_val(x):
        return x * 2

    # UDF 2: Subtract by 10. series at a time.
    def series_subtract_ten(s):
        return s - 10

    # UDF 3: Square the value. column at a time. .015 seconds.
    def col_square_val(col):
        return df[col] ** 2

    # UDF 4: apply sum columns. row at a time.
    def apply_col_sum(r):
        return r['A']+r['B']+r['C']

    t = time.time()
    df['A'].map(map_double_val) # 1.81 seconds.
    print('map_double_val:',time.time()-t)

    t = time.time()
    series_subtract_ten(df['B']) # .015 seconds.
    print('series_subtract_ten:',time.time()-t)

    t = time.time()
    col_square_val('C') # .0 seconds.
    print('col_square_val:',time.time()-t)

    t = time.time()
    df.apply(apply_col_sum,axis='columns') # 99.84 seconds.
    print('apply_col_sum:',time.time()-t)

    # Use joblib's Parallel and delayed to run tasks in parallel
    t = time.time()
    Parallel(n_jobs=1)([delayed(df['A'].map)(map_double_val)]) # 1.81 seconds.
    print('Parallel: map_double_val:',time.time()-t)

    t = time.time()
    Parallel(n_jobs=1)([delayed(series_subtract_ten)(df['B'])]) # .015 seconds.
    print('Parallel: series_subtract_ten:',time.time()-t)

    t = time.time()
    Parallel(n_jobs=1)([delayed(col_square_val)('C')]) # .015 seconds.
    print('Parallel: col_square_val:',time.time()-t)

    t = time.time()
    Parallel(n_jobs=1)([delayed(df.apply)(apply_col_sum,axis='columns')]) # 106 seconds.
    print('Parallel: apply_col_sum:',time.time()-t)

    t = time.time()
    results = Parallel(n_jobs=4)([delayed(df['A'].map)(map_double_val),delayed(series_subtract_ten)(df['B']),delayed(col_square_val)('C'),delayed(df.apply)(apply_col_sum,axis='columns')])
    print('Parallel: all:',time.time()-t)

    # Reset the DataFrame to its original state for the parallel execution
    df2 = pd.DataFrame()
    df2['A'], df2['B'], df2['C'], df2['D'] = results

    print(df2)

if __name__ == "__main__":
    main()
