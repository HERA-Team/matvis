import numpy as np

def get_submatrices(pairs, max_subs: int=4, verbose: bool=False):
    """
    Split a full matrix of antenna-pairs into submatrices that contain given pairs.

    Notes
    -----
    This function is not gauranteed to return the optimal decomposition. It does the
    following:

    1. Gets the arrays of ant1 and ant2, and sorts each one (in reverse) in order of
       how many times it appears in the pairs given. Since the pairs given (if 
       representing unique baselines) are not a unique representative set, this is
       is not necessarily optimum.   
    2. Try different numbers of sub-matrices, starting with 1 and going up by one up 
       to ``max_subs``. Each submatrix is limited to placing cuts in the ant1 array. 
       That is, the final result is just a partition of ant1 into Nsub sections.
    3. To decide where these cuts should be, we start with the first cut, try it at one 
       antenna, check the density of unique pairs for that antenna matched with all ant2 
       antennas it requires, and if it's above a certain threshold density, try 
       including the next antenna in ant1, and so on until it crosses the threshold 
       density (starting at 0.8).
    4. Once it hits that threshold, it starts trying to place the next cut in the same 
       way, starting from the antenna its up to.
    5. Once all the cuts are placed, checks the density of the last matrix (which didn't
       get checked when placing the cuts), and if it is below the threshold density, we 
       start again with a lowered threshold.
    
    This obtains N submatrices that all have roughly the same density of desired pairs 
    within the size of the matrices, as high as possible under the constraints of the 
    method. Note that having a roughly uniform density for the sub-matrices is not 
    strictly necessary to achieve optimum *overall* density, but is a reasonably simple
    metric to optimize that correlates with the desired metric.

    Furthermore, the "best" solution is the one that affords the fastest matrix
    multiplication as a sum over all sub-matrices. This is not achieved simply by 
    obtaining the highest density partition, but is also a function of the number of 
    submatrices. 

    Since the method is constrained to slicing up the ant1 array, it is not symmetric.
    One should also try reversing the pairs as an input to see if that yields a better
    solution. In principle, one can arbitrarily reverse individual pairs as well, but
    we do not try that here.

    Parameters
    ----------
    pairs : numpy.ndarray
        A numpy array of shape (n, 2) representing pairs.
    max_subs : int, optional
        The maximum number of sub-matrices to try. Default is 4.
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    Returns
    -------
    antmap : dict
        A dictionary mapping ants to pairs.
    all_antlists : list
        A list of lists of lists representing the ant lists for each sub-matrix.

    Examples
    --------
    >>> pairs = np.array([[1, 2], [2, 3], [3, 4]])
    >>> antmap, all_antlists = get_submatrices(pairs, max_subs=3, verbose=True)
    There are 3 ants to go around
    Trying 2 sub-matrices
      With tol_density=0.80
        Got 0 ants for submatrix 0 with final density 0.000 going to 0.000
      With tol_density=0.77
        Got 1 ants for submatrix 1 with final density 0.000 going to 0.000
    Trying 3 sub-matrices
      With tol_density=0.80
        Got 0 ants for submatrix 0 with final density 0.000 going to 0.000
      With tol_density=0.77
        Got 1 ants for submatrix 1 with final density 0.000 going to 0.000
      With tol_density=0.74
        Got 2 ants for submatrix 2 with final density 0.000 going to 0.000
    ({1: [2], 2: [3], 3: [4]}, [[[1]], [[2]], [[3]]])
    """
    # Here's where the magic happens. 
    ant1, ant1_counts = np.unique(pairs[:, 0], return_counts=True)
    ant2, ant2_counts = np.unique(pairs[:, 1], return_counts=True)

    idx1 = np.argsort(ant1_counts)[::-1]
    idx2 = np.argsort(ant2_counts)[::-1]

    ant1 = ant1[idx1]
    ant2 = ant2[idx2]
    n1 = len(ant1)

    if verbose:
        print(f"There are {n1} ants to go around")
    antmap = {a: [p[1] for p in pairs if p[0]==a] for a in ant1}

    densities = [len(pairs) / (len(ant1)*len(ant2))]

    all_densities = [densities]
    all_antlists = [[ant1.tolist()]]

    for n_subs in range(2, max_subs):  # try different numbers of sub-matrices
        
        if verbose:
            print(f"Trying {n_subs} sub-matrices")

        # the fraction of the area upper triangle to the blocky upper triangle made
        # n_subs equal-height rectangles. This is the highest density we could 
        # possibly get for this n_subs.
        tol_density = 0.8

        while True:
            if verbose:
                print(f"  With tol_density={tol_density:.2f}")

            sub_densities = [0]*n_subs
            antlists = []
        
            curr_ant = 0
            next_densities = []
            for subline in range(n_subs-1):  # play with where each submatrix starts and stops
                cols_included = set()
                pairs_included = []
                antlist = []

                for idx, ant in enumerate(ant1[curr_ant:], start=curr_ant): # try different number of rows
                    cols_included.update(antmap[ant])
                    pairs_included.extend(antmap[ant])
                    #print(len(cols_included), len(pairs_included))
                    this_density = len(pairs_included) / ((idx - curr_ant + 1) * len(cols_included))

                    if this_density < tol_density:
                        if verbose:
                            print(f"    Got {idx-curr_ant} ants for submatrix {subline} with final density {sub_densities[subline]:.3f} going to {this_density}")
                        curr_ant = idx
                        next_densities.append(this_density)
                        break

                    antlist.append(ant)
                    sub_densities[subline] = this_density
                else:
                    # We hit the end of all the ants and never broke the density!
                    antlists.append(antlist)
                    break
                    
                antlists.append(antlist)
            else:
                # The last square also should be above the tolerance.
                pairs_included = sum((antmap[ant] for ant in ant1[curr_ant:]), start=[])
                cols_included = set(pairs_included)
                this_density = len(pairs_included) / ((n1-curr_ant) * len(cols_included))
                sub_densities[-1] = this_density
                antlists.append(ant1[curr_ant:])

                if this_density < tol_density:
                    # not good enough. try with a lower tolerance everywhere.
                    if verbose:
                        print(f"    Final sub-matrix density too low ({this_density:.3f}). Size={n1-curr_ant}x{len(cols_included)} Reducing required density...")
                    tol_density = min(tol_density - 0.03, min(next_densities))
                else:
                    all_densities.append(sub_densities)
                    all_antlists.append(antlists)
                    if verbose:
                        print(f"  Success! Final sub-matrix has {len(ant1)-curr_ant} ants with density {this_density:.3f}")
                    break
            
            if len(antlists) < n_subs:
                # In the case we hit the end without using all our submatrices
                if verbose:
                    print(f"  Success only required {len(antlists)} sub-matrices.")
                all_antlists.append(antlists)
                break

    return antmap, all_antlists