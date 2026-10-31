"""Stub implementation for the active-contour (snake) algorithm.

Member 3 must replace the body.  Signature must not change.
"""
from __future__ import annotations

import numpy as np

from core.canny_lines import sobel_gradients
from core.types import SnakeParams, SnakeResult


def run_snake(
    gray: np.ndarray,
    init_pts: list,
    params: SnakeParams,
) -> SnakeResult:
    """Run the active-contour (snake) algorithm on a grayscale image.

    Parameters
    ----------
    gray:      H×W uint8 grayscale image.
    init_pts:  List of (x, y) tuples forming the initial contour.
    params:    SnakeParams with alpha, beta, gamma, max_iter.

    Returns
    -------
    SnakeResult with pts, chain, perimeter, area and history.
    """

    image = gray.copy()
    _, _, mag, _ = sobel_gradients(image)
    mag = normalize(mag)

    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    max_iter = params['max_iter']
    
    contour = init_pts.copy()
    hist = [contour.copy()]
    
    for iter in range(max_iter): # iterate through contours
        moved = 0
        for i in range(len(contour)):  # for each point in contour: calc energy & converge
            p0 = contour[(i-1) % len(contour)]
            p1 = contour[i]
            p2 = contour[(i+1) % len(contour)]
            Candidates = get_candidate(p1, image) # reutrn list of tuples
            pnts = []   # (row, col, Energy)
            
            for cand in Candidates:  # calc Energies for the neigbors & current point 
                pnt = cand + (total_energy(p0, cand, p2, alpha, beta, gamma, mag), )
                pnts.append(pnt)
            
            min_pnt = get_min(pnts)  # returned only (row, col)           
            if min_pnt != p1:
                moved += 1
            contour[i] = min_pnt    # In-place update, move toward min Energy  
        
        hist.append(contour.copy())
        if moved == 0:                  # converged
            break
    
    return SnakeResult(
        pts = contour,
        chain = get_chain_code(contour),
        perimeter = calc_perimeter(contour),
        area = calc_area(contour),
        history = hist  
    )



def total_energy(p0, p1, p2, alpha, beta, gamma, mag):
    e_elstc = elastic_energy(p1, p2)
    e_smooth = smooth_energy(p0, p1, p2)
    e_img = image_energy(mag, p1)
    return alpha * e_elstc + beta * e_smooth + gamma * e_img  


def elastic_energy(p1, p2):
    pass

def smooth_energy(p0, p1, p2):
    pass

def image_energy(mag, pnt):
    r, c = pnt                      # (row, col)
    return -mag[int(r), int(c)]

def calc_perimeter(contour):
    pass

def calc_area(contour):
    pass

def get_chain_code(contour):
    pass

def get_candidate(pnt, image):
    pass

def extract_xy(list):
    return [t[:2] for t in list]

def get_min(E):
    pnt_3d = min(E, key = lambda x: x[2])
    return pnt_3d[:2] # return only (row, col)

def normalize(mag):
    max_mag = mag.max()
    if max_mag > 0:
        return mag / max_mag