"""Stub implementation for the active-contour (snake) algorithm.

Member 3 must replace the body.  Signature must not change.
"""
from __future__ import annotations

import numpy as np
from math import sqrt

from core.canny_lines import sobel_gradients
from scipy.ndimage import gaussian_filter
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
    mag = gaussian_filter(mag.astype(float), sigma=8)   # extra blur for noise reduction: Sobel's 3x3 smoothing is too small to reach distant points
    mag = normalize(mag)

    contour = init_pts.copy()
    hist = [contour.copy()]
    
    alpha, beta = scale_weights(contour, params.alpha, params.beta)
    gamma = params.gamma
    max_iter = params.max_iter
    
    n = len(contour)
    d2 = sum(hypot(contour[i], contour[(i+1) % n])**2 for i in range(n)) / n   # mean squared spacing
    alpha = params.alpha / d2
    beta = params.beta / d2
    
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
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2

def smooth_energy(p0, p1, p2):
    return (p2[0] - 2*p1[0] + p0[0])**2 + (p2[1] - 2*p1[1] + p0[1])**2

def image_energy(mag, pnt):
    r, c = pnt[1], pnt[0]             # pnt is (x, y): row = y, col = x
    return -mag[int(r), int(c)]

def calc_perimeter(contour):
    n = len(contour)
    perimeter_pixel = 0.0
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i+1) % n]
        perimeter_pixel +=  hypot(p1, p2)
    return (convert_pixel_cm(perimeter_pixel), perimeter_pixel )

def calc_area(contour):
    n = len(contour)
    s = 0.0
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i+1) % n]
        s += (p1[0] * p2[1]) - (p2[0] * p1[1]) 
        area_pixel = abs(s)/2
    return (convert_pixel_cm(area_pixel), area_pixel)

def convert_pixel_cm(value):
    cm_per_px = 0.05
    return value * cm_per_px

def get_chain_code(contour):
    # (dx, dy) -> Freeman code; y grows downward in images
    DIRS = {( 1,  0): 0, ( 1, -1): 1, ( 0, -1): 2, (-1, -1): 3,
            (-1,  0): 4, (-1,  1): 5, ( 0,  1): 6, ( 1,  1): 7}

    n = len(contour)
    chain = []
    for i in range(n):
        x1, y1 = int(contour[i][0]), int(contour[i][1])
        x2, y2 = int(contour[(i+1) % n][0]), int(contour[(i+1) % n][1])

        # Bresenham from (x1, y1) to (x2, y2); record each 1-pixel step
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        err = dx - dy
        x, y = x1, y1
        while (x, y) != (x2, y2):
            e2 = 2 * err
            step_x = step_y = 0
            if e2 > -dy:
                err -= dy
                x += sx
                step_x = sx
            if e2 < dx:
                err += dx
                y += sy
                step_y = sy
            chain.append(DIRS[(step_x, step_y)])
    return chain

def get_candidate(pnt, image):
    h, w = image.shape[:2]
    x, y = int(pnt[0]), int(pnt[1])

    candidates = [(x, y)]                       # current point first
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue                        # already added
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:     # x vs width, y vs height
                candidates.append((nx, ny))
    return candidates

def get_min(E):
    pnt_3d = min(E, key = lambda x: x[2])
    return pnt_3d[:2] # return only (row, col)

def normalize(mag):
    max_mag = mag.max()
    if max_mag == 0:
        raise ZeroDivisionError("Gradient magnitude is all zeros (image has no edges)")
    return mag / max_mag

def hypot(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def scale_weights(contour, alpha, beta):
    """Divide alpha and beta by the mean squared point spacing, so they are scale-free."""
    n = len(contour)
    d2 = sum(hypot(contour[i], contour[(i+1) % n])**2 for i in range(n)) / n
    if d2 == 0:                      # all points identical: avoid division by zero
        return alpha, beta
    return alpha / d2, beta / d2