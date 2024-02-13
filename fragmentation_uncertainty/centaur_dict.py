"""Dictionaries containing Centaur catalog data

================================================================================"""

from datetime import datetime

# Event 1 - 2009-047B
Case09047B = {
    'AIUB file': 'fragm09047B_20200302tle.txt',  
    'Vimpel file': 'orbits.20190401_09047B.txt',
    'Vimpel parent file': 'orbits.parent_09047B.txt',
    'breakup time': datetime(2019, 3, 24, 3, 33, 0),
    'satnum': '09047',  # NORAD satellite number
    'stdr_along': 450,  # alongtrack uncertainty [km]
    'stdr_cross': 300,  # crosstrack uncertainty [km]
    'stdr_radial': 280  # radial uncertainty [km]
}

# Event 2 - 2014-055B
Case14055B = {
    'AIUB file': 'fragm14055B_20200302.txt',  
    'spacetrack file': '14055st.txt',         
    'Vimpel file': 'orbits.20180910_14055B.txt',
    'Vimpel parent file': 'orbits.parents_14055B.txt',
    'breakup time': datetime(2018, 8, 30, 22, 4, 0),
    'satnum': '14055',  # NORAD satellite number
    'stdr_along': 400,  # alongtrack uncertainty [km]
    'stdr_cross': 300,  # crosstrack uncertainty [km]
    'stdr_radial': 250  # radial uncertainty [km]
}

# Event 3 - 2018-079B
Case18079B = {
    'AIUB file': 'fragm18079B_20200302.txt',  
    'spacetrack file': '18079st.txt', 
    'Vimpel file': 'orbits.20190415_18079B.txt',        
    'Vimpel parent file': 'orbits.parents_18079B.txt',
    'TLE parent file': 'fragm18079B_20200302.txt',
    'breakup time': datetime(2019, 4, 6, 20, 21, 0),  # (2019, 4, 6, 18, 57, 0),
    'satnum': '18079',  # NORAD satellite number
    'stdr_along': 700,  # alongtrack uncertainty [km]
    'stdr_cross': 600,  # crosstrack uncertainty [km]
    'stdr_radial': 590  # radial uncertainty [km]
}
