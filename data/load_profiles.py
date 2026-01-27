"""Load profiles (kW) for 24 hours.

Constraints satisfied:
- Bus3 peak: 33 kW at 16:00 (hour index 16)
- Bus4 peak: 50 kW at 20:00 (hour index 20)
"""

import numpy as np

P1_KW = 10.0
P2_KW = 12.0

P3_KW_24 = np.array([3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 31.0, 33.0, 30.0, 26.0, 20.0, 14.0, 10.0, 6.0, 4.0], dtype=float)
P4_KW_24 = np.array([2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 26.0, 30.0, 34.0, 36.0, 40.0, 45.0, 50.0, 42.0, 30.0, 15.0], dtype=float)
