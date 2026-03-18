import pandas as pd
import numpy as np

# --- 1. DEFINING THE RESEARCH ENVELOPE ---
# Speed: 15 m/min (Ultra-low) to 350 m/min (Extreme High-speed)
# Feed: 0.04 mm/rev (Super-finish) to 0.3 mm/rev (Heavy Roughing)
# DOC: 0.15 mm (Micro-cut) to 2.5 mm (Deep Roughing)

num_samples = 150
speeds = np.geomspace(15, 350, num_samples) # Log spacing for thermal accuracy
feeds = np.tile(np.linspace(0.04, 0.3, 15), 10)
docs = np.repeat(np.linspace(0.15, 2.5, 10), 15)

# --- 2. PHYSICS-BASED TARGET GENERATOR (Inconel 718 Research Benchmarks) ---
def get_precision_targets(s, f, d):
    # Temperature (T) - High sensitivity to Speed (Inconel low thermal conductivity)
    t = 190 + (16 * s**0.75) + (240 * f**0.45) + (95 * d**0.32)
    
    # Cutting Force (Fy) - The primary power component
    # Specific cutting energy of Inconel 718 is ~3000-4000 N/mm^2
    fy = (2100 * f**0.78 * d**0.95) + (s * 0.12)
    
    # Feed Force (Fx) - Axial component (~45% of Fy)
    fx = fy * (0.42 + (0.05 * (s/350))) # Increases slightly at extreme speeds
    
    # Thrust Force (Fz) - Radial component (~65% of Fy)
    # Fz is highly sensitive to tool nose radius and flank wear
    fz = fy * (0.62 + (0.1 * (s/350)))
    
    # Flank Wear (Vb) - Diamond coated tools stay low until 200m/min
    # At high ends, even diamond coatings delaminate
    vb = (0.000075 * s**1.95) + (0.07 * f) + (0.012 * d)
    
    return [t, fx, fy, fz, vb]

# Create the master dataframe
results = [get_precision_targets(s, f, d) for s, f, d in zip(speeds, feeds, docs)]
df = pd.DataFrame(results, columns=['Temp', 'Fx', 'Fy', 'Fz', 'Vb'])
df['Speed'], df['Feed'], df['DOC'] = speeds, feeds, docs
