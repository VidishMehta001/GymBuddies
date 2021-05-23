import numpy as np


class FourierTransform(object):
    
    def __init__(self, n_frames, Dist):
        self.n_frames = n_frames
        self.Dist = Dist

    def FFT(self):
        # Sample Rate for the frequency
        sample_rate = 1
        
        # Estimate frequency
        freq = 3
        x = np.linspace(0,self.n_frames,sample_rate*self.n_frames, endpoint=False)
        frequencies = x * freq

        # 2 pi radians frequency
        y = np.sin((2*np.pi)*frequencies)
        return x,y

