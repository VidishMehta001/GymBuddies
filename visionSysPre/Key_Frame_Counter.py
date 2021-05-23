from data_preprocessing import DataPreprocessing as dp
from dist_calc import PairDistCalc
from CountRep import CountRepitition


# counter_val, key_frame_detected, Feature_map = Key-frame/Counter Module > (int, int, array)

class KeyFrame_Counter_FeatureMap(object):

    def __init__(self, x, y, refx, refy, Dist, Gradient_X, Gradient_Y, Feature_Map, Counter, sampled_frame):
        self.x = x
        self.y = y
        self.refx = refx
        self.refy = refy
        self.Dist = Dist
        self.Gradient_X = Gradient_X
        self.Gradient_Y = Gradient_Y
        self.Feature_Map = Feature_Map
        self.Counter = Counter
        self.sampled_frame = sampled_frame

    def key_frame_counter_fm(self):
        if self.sampled_frame != False:
            # Get Nested Maps for the feature image vector computation
            refxy_pair, curxy_pair = dp.getNestedPair(self.refx, self.refy, self.x, self.y)
            #Calculate Euclidean distance between refx & refy
            BPD = PairDistCalc(self.refx, self.refy, self.x, self.y, refxy_pair, curxy_pair)
            # Append the distance frame with the new distance calculated
            self.Dist.append(BPD.eucDist())
            # Compute the feature map
            FM = BPD.eucDist_FM()
            # Count Computation - using geometric gradient descent algorithms       
            grd_x, grd_y = CountRepitition(self.Dist).gradientRepCounter_2pts()
            # Append the calculated gradients
            self.Gradient_X.append(grd_x)
            self.Gradient_Y.append(grd_y)
            # Save image as a keypoint detected - returns true if key pts
            KeyPt_check = dp.keyPointCheck(self.Gradient_Y)
            # Final count based on Gradient_x and Gradient_y arrays - check for 1 to 0 transitions
            count_x = dp.BinTransitions(self.Gradient_X)
            # Count_Y is more stable
            count_y = dp.BinTransitions(self.Gradient_Y)
            self.Counter = self.Counter + count_y
            if ((KeyPt_check != 0) and (len(FM)!=0) and (self.Counter<5)):
                self.Feature_Map.append(FM)

            return self.Counter, KeyPt_check, self.Feature_Map

        KeyPt_check = 0
        return self.Counter, KeyPt_check, self.Feature_Map


