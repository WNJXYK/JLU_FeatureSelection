from Tool import Test
from Algorithm.Lasso import Lasso
# from Algorithm.GDFS import GDFS
from Algorithm.FSFOA import FSFOA
from Algorithm.IFSFOA import IFSFOA
from Algorithm.LassoGA import LGA

if __name__ == "__main__":
    # Test(Lasso, "Logs/Lasso.txt")
    # Test(GDFS, "Logs/GDFS.txt")
    # Test(FSFOA, "Logs/FSFOA.txt")
    # Test(IFSFOA, "Logs/IFSFOA.txt")
    Test(LGA, "Logs/LGA.txt")