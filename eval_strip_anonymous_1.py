import strip
import sys

if __name__ == '__main__':
    
    model_path = 'models/anonymous_1_bd_net.h5'
    img_path = str(sys.argv[1])
    
    strip.eval(model_path, img_path)
    
    pass
