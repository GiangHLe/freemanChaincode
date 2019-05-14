from __future__ import print_function, division
import numpy as np
import math
import cv2
from matplotlib import pyplot as mlt

class FreemanChainCode(object):
    def __init__(self, image_path, **kwargs):
        """
        Required arguments:
        - image_path: the location of image, remember '/' with Linux and '\\' with Windows
        Optional arguments:
        - direction: number of direction of chaincode
        Default is 8
        - object_location: list of tuple elements location of object in image
        Default is [(20,218),(20,218)]
        - threshold: choose a threshold for detect object clearly with environment
        Default is 60
        - kernel_size: the size of kernel use for opening and closing to let picture clear, this kernel with have square shape
        Default is 10
        - morno: choose that you will use closing or opening or none
        Default is 'closing'
        """
        self.path = image_path
        self.direction = kwargs.pop('direction', 8)
        self.object_location = kwargs.pop('location', [(20,218),(20,218)])
        self.thresh = kwargs.pop('threshold', 60)
        self.kernel_size = kwargs.pop('kernel_size', 10)
        self.mor = kwargs.pop('morno', 'closing')

    def _preprocessing(self):
        image = cv2.imread(self.path,0)
        boudary = self.object_location
        image = image[boudary[0][0]:boudary[0][1],boudary[1][0]:boudary[1][1]]
        if image[0,0] == 255:
            ret, img = cv2.threshold(image, self.thresh,255, cv2.THRESH_BINARY_INV)
        else:
            ret, img = cv2.threshold(image, self.thresh,255, cv2.THRESH_BINARY)
        return img
    
    def _start_point(self):
        if self.mor == 'closing':
            img = self._closing()
        if self.mor == 'opening':
            img = self._opening()
        if self.mor == 'None':
            img = self._preprocessing()
        for i, row in enumerate(img):
            for j, value in enumerate(row):
                if value == 255:
                    start_point = (i, j)
                    break
            else:
                continue
            break
        return start_point, img
    def _closing(self):
        kernel = np.ones((self.kernel_size,self.kernel_size),np.uint8)
        img = self._preprocessing()
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closing
    def _opening(self):
        kernel = np.ones((self.kernel_size,self.kernel_size),np.uint8)
        img = self._preprocessing()
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opening
    def _get_chain_code(self):
        '''
        Get chain code by direction
        '''
        start_point, img = self._start_point()
        if self.direction == 4:
            dicrections = [   0,
                           1,    2,
                              3    ]
            num_necess = 3
        elif self.direction == 8:
            directions = [ 0,  1,  2,
                           7,      3,
                           6,  5,  4]
            num_necess = 5
        dir2idx = dict(zip(directions, range(len(directions))))
        move = 1
        # Move the current point follow the direction
        if self.direction == 8:
            # Columns
            change_x =   [-move,  0, move, 
                          -move,     move,
                          -move,  0, move]

            # Rows
            change_y =   [-move,-move,-move,
                           0,             0,
                           move, move, move]
        elif self.direction == 4:
            # Columns
            change_x = [       0,    
                        -move,    move,
                               0       ]
            # Rows
            change_y = [     -move,
                         0,        0,
                              move    ]

        # Border for drawing the boundary when finish the job
        border = []
        chain = []
        curr_point = start_point
        '''
        This find the first shape, the direction we choose to move our chain code
        '''
        for direction in directions:
            idx = dir2idx[direction]
            # Move to each directions to find the true way
            new_point = (start_point[0]+change_y[idx], start_point[1]+change_x[idx])
            if img[new_point] != 0:
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        count = 0
        while(count < 30):
            while curr_point != start_point:
                # Make sure the direction always be the first opposite with the current direction
                b_direction = (direction + num_necess) % self.direction
                if self.direction > 4:
                    b_direction += 1
                dirs_1 = range(b_direction, 8)
                dirs_2 = range(0, b_direction)
                dirs = []
                dirs.extend(dirs_1)
                dirs.extend(dirs_2)
                for direction in dirs:
                    idx = dir2idx[direction]
                    new_point = (curr_point[0]+change_y[idx], curr_point[1]+change_x[idx])
                    if img[new_point] != 0:
                        border.append(new_point)
                        chain.append(direction)
                        curr_point = new_point
                        break
                count += 1
            move+=1
        return border, chain, img
    def _different(self):
        '''
        Return the different from chain code
        '''
        directions = self.direction
        _,chain_code,_ = self._get_chain_code()
        result = []
        dir = list(range(directions))
        len_code = len(chain_code)
        for i in range(len_code):
            step = [dir.index(chain_code[i]), dir.index(chain_code[i-1])]
            if i ==0:
                step = [dir.index(chain_code[len_code-1]), dir.index(chain_code[0])]
            temp = (dir.index(chain_code[i])-dir.index(chain_code[i-1]))
            if temp < 0:
                minimum = min(step)
                maximum = max(step)
                temp = len(dir[:minimum])+len(dir[maximum:])
            result.append(temp)
        return result        
    def result(self):
        '''
        This function give the final result, return the boundary point of object
        '''
        border, chain, img = self._get_chain_code()
        diff = self._different()
        shape = [diff[len(diff)-1]] + diff[:(len(diff)-1)]
        chain_code = (''.join([str(i) for i in chain]))
        differ = (''.join([str(i) for i in diff]))
        shape_no = (''.join([str(i) for i in chain_code]))
        return border, chain_code, differ, shape_no, img
    def show(self):
        '''
        Show the final solution
        '''
        border, chain_code, differ, shape_no, img = self.result()
        print("Our chain code is {}:".format(chain_code))
        print("Different {}:".format(differ))
        print("Shape number {}:".format(shape_no))
        mlt.imshow(img, cmap='Greys')
        mlt.plot([i[1] for i in border], [i[0] for i in border])
