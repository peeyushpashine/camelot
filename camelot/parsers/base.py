# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd
import copy
import numpy as np
import re
from ast import literal_eval
import pickle
from ..utils import get_page_layout, get_text_objects


class BaseParser(object):
    """Defines a base parser.
    """

    def _generate_layout(self, filename, layout_kwargs,jtd_df):
        self.filename = filename
        self.layout_kwargs = layout_kwargs
        self.layout, self.dimensions = get_page_layout(filename, **layout_kwargs)
        self.images = get_text_objects(self.layout, ltype="image")
        self.horizontal_text = get_text_objects(self.layout, ltype="horizontal_text")
        self.vertical_text = get_text_objects(self.layout, ltype="vertical_text")
        self.pdf_width, self.pdf_height = self.dimensions
        self.rootname, __ = os.path.splitext(self.filename)
        
        
        if bool(jtd_df):
            page_num = int(os.path.realpath(self.filename).split("-")[1].split(".")[0])-1
            if len(self.horizontal_text) > len(jtd_df[page_num]):
                self.horizontal_text[len(jtd_df[page_num]):]=[]
            elif len(self.horizontal_text) == 0:
                
                with open(os.path.join(os.path.abspath(os.path.dirname('model_dependency')),"model_dependency/pdfminer_dummy_obj.pickle"), "rb") as input_file:
                    self.horizontal_text = pickle.load(input_file)
                    self.horizontal_text[1:]=[]
                self.horizontal_text.extend([copy.deepcopy(self.horizontal_text[0]) for x in range(len(self.horizontal_text), len(jtd_df[page_num]))])
            else:
                self.horizontal_text.extend([copy.deepcopy(self.horizontal_text[0]) for x in range(len(self.horizontal_text), len(jtd_df[page_num]))])
            for i in range(0, len(jtd_df[page_num])):
                #print("###############################")
                #print(i)
                self.horizontal_text[i].bbox = jtd_df[page_num]["scaled_cords"][i]
                self.horizontal_text[i].x0 = jtd_df[page_num]["scaled_cords"][i][0]
                self.horizontal_text[i].x1 = jtd_df[page_num]["scaled_cords"][i][2]
                self.horizontal_text[i].y0 = jtd_df[page_num]["scaled_cords"][i][1]
                self.horizontal_text[i].y1 = jtd_df[page_num]["scaled_cords"][i][3]
                # temp_obj = copy.deepcopy(self.horizontal_text[i]._objs[0])
                if len(list(jtd_df[page_num]["content"][i])) > len(self.horizontal_text[i]._objs):
                    self.horizontal_text[i]._objs.extend([copy.deepcopy(self.horizontal_text[i]._objs[0]) for x in range(len(self.horizontal_text[i]._objs), len(list(jtd_df[page_num]["content"][i])))])
                else:
                    self.horizontal_text[i]._objs[len(list(jtd_df[page_num]["content"][i])):]=[]

                for j in range(0,len(list(jtd_df[page_num]["content"][i]))):
                    self.horizontal_text[i]._objs[j]._text= jtd_df[page_num]["content"][i][j]

            
            print("Done with page", page_num)
    
