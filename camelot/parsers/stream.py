# -*- coding: utf-8 -*-

from __future__ import division
import os, re
import logging
import warnings

import numpy as np
import pandas as pd

from .base import BaseParser
from ..core import TextEdges, Table
from ..utils import text_in_bbox, get_table_index, compute_accuracy, compute_whitespace


logger = logging.getLogger("camelot")


class Stream(BaseParser):
    """Stream method of parsing looks for spaces between text
    to parse the table.

    If you want to specify columns when specifying multiple table
    areas, make sure that the length of both lists are equal.

    Parameters
    ----------
    table_regions : list, optional (default: None)
        List of page regions that may contain tables of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    table_areas : list, optional (default: None)
        List of table area strings of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    columns : list, optional (default: None)
        List of column x-coordinates strings where the coordinates
        are comma-separated.
    split_text : bool, optional (default: False)
        Split text that spans across multiple cells.
    flag_size : bool, optional (default: False)
        Flag text based on font size. Useful to detect
        super/subscripts. Adds <s></s> around flagged text.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.
    edge_tol : int, optional (default: 50)
        Tolerance parameter for extending textedges vertically.
    row_tol : int, optional (default: 2)
        Tolerance parameter used to combine text vertically,
        to generate rows.
    column_tol : int, optional (default: 0)
        Tolerance parameter used to combine text horizontally,
        to generate columns.

    """

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        columns=None,
        split_text=False,
        flag_size=False,
        strip_text="",
        edge_tol=50,
        row_tol=2,
        column_tol=0,
        **kwargs
    ):
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.columns = columns
        self._validate_columns()
        self.split_text = split_text
        self.flag_size = flag_size
        self.strip_text = strip_text
        self.edge_tol = edge_tol
        self.row_tol = row_tol
        self.column_tol = column_tol
        self.keywords = kwargs['keywords']

    @staticmethod
    def _text_bbox(t_bbox):
        """Returns bounding box for the text present on a page.

        Parameters
        ----------
        t_bbox : dict
            Dict with two keys 'horizontal' and 'vertical' with lists of
            LTTextLineHorizontals and LTTextLineVerticals respectively.

        Returns
        -------
        text_bbox : tuple
            Tuple (x0, y0, x1, y1) in pdf coordinate space.

        """
        xmin = min([t.x0 for direction in t_bbox for t in t_bbox[direction]])
        ymin = min([t.y0 for direction in t_bbox for t in t_bbox[direction]])
        xmax = max([t.x1 for direction in t_bbox for t in t_bbox[direction]])
        ymax = max([t.y1 for direction in t_bbox for t in t_bbox[direction]])
        text_bbox = (xmin, ymin, xmax, ymax)
        return text_bbox

    @staticmethod
    def _group_rows(text, keywords, row_tol=2):
        """Groups PDFMiner text objects into rows vertically
        within a tolerance.

        Parameters
        ----------
        text : list
            List of PDFMiner text objects.
        row_tol : int, optional (default: 2)

        Returns
        -------
        rows : list
            Two-dimensional list of text objects grouped into rows.

        """
        row_y = 0
        rows = []
        temp = []
        keyword_matched = False
        keyword_pattern = '|'.join(r"\b{}\b".format(keyword) for keyword in keywords)
        for t in text:
            # is checking for upright necessary?
            # if t.get_text().strip() and all([obj.upright for obj in t._objs if
            # type(obj) is LTChar]):
            line_text = t.get_text().strip()
            if not keyword_matched:
                keyword_matched = True if  re.findall(keyword_pattern, line_text.lower(), re.IGNORECASE) else False

            if line_text:
                if not np.isclose(row_y, t.y0, atol=row_tol):
                    rows.append(sorted(temp, key=lambda t: t.x0))
                    temp = []
                    row_y = t.y0
                temp.append(t)
        rows.append(sorted(temp, key=lambda t: t.x0))
        __ = rows.pop(0)  # TODO: hacky
        return rows, keyword_matched

    @staticmethod
    def _merge_columns(l, column_tol=0):
        """Merges column boundaries horizontally if they overlap
        or lie within a tolerance.

        Parameters
        ----------
        l : list
            List of column x-coordinate tuples.
        column_tol : int, optional (default: 0)

        Returns
        -------
        merged : list
            List of merged column x-coordinate tuples.

        """
        merged = []
        for higher in l:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                if column_tol >= 0:
                    if higher[0] <= lower[1] or np.isclose(
                        higher[0], lower[1], atol=column_tol
                    ):
                        upper_bound = max(lower[1], higher[1])
                        lower_bound = min(lower[0], higher[0])
                        merged[-1] = (lower_bound, upper_bound)
                    else:
                        merged.append(higher)
                elif column_tol < 0:
                    if higher[0] <= lower[1]:
                        if np.isclose(higher[0], lower[1], atol=abs(column_tol)):
                            merged.append(higher)
                        else:
                            upper_bound = max(lower[1], higher[1])
                            lower_bound = min(lower[0], higher[0])
                            merged[-1] = (lower_bound, upper_bound)
                    else:
                        merged.append(higher)
        return merged

    @staticmethod
    def _join_rows(rows_grouped, text_y_max, text_y_min):
        """Makes row coordinates continuous.

        Parameters
        ----------
        rows_grouped : list
            Two-dimensional list of text objects grouped into rows.
        text_y_max : int
        text_y_min : int

        Returns
        -------
        rows : list
            List of continuous row y-coordinate tuples.

        """
        row_mids = [
            sum([(t.y0 + t.y1) / 2 for t in r]) / len(r) if len(r) > 0 else 0
            for r in rows_grouped
        ]
        rows = [(row_mids[i] + row_mids[i - 1]) / 2 for i in range(1, len(row_mids))]
        rows.insert(0, text_y_max)
        rows.append(text_y_min)
        rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]
        return rows

    @staticmethod
    def _add_columns(cols, text,keywords, row_tol):
        """Adds columns to existing list by taking into account
        the text that lies outside the current column x-coordinates.

        Parameters
        ----------
        cols : list
            List of column x-coordinate tuples.
        text : list
            List of PDFMiner text objects.
        ytol : int

        Returns
        -------
        cols : list
            Updated list of column x-coordinate tuples.

        """
        keyword_matched = False
        if text:
            text, keyword_matched = Stream._group_rows(text, keywords, row_tol=row_tol)
            elements = [len(r) for r in text]
            new_cols = [
                (t.x0, t.x1) for r in text if len(r) == max(elements) for t in r
            ]
            cols.extend(Stream._merge_columns(sorted(new_cols)))
        return cols, keyword_matched

    @staticmethod
    def _join_columns(cols, text_x_min, text_x_max):
        """Makes column coordinates continuous.

        Parameters
        ----------
        cols : list
            List of column x-coordinate tuples.
        text_x_min : int
        text_y_max : int

        Returns
        -------
        cols : list
            Updated list of column x-coordinate tuples.

        """
        cols = sorted(cols)
        cols = [(cols[i][0] + cols[i - 1][1]) / 2 for i in range(1, len(cols))]
        cols.insert(0, text_x_min)
        cols.append(text_x_max)
        cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        return cols

    def _validate_columns(self):
        if self.table_areas is not None and self.columns is not None:
            if len(self.table_areas) != len(self.columns):
                raise ValueError("Length of table_areas and columns" " should be equal")

    def _nurminen_table_detection(self, textlines):
        """A general implementation of the table detection algorithm
        described by Anssi Nurminen's master's thesis.
        Link: https://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

        Assumes that tables are situated relatively far apart
        vertically.
        """
        # TODO: add support for arabic text #141
        # sort textlines in reading order
        textlines.sort(key=lambda x: (-x.y0, x.x0))
        textedges = TextEdges(edge_tol=self.edge_tol)
        # generate left, middle and right textedges
        textedges.generate(textlines)
        # select relevant edges
        relevant_textedges = textedges.get_relevant()
        self.textedges.extend(relevant_textedges)
        # guess table areas using textlines and relevant edges
        table_bbox = textedges.get_table_areas(textlines, relevant_textedges)
        # treat whole page as table area if no table areas found
        if not len(table_bbox):
            table_bbox = {(0, 0, self.pdf_width, self.pdf_height): None}

        return table_bbox

    def _generate_table_bbox(self):
        self.textedges = []
        if self.table_areas is None:
            hor_text = self.horizontal_text
            if self.table_regions is not None:
                # filter horizontal text
                hor_text = []
                for region in self.table_regions:
                    x1, y1, x2, y2 = region.split(",")
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    region_text = text_in_bbox((x1, y2, x2, y1), self.horizontal_text)
                    hor_text.extend(region_text)
            # find tables based on nurminen's detection algorithm
            table_bbox = self._nurminen_table_detection(hor_text)
        else:
            table_bbox = {}
            for area in self.table_areas:
                x1, y1, x2, y2 = area.split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                table_bbox[(x1, y2, x2, y1)] = None
        self.table_bbox = table_bbox

    def _generate_columns_and_rows(self, table_idx, tk,h_and_f):
        # select elements which lie within table_bbox
        t_bbox = {}
        t_bbox["horizontal"] = text_in_bbox(tk, self.horizontal_text)
        t_bbox["vertical"] = text_in_bbox(tk, self.vertical_text)

        t_bbox["horizontal"].sort(key=lambda x: (-x.y0, x.x0))
        t_bbox["vertical"].sort(key=lambda x: (x.x0, -x.y0))

        if h_and_f is True:
            return t_bbox["horizontal"]

        self.t_bbox = t_bbox

        text_x_min, text_y_min, text_x_max, text_y_max = self._text_bbox(self.t_bbox)
        rows_grouped, keyword_processing_row = self._group_rows(self.t_bbox["horizontal"],self.keywords, row_tol=self.row_tol)
        rows = self._join_rows(rows_grouped, text_y_max, text_y_min)
        elements = [len(r) for r in rows_grouped]

        if self.columns is not None and self.columns[table_idx] != "":
            # user has to input boundary columns too
            # take (0, pdf_width) by default
            # similar to else condition
            # len can't be 1
            cols = self.columns[table_idx].split(",")
            cols = [float(c) for c in cols]
            cols.insert(0, text_x_min)
            cols.append(text_x_max)
            cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        else:
            # calculate mode of the list of number of elements in
            # each row to guess the number of columns
            ncols = max(set(elements), key=elements.count)
            if ncols == 1:
                # if mode is 1, the page usually contains not tables
                # but there can be cases where the list can be skewed,
                # try to remove all 1s from list in this case and
                # see if the list contains elements, if yes, then use
                # the mode after removing 1s
                elements = list(filter(lambda x: x != 1, elements))
                if len(elements):
                    ncols = max(set(elements), key=elements.count)
                else:
                    warnings.warn(
                        "No tables found in table area {}".format(table_idx + 1)
                    )
            cols = [(t.x0, t.x1) for r in rows_grouped if len(r) == ncols for t in r]
            cols = self._merge_columns(sorted(cols), column_tol=self.column_tol)
            inner_text = []
            for i in range(1, len(cols)):
                left = cols[i - 1][1]
                right = cols[i][0]
                inner_text.extend(
                    [
                        t
                        for direction in self.t_bbox
                        for t in self.t_bbox[direction]
                        if t.x0 > left and t.x1 < right
                    ]
                )
            outer_text = [
                t
                for direction in self.t_bbox
                for t in self.t_bbox[direction]
                if t.x0 > cols[-1][1] or t.x1 < cols[0][0]
            ]
            inner_text.extend(outer_text)
            cols,keyword_processing_col = self._add_columns(cols, inner_text,self.keywords , self.row_tol)
            cols = self._join_columns(cols, text_x_min, text_x_max)
            keyword_matched = keyword_processing_row or keyword_processing_col
        return cols, rows, keyword_matched

    def _generate_table(self, table_idx, cols, rows,keyword_matched,row_postprocessing,col_postprocessing,row_align, **kwargs):
        table = Table(cols, rows)
        table = table.set_all_edges()

        pos_errors = []
        # TODO: have a single list in place of two directional ones?
        # sorted on x-coordinate based on reading order i.e. LTR or RTL
        for direction in ["vertical", "horizontal"]:
            for t in self.t_bbox[direction]:
                try:
                    indices, error = get_table_index(
                        table,
                        t,
                        direction,
                        split_text=self.split_text,
                        flag_size=self.flag_size,
                        strip_text=self.strip_text,
                    )
                    if indices[:2] != (-1, -1):
                        pos_errors.append(error)
                        for r_idx, c_idx, text in indices:
                            table.cells[r_idx][c_idx].text = text
                except Exception as e:
                    print(e)
        accuracy = compute_accuracy([[100, pos_errors]])

        data = table.data
        #####  ALL POST PROCESSING CODE BEGIN #######
        ### ROW POST PROCESSING CODE START ###
        # column density approach to decide on doing postprocessing or  not.
        elmcounter = []
        for elm in data:
            flag=1
            for i in range(0,len(elm)):
                if elm[i] is "":
                    continue
                else:
                    break
            strt = i 
            for jj in range(strt+1,len(elm)):
                if elm[jj] is "":
                    flag=1
                    continue
                else:
                    flag=0
                    break
            elmcounter.append(1-flag)
         # Histogram std deviation approach to decide on doing postprocessing or not. 

        dist_list_all =[]
        for j in range(0, len(data[0])):
            for i in range(0, len(data)):
                temp1 = data[i][j]
                if temp1 is not "":
                    for k in range(i+1,len(data)):
                        temp2 = data[k][j]
                        if temp2 is not "":
                            dist_all = rows[i][0]-rows[k][0] + rows[i][1]-rows[k][1]
                            dist_list_all.append(dist_all)
                            break
                        else:
                            continue
                else:
                    continue
        std_var = np.std(dist_list_all)
 
        if row_postprocessing:
            if elmcounter.count(1)/len(elmcounter) >=0.5:
                row_postprocessing = False
                if std_var < 30:
                    row_postprocessing = False
                else:
                    row_postprocessing = True
            if keyword_matched:
                row_postprocessing = False
        

        if row_postprocessing:
            dist_list=[]
            # find difference between successive rows
            for ii,jj in enumerate(rows[:-1]):
                dist = rows[ii][0]-rows[ii+1][0] + rows[ii][1]-rows[ii+1][1]
                dist_list.append(dist)

            # dist_list.append(0.0)
            inds=[]
            # find the row index for which distance after that row and before that row are lesser than it based on distlist
            for i in range(1, len(dist_list) - 1, 1): 
                if (dist_list[i] > dist_list[i - 1] and 
                    dist_list[i] > dist_list[i + 1]): 
                    inds.append(i)

            # result = len(np.diff(sorted(inds))) > 0 and all(elem == np.diff(sorted(inds))[0] for elem in np.diff(sorted(inds)))
            for id, ind in enumerate(inds):
                if id==0:
                    data1 = data[0:ind+1]
                    data.insert(0,list(map(' '.join, zip(*data1))))
                    for i in range(1,ind+1):
                        data[i]=['']*len(data[0])
                else:
                    data1 = data[inds[id-1]+1:ind+1]
                    data.insert(inds[id-1]+1,list(map(' '.join, zip(*data1))))
                    for i in range(inds[id-1]+2,ind+2):
                        data[i]=['']*len(data[0])        
                data.remove(data[ind+1])

                if id == len(inds)-1:
                    if ind+1 <= len(data):
                        data2 = data[ind+1:]
                        data.insert(ind+1,list(map(' '.join, zip(*data2))))
                        data[ind+2:]=[]
                        break
            table.df = pd.DataFrame(data)
            '''inds = sorted(range(len(dist_list)), key = lambda sub: dist_list[sub])[-RowBreaks:]
                check if rows are homogenous, equispaces due to gaps or not
            result = len(np.diff(sorted(inds))) > 0 and all(elem == np.diff(sorted(inds))[0] for elem in np.diff(sorted(inds)))
            for id, (ind, Numlines) in enumerate(zip(inds,NumOfLines)):
                if result is True:
                    data1 = data[ind-Numlines+1:ind+1]
                    data.insert(ind-Numlines+1,list(map(' '.join, zip(*data1))))
                    for i in range(ind-Numlines+2,ind+2):
                        data[i]=['']*len(data[0])
                    data.remove(data[ind+1])
                else:
                    data1 = data[ind-Numlines-1:ind+1]
                    data.insert(ind-Numlines-1,list(map(' '.join, zip(*data1))))
                    for i in range(ind-Numlines,ind+1):
                        data[i]=['']*len(data[0])
                    data.remove(data[ind+1])
                if id == len(inds)-1:
                        if ind+Numlines <= len(data)-2:
                        data2 = data[ind+1:ind+1+Numlines+1]
                        data.insert(ind+1,list(map(' '.join, zip(*data2))))
                        data[ind+2:ind+2+Numlines+1]=[]
                        break '''
        nan_value = float("NaN")
         ### ROW POST PROCESSING CODE END ###
         ### ROW ALIGN POST PROCESSING CODE START ###
        if row_align:
            df = pd.DataFrame(data).replace("", nan_value).dropna(axis=0,how='all')
            df.replace(np.nan, "",inplace=True)
            df.reset_index(inplace=True)
            df.drop("index",axis=1,inplace=True)
            col_elmInd = []
            for col in df.columns:
                ind = []
                for i in range(0, len(df)):
                    if df.iloc[i][col] is not "":
                        ind.append(i)
                col_elmInd.append(ind)

            for k,sublist in enumerate(col_elmInd[:-1]):
                print(k, sublist)
                ind = []
                for i in range(0, len(df)):
                    if df.iloc[i][k] is not "":
                        ind.append(i)
                ind.append(0)
                sublist = ind
                col_elmInd[k] = sublist
                last_indices = [x[-1] for x in col_elmInd]
                sec_large = sorted(last_indices)[-2]
                first_large = max(last_indices)
                for cur , nxt  in zip(sublist, sublist[1:]):     
                    if nxt != sublist[-1] :
                        arr = df.iloc[cur:nxt].values
                    elif k==len(col_elmInd)-1:
                        break
                    else:
                        if k==0:
                            nxt = cur+1
                            arr = df.iloc[cur:nxt].values
                        else:
                            if cur == sec_large:
                                nxt = first_large
                                arr = df.iloc[cur:nxt].values
                            else:
                                for val in col_elmInd[k-1]:
                                    if val >= cur:
                                        break
                                nxt = val-1
                                if nxt > cur:
                                    arr = df.iloc[cur:nxt].values
                                else:
                                    break
                    for i in range(0,arr.shape[1]):
                        tempstr=""
                        for j in range(0, arr.shape[0]):
                            if arr[j][i] !="":
                                arr[j][i] =arr[j][i]+" "
                            tempstr = tempstr+ arr[j][i]
                        try:
                            arr[0][i] = tempstr
                        except:
                            continue
                    for i in range(1,arr.shape[0]):
                        for j in range(0, arr.shape[1]):
                            arr[i][j]=""

                    df.iloc[cur:nxt] =arr
            table.df = df
        ### ROW ALIGN POST PROCESSING CODE END ###
        ### COL POST PROCESSING CODE START ###
        if col_postprocessing:
            if row_align:
                table.df.replace("", nan_value).dropna(axis=0,how='all')
            col_proc = {}
            for col_ind in table.df.columns.values:
                for row_ind in range(0,len(table.df)):
                    templist  = table.df[col_ind].iloc[row_ind:].values.tolist()
                    # check if all is NaN in the list or not
                    if not all(i != i for i in templist):
                        continue
                    else:
                        try:
                            if row_ind != len(table.df)-1:
                                nextColList = table.df[col_ind+1].iloc[0:row_ind].values.tolist()
                                if all(i != i for i in nextColList):
                                    col_proc[col_ind] = row_ind
                                    break
                        except:
                            continue
            table.df = table.df.replace(np.nan, '', regex=True)
            for k,v in col_proc.items():
                table.df[k] = table.df[k]+table.df[k+1]
                del table.df[k+1]

        ### COL POST PROCESSING CODE END ###
        #####  ALL POST PROCESSING CODE ENDS #######
        if isinstance(table.df, type(None)):
             table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        whitespace = compute_whitespace(data)
        table.flavor = "stream"
        table.accuracy = accuracy
        table.whitespace = whitespace
        table.order = table_idx + 1
        table.page = int(os.path.basename(self.rootname).replace("page-", ""))

        # for plotting
        _text = []
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
        table._text = _text
        table._image = None
        table._segments = None
        table._textedges = self.textedges

        return table

    def extract_tables(self, filename, jtd_df, keywords, row_postprocessing,col_postprocessing,row_align,h_and_f,suppress_stdout=False, layout_kwargs={}):
        self._generate_layout(filename, layout_kwargs, jtd_df)
        if not suppress_stdout:
            logger.info("Processing {}".format(os.path.basename(self.rootname)))

        if not self.horizontal_text:
            if self.images:
                warnings.warn(
                    "{} is image-based, camelot only works on"
                    " text-based pages.".format(os.path.basename(self.rootname))
                )
            else:
                warnings.warn(
                    "No tables found on {}".format(os.path.basename(self.rootname))
                )
            return []

        self._generate_table_bbox()

        _tables = []
        text_h_f = []
        # sort tables based on y-coord
        for table_idx, tk in enumerate(
            sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            if h_and_f is True:
                text = self._generate_columns_and_rows(table_idx, tk,h_and_f)
                text_h_f.append(text)
            else:
                cols, rows,keyword_matched = self._generate_columns_and_rows(table_idx, tk,h_and_f)
                table = self._generate_table(table_idx, cols, rows,keyword_matched,row_postprocessing,col_postprocessing,row_align)
                table._bbox = tk
                _tables.append(table)
        
        if h_and_f is True:
            return text_h_f
        

        return _tables
