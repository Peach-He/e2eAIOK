from pyrecdp.primitives.generators import *
from pyrecdp.core import DataFrameSchema, SparkDataProcessor, DiGraph
from pyrecdp.primitives.operations import Operation, DataFrameOperation, RDDToDataFrameConverter
import pandas as pd
import logging
import graphviz
import json
from pyrecdp.core.utils import Timer, sample_read, deepcopy
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD as SparkRDD
from IPython.display import display

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class BasePipeline:
    def __init__(self, dataset, label, *args, **kwargs):
        if isinstance(dataset, pd.DataFrame):
            self.dataset = {'main_table': dataset}
        elif isinstance(dataset, list):
            self.dataset = dict((idx, data) for idx, data in enumerate(dataset))
        elif isinstance(dataset, dict):
            self.dataset = dataset
        else:
            self.dataset = {'main_table': dataset}
        main_table = None
        input_is_path = False
        if isinstance(label, str):    
            for data_key, data in self.dataset.items():
                # optional: data is file_path
                if isinstance(data, str):
                    input_is_path = True
                    data = sample_read(data)
                if label in data.columns:
                    main_table = data_key
                    break
            self.y = label
        elif isinstance(label, pd.Series):
            self.y = label.name
        else:
            self.y = None
        if not main_table:
            raise ValueError(f"label {label} is not found in dataset")
        self.main_table = main_table
        
        # Set properties for BasePipeline
        if not input_is_path:
            original_data = self.dataset[main_table]
        else:
            original_data = sample_read(self.dataset[main_table])
        
        self.feature_columns = [i for i in original_data.columns if i != self.y]
        feature_data = original_data[self.feature_columns]
            
        self.generators = []
        self.pipeline = DiGraph()
        if not input_is_path:
            op = 'DataFrame'
            config = main_table
        else:
            op = 'DataLoader'
            config = {'table_name': main_table, 'file_path': self.dataset[main_table]}
        
        cur_id = 0
        self.pipeline[cur_id] = Operation(
            cur_id, None, output = DataFrameSchema(feature_data), op = op, config = config)

        if len(self.dataset) > 1:
            self.supplementary = dict((k, v) for k, v in self.dataset.items() if k != main_table)
        else:
            self.supplementary = None
        self.rdp = None
     
    def fit_analyze(self, *args, **kwargs):
        child = list(self.pipeline.keys())[-1]
        max_id = child
        for i in range(len(self.generators)):
            for generator in self.generators[i]:
                self.pipeline, child, max_id = generator.fit_prepare(self.pipeline, [child], max_id)
        return child, max_id

    def __repr__(self):
        return repr(self.pipeline)

    def export(self, file_path = None):
        json_object = self.pipeline.json_dump()
        if file_path:
            # Writing to sample.json
            with open(file_path, "w") as outfile:
                outfile.write(json_object)
        else:
            print(json_object)
    
    def import_from_json(self, file_path):
        with open(file_path, "r") as infile:
            json_object = json.load(infile)
        for idx, op_config in json_object.items():
            idx = int(idx)
            if idx in self.pipeline:
                continue
            self.pipeline[idx] = Operation.load(idx, op_config)
        #self.create_executable_pipeline()
        
    def create_executable_pipeline(self):
        node_chain = self.pipeline.convert_to_node_chain()
        executable_pipeline = DiGraph()
        executable_sequence = []
        for idx in node_chain:
            executable_pipeline[idx] = self.pipeline[idx].instantiate()
            executable_sequence.append(executable_pipeline[idx])
        return executable_pipeline, executable_sequence
    
    def add_operation(self, config):
        # check if below keys are existing
        # children: who is its child
        # next: who is will use this operation
        # inline_function: the function to process data
        example = """
            def add_one_print(df):
                df['col_1'] = df['col_1'] + 1
                display(df)
                return df
            {
                "children": [7],
                "next": [8],
                "inline_function": add_one_print,
            }"""
        try:
            children = config["children"]
            next = config["next"]
            inline_function = config["inline_function"]
        except:
            raise ValueError(f"please follow this example to create customer operator: {example}")
        if not isinstance(children, list):
            children = [children]
        if not isinstance(next, list):
            next = [next]
        if len(children) > 1 and len(next) > 1:
            raise NotImplementedError("Current we didn't support add operation to case that children and next are all multiple operators.")
        
        # get current max operator id
        max_idx = self.pipeline.get_max_idx()
        cur_idx = max_idx + 1
        
        config = {
            "func_name": inline_function,
        }
        self.pipeline[cur_idx] = Operation(
            cur_idx, children, output = None, op = "custom_operator", config = config)
        for idx in next:
            # replace next's children with new added operator
            children_in_next = self.pipeline[idx].children
            found = {}
            for id, child in enumerate(children_in_next):
                if child in children:
                    found[id] = cur_idx
            for k, v in found.items():
                self.pipeline[idx].children[k] = v
        display(self.plot())

    def delete_operation(self, config):
        example = """
            config = {
                "idx": 7,
                "next": [8]
            }"""
        try:
            cur_idx = config["idx"]
            next = config["next"]
        except:
            raise ValueError(f"please follow this example to create customer operator: {example}")
        if not isinstance(next, list):
            next = [next]
            
        children = self.pipeline[cur_idx].children
        if len(children) > 1 and len(next) > 1:
            raise NotImplementedError("Current we didn't support add operation to case that children and next are all multiple operators.")
        
        if len(next) == 1:
            self.pipeline[next[0]].children = children
        else:            
            for idx in next:
                # replace next's children with new added operator
                children_in_next = self.pipeline[idx].children
                found = {}
                for id, child in enumerate(children_in_next):
                    if child == cur_idx:
                        found[id] = children[0]
                for k, v in found.items():
                    self.pipeline[idx].children[k] = v
        del self.pipeline[cur_idx]
        display(self.plot())        
           
    def plot(self):
        f = graphviz.Digraph()
        edges = []
        nodes = []
        f.attr(fontsize='10')
        def add_escape(input):
            input = input.replace('<', '\<').replace('>', '\>')
            #input = input.replace("'", "\\\'").replace("\"", "\\\"")
            return input

        def add_break(input):
            if isinstance(input, list) and len(input) < 3:
                for line in input:
                    if isinstance(line, str):
                        ret = str(input)
                        return ret
            if isinstance(input, dict):
                input = [f"{k}: {add_break(v)}" for k, v in input.items()]
            if isinstance(input, list):
                try_str = str(input)
                if len(try_str) < 200:
                    return try_str
                ret = "" + "\l"
                for line in input:
                    ret += str(add_break(line)) + "\l"
                return ret
            return input

        for node_id, config in self.pipeline.items():
            nodes.append([str(node_id), f"{node_id}:{config.op} |{add_escape(str(add_break(config.config)))}"])
            if config.children:
                for src_id in config.children:
                    edges.append([str(src_id), str(node_id)])
        for node in nodes:
            f.node(node[0], node[1], shape='record', fontsize='12')
        for edge in edges:
            f.edge(*edge)
        return f  

    def to_chain(self):
        return self.pipeline.convert_to_node_chain()
        
    def execute(self, engine_type = "pandas", no_cache = False):
        # prepare pipeline
        executable_pipeline, executable_sequence = self.create_executable_pipeline()
        print(executable_pipeline)

        # execute
        if engine_type == 'pandas':
            with Timer(f"execute with pandas"):
                for op in executable_sequence:
                    if isinstance(op, DataFrameOperation):
                        op.set(deepcopy(self.dataset))
                    with Timer(f"execute {op}"):
                        op.execute_pd(executable_pipeline, no_cache)
            df = executable_sequence[-1].cache
        elif engine_type == 'spark':
            for op in executable_sequence:
                if isinstance(op, DataFrameOperation):
                    op.set(deepcopy(self.dataset))
                print(f"append {op}")
                op.execute_spark(executable_pipeline, self.rdp, no_cache)
            ret = executable_sequence[-1].cache
            if isinstance(ret, SparkDataFrame):
                with Timer(f"execute with spark"):
                    df = self.rdp.transform(ret)
            elif isinstance(ret, SparkRDD):
                _convert = RDDToDataFrameConverter().get_function(self.rdp)
                with Timer(f"execute with spark"):
                    df = _convert(ret)
            else:
                df = ret
        else:
            raise NotImplementedError('pipeline only support pandas and spark as engine')
        
        # fetch result
        return df

    def fit_transform(self, engine_type = 'pandas', no_cache = False, *args, **kwargs):
        if not no_cache and hasattr(self, 'transformed_cache') and self.transformed_cache is not None:
            print("Detect pre-transformed cache, return cached data")
            print("If re-transform is required, please use fit_transform(no_cache = True)")
            return self.transformed_cache
        if engine_type == "spark":
            self.rdp = SparkDataProcessor()
        ret = self.execute(engine_type, no_cache)
        if engine_type == "spark":
            del self.rdp 
            self.rdp = None
        self.transformed_cache = ret
        return ret
    
    def get_transformed_cache(self):
        if hasattr(self, 'transformed_cache') and self.transformed_cache is not None:
            return self.transformed_cache
        else:
            print("No transformed data detected.")